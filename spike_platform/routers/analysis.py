"""Spike analysis endpoints — computes biomechanical metrics from annotated phases."""

import json
import math
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from spike_platform.database import get_db
from spike_platform.models.db_models import (
    BallDetection,
    Segment,
    SegmentPhase,
    Track,
    TrackFrame,
    Video,
)

router = APIRouter()

# Assumed player height for pixel-to-real-world conversion
PLAYER_HEIGHT_M = 1.83  # 6'0" — typical volleyball player
M_TO_FT = 3.28084
MS_TO_MPH = 2.23694


@router.get("/videos/{video_id}/spike-analysis")
def get_spike_analysis(video_id: str, db: Session = Depends(get_db)):
    """Compute biomechanical metrics per track with phase-annotated spikes."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    fps = video.fps or 30
    vid_height = video.height or 1080
    vid_width = video.width or 1920

    # Find spike segments with human-annotated phases, grouped by track
    spike_segments = (
        db.query(Segment)
        .filter(Segment.video_id == video_id, Segment.human_label == 1)
        .all()
    )

    seg_ids_by_track = defaultdict(list)
    for seg in spike_segments:
        seg_ids_by_track[seg.track_id].append(seg.id)

    # For each track, merge phase boundaries across overlapping segments
    tracks_with_phases = {}
    for track_pk, seg_ids in seg_ids_by_track.items():
        merged = (
            db.query(
                SegmentPhase.phase,
                func.min(SegmentPhase.start_frame).label("start_frame"),
                func.max(SegmentPhase.end_frame).label("end_frame"),
            )
            .filter(
                SegmentPhase.segment_id.in_(seg_ids),
                SegmentPhase.human_label.isnot(None),
            )
            .group_by(SegmentPhase.phase)
            .all()
        )
        if merged:
            phase_dict = {}
            for row in merged:
                phase_dict[row.phase] = {"start": row.start_frame, "end": row.end_frame}
            tracks_with_phases[track_pk] = phase_dict

    results = []
    for track_pk, phase_dict in tracks_with_phases.items():
        track = db.query(Track).filter(Track.id == track_pk).first()
        if not track:
            continue

        all_starts = [p["start"] for p in phase_dict.values()]
        all_ends = [p["end"] for p in phase_dict.values()]
        frame_min = min(all_starts)
        frame_max = max(all_ends)

        track_frames = (
            db.query(TrackFrame)
            .filter(
                TrackFrame.track_id == track_pk,
                TrackFrame.frame_number >= frame_min,
                TrackFrame.frame_number <= frame_max,
            )
            .order_by(TrackFrame.frame_number)
            .all()
        )
        frame_by_num = {tf.frame_number: tf for tf in track_frames}

        # --- Pixel-to-meter conversion from approach bbox height ---
        px_per_meter = None
        ground_y = None
        standing_bbox_height = None
        approach_frames = []
        if "approach" in phase_dict:
            ap = phase_dict["approach"]
            approach_frames = [
                tf for tf in track_frames
                if ap["start"] <= tf.frame_number <= ap["end"]
            ]
            if approach_frames:
                standing_bbox_height = sum(
                    (tf.bbox_y2 - tf.bbox_y1) for tf in approach_frames
                ) / len(approach_frames)
                if standing_bbox_height > 0:
                    px_per_meter = standing_bbox_height / PLAYER_HEIGHT_M
                ground_y = max(tf.bbox_y2 for tf in approach_frames)

        # Fallback: estimate from land phase or first track frames
        if px_per_meter is None:
            fallback = []
            if "land" in phase_dict:
                lp = phase_dict["land"]
                fallback = [
                    tf for tf in track_frames
                    if lp["start"] <= tf.frame_number <= lp["end"]
                ]
            if not fallback:
                fallback = track_frames[:20]
            if fallback:
                standing_bbox_height = sum(
                    (tf.bbox_y2 - tf.bbox_y1) for tf in fallback
                ) / len(fallback)
                if standing_bbox_height > 0:
                    px_per_meter = standing_bbox_height / PLAYER_HEIGHT_M
                ground_y = max(tf.bbox_y2 for tf in fallback)

        # Key frames for 3-frame display
        key_frames = {}
        if "approach" in phase_dict:
            key_frames["approach_end"] = phase_dict["approach"]["end"]
        if "jump" in phase_dict:
            key_frames["jump_peak"] = phase_dict["jump"]["end"]

        # --- Jump height ---
        jump_height = None
        if "jump" in phase_dict:
            jp = phase_dict["jump"]
            jump_frames = [
                tf for tf in track_frames
                if jp["start"] <= tf.frame_number <= jp["end"]
            ]
            if jump_frames:
                # Standing baseline: prefer approach bbox center (ground-level)
                if approach_frames:
                    standing_y = sum(
                        (tf.bbox_y1 + tf.bbox_y2) / 2 for tf in approach_frames
                    ) / len(approach_frames)
                else:
                    first = jump_frames[0]
                    standing_y = (first.bbox_y1 + first.bbox_y2) / 2
                peak_frame_tf = min(jump_frames, key=lambda tf: tf.bbox_y2)
                peak_y = (peak_frame_tf.bbox_y1 + peak_frame_tf.bbox_y2) / 2
                height_px = standing_y - peak_y
                key_frames["jump_peak"] = peak_frame_tf.frame_number

                jump_height = {
                    "peak_frame": peak_frame_tf.frame_number,
                    "height_px": round(height_px, 1),
                    "standing_y": round(standing_y, 1),
                    "peak_y": round(peak_y, 1),
                }
                if px_per_meter and px_per_meter > 0:
                    height_m = height_px / px_per_meter
                    jump_height["height_m"] = round(height_m, 2)
                    jump_height["height_ft"] = round(height_m * M_TO_FT, 2)

        # --- Ball contact (velocity-assisted: find max acceleration frame) ---
        ball_contact = None
        if "swing" in phase_dict:
            sp = phase_dict["swing"]
            ball_dets = (
                db.query(BallDetection)
                .filter(
                    BallDetection.video_id == video_id,
                    BallDetection.frame_number >= sp["start"],
                    BallDetection.frame_number <= sp["end"],
                )
                .order_by(BallDetection.frame_number)
                .all()
            )
            if ball_dets:
                best_det = None
                best_wrist = None
                best_dist = float("inf")

                # Try velocity-based detection first (need >=3 detections)
                if len(ball_dets) >= 3:
                    centers = [
                        ((bd.bbox_x1 + bd.bbox_x2) / 2,
                         (bd.bbox_y1 + bd.bbox_y2) / 2,
                         bd.frame_number)
                        for bd in ball_dets
                    ]
                    # Compute velocity between consecutive detections
                    vels = []
                    for i in range(1, len(centers)):
                        dt = centers[i][2] - centers[i - 1][2]
                        if dt <= 0:
                            vels.append((0, 0))
                            continue
                        vx = (centers[i][0] - centers[i - 1][0]) / dt
                        vy = (centers[i][1] - centers[i - 1][1]) / dt
                        vels.append((vx, vy))
                    # Compute acceleration (velocity change) — max = contact
                    max_accel = 0
                    accel_idx = None
                    for i in range(1, len(vels)):
                        ax = vels[i][0] - vels[i - 1][0]
                        ay = vels[i][1] - vels[i - 1][1]
                        accel = math.sqrt(ax ** 2 + ay ** 2)
                        if accel > max_accel:
                            max_accel = accel
                            accel_idx = i + 1  # index into ball_dets
                    if accel_idx is not None and accel_idx < len(ball_dets):
                        best_det = ball_dets[accel_idx]

                # Fall back to proximity-based detection
                if best_det is None:
                    for bd in ball_dets:
                        ball_cx = (bd.bbox_x1 + bd.bbox_x2) / 2
                        ball_cy = (bd.bbox_y1 + bd.bbox_y2) / 2
                        tf = frame_by_num.get(bd.frame_number)
                        if not tf:
                            continue
                        for wx, wy in [
                            (tf.wrist_r_x, tf.wrist_r_y),
                            (tf.wrist_l_x, tf.wrist_l_y),
                        ]:
                            if wx is None or wy is None:
                                continue
                            dist = math.sqrt(
                                (ball_cx - wx) ** 2 + (ball_cy - wy) ** 2
                            )
                            if dist < best_dist:
                                best_dist = dist
                                best_det = bd
                                best_wrist = (wx, wy)

                # Find closest wrist for the chosen contact frame
                if best_det and best_wrist is None:
                    ball_cx = (best_det.bbox_x1 + best_det.bbox_x2) / 2
                    ball_cy = (best_det.bbox_y1 + best_det.bbox_y2) / 2
                    tf = frame_by_num.get(best_det.frame_number)
                    if tf:
                        for wx, wy in [
                            (tf.wrist_r_x, tf.wrist_r_y),
                            (tf.wrist_l_x, tf.wrist_l_y),
                        ]:
                            if wx is None or wy is None:
                                continue
                            dist = math.sqrt(
                                (ball_cx - wx) ** 2 + (ball_cy - wy) ** 2
                            )
                            if dist < best_dist:
                                best_dist = dist
                                best_wrist = (wx, wy)

                if best_det:
                    ball_cx = (best_det.bbox_x1 + best_det.bbox_x2) / 2
                    ball_cy = (best_det.bbox_y1 + best_det.bbox_y2) / 2
                    key_frames["contact"] = best_det.frame_number
                    ball_contact = {
                        "contact_frame": best_det.frame_number,
                        "ball_center": [round(ball_cx, 1), round(ball_cy, 1)],
                    }
                    if best_wrist:
                        ball_contact["wrist"] = [
                            round(best_wrist[0], 1), round(best_wrist[1], 1),
                        ]
                        ball_contact["distance_px"] = round(best_dist, 1)

        # --- Reach height (highest wrist position during swing) ---
        reach_height = None
        if "swing" in phase_dict:
            sp = phase_dict["swing"]
            min_wrist_y = None
            reach_frame = None
            for tf in track_frames:
                if sp["start"] <= tf.frame_number <= sp["end"]:
                    for wy in [tf.wrist_r_y, tf.wrist_l_y]:
                        if wy is not None and (min_wrist_y is None or wy < min_wrist_y):
                            min_wrist_y = wy
                            reach_frame = tf.frame_number
            if min_wrist_y is not None and px_per_meter and ground_y is not None:
                reach_px = ground_y - min_wrist_y
                reach_m = reach_px / px_per_meter
                reach_height = {
                    "reach_frame": reach_frame,
                    "reach_height_m": round(reach_m, 2),
                    "reach_height_ft": round(reach_m * M_TO_FT, 2),
                }

        # --- Approach step count ---
        approach = None
        if "approach" in phase_dict:
            ap = phase_dict["approach"]
            # Detect steps via ankle Y-position crossovers in pose features
            approach_steps = None
            if approach_frames:
                diffs = []
                for tf in approach_frames:
                    if tf.pose_features:
                        pf = json.loads(tf.pose_features)
                        if len(pf) >= 32:
                            r_ankle_y = pf[19]  # right ankle Y (pelvis-centered)
                            l_ankle_y = pf[31]  # left ankle Y
                            diffs.append(r_ankle_y - l_ankle_y)
                # Hysteresis step counter — require signal to cross ±threshold
                STEP_THRESH = 0.3  # torso-lengths
                steps = 0
                state = "neutral"  # neutral | right_forward | left_forward
                for d in diffs:
                    if state == "neutral":
                        if d > STEP_THRESH:
                            state = "right_forward"
                            steps += 1
                        elif d < -STEP_THRESH:
                            state = "left_forward"
                            steps += 1
                    elif state == "right_forward":
                        if d < -STEP_THRESH:
                            state = "left_forward"
                            steps += 1
                    elif state == "left_forward":
                        if d > STEP_THRESH:
                            state = "right_forward"
                            steps += 1
                if steps > 0:
                    approach_steps = steps
            approach = {
                "start_frame": ap["start"],
                "end_frame": ap["end"],
                "steps": approach_steps,
            }

        # --- Arm swing speed ---
        arm_swing = None
        if "swing" in phase_dict:
            sp = phase_dict["swing"]
            swing_frames = sorted(
                [tf for tf in track_frames if sp["start"] <= tf.frame_number <= sp["end"]],
                key=lambda tf: tf.frame_number,
            )
            if len(swing_frames) >= 2:
                speeds_r = []
                speeds_l = []
                for i in range(1, len(swing_frames)):
                    prev, curr = swing_frames[i - 1], swing_frames[i]
                    dt = (curr.frame_number - prev.frame_number) / fps
                    if dt <= 0:
                        speeds_r.append(0)
                        speeds_l.append(0)
                        continue
                    if prev.wrist_r_x is not None and curr.wrist_r_x is not None:
                        dx = curr.wrist_r_x - prev.wrist_r_x
                        dy = curr.wrist_r_y - prev.wrist_r_y
                        speeds_r.append(math.sqrt(dx**2 + dy**2) / dt)
                    else:
                        speeds_r.append(0)
                    if prev.wrist_l_x is not None and curr.wrist_l_x is not None:
                        dx = curr.wrist_l_x - prev.wrist_l_x
                        dy = curr.wrist_l_y - prev.wrist_l_y
                        speeds_l.append(math.sqrt(dx**2 + dy**2) / dt)
                    else:
                        speeds_l.append(0)

                max_r = max(speeds_r) if speeds_r else 0
                max_l = max(speeds_l) if speeds_l else 0
                speeds = speeds_r if max_r >= max_l else speeds_l

                if speeds and max(speeds) > 0:
                    peak_idx = speeds.index(max(speeds))
                    frame_count = sp["end"] - sp["start"] + 1
                    peak_speed_px = max(speeds)
                    arm_swing = {
                        "start_frame": sp["start"],
                        "end_frame": sp["end"],
                        "peak_speed_px_per_sec": round(peak_speed_px, 1),
                        "peak_frame": swing_frames[peak_idx + 1].frame_number,
                        "duration_sec": round(frame_count / fps, 3),
                    }
                    if px_per_meter and px_per_meter > 0:
                        peak_speed_ms = peak_speed_px / px_per_meter
                        arm_swing["peak_speed_ms"] = round(peak_speed_ms, 2)
                        arm_swing["peak_speed_mph"] = round(
                            peak_speed_ms * MS_TO_MPH, 1
                        )

        # --- Arm swing range (total wrist arc length during swing) ---
        arm_swing_range = None
        if "swing" in phase_dict:
            sp = phase_dict["swing"]
            swing_frames_r = sorted(
                [tf for tf in track_frames if sp["start"] <= tf.frame_number <= sp["end"]],
                key=lambda tf: tf.frame_number,
            )
            if len(swing_frames_r) >= 2:
                dist_r = 0.0
                dist_l = 0.0
                for i in range(1, len(swing_frames_r)):
                    prev, curr = swing_frames_r[i - 1], swing_frames_r[i]
                    if prev.wrist_r_x is not None and curr.wrist_r_x is not None:
                        dist_r += math.sqrt(
                            (curr.wrist_r_x - prev.wrist_r_x) ** 2
                            + (curr.wrist_r_y - prev.wrist_r_y) ** 2
                        )
                    if prev.wrist_l_x is not None and curr.wrist_l_x is not None:
                        dist_l += math.sqrt(
                            (curr.wrist_l_x - prev.wrist_l_x) ** 2
                            + (curr.wrist_l_y - prev.wrist_l_y) ** 2
                        )
                range_px = max(dist_r, dist_l)
                if range_px > 0:
                    arm_swing_range = {
                        "start_frame": sp["start"],
                        "end_frame": sp["end"],
                        "range_px": round(range_px, 1),
                    }
                    if px_per_meter and px_per_meter > 0:
                        range_m = range_px / px_per_meter
                        arm_swing_range["range_m"] = round(range_m, 2)
                        arm_swing_range["range_ft"] = round(range_m * M_TO_FT, 2)

        # --- Ball detections + track bboxes for playback overlay ---
        spike_ball_dets = (
            db.query(BallDetection)
            .filter(
                BallDetection.video_id == video_id,
                BallDetection.frame_number >= frame_min,
                BallDetection.frame_number <= frame_max,
            )
            .order_by(BallDetection.frame_number)
            .all()
        )
        ball_overlay = [
            {"frame": bd.frame_number, "bbox": [bd.bbox_x1, bd.bbox_y1, bd.bbox_x2, bd.bbox_y2]}
            for bd in spike_ball_dets
        ]
        track_bboxes = [
            {"frame": tf.frame_number, "bbox": [tf.bbox_x1, tf.bbox_y1, tf.bbox_x2, tf.bbox_y2]}
            for tf in track_frames
        ]

        results.append({
            "track_id": track.track_id,
            "px_per_meter": round(px_per_meter, 1) if px_per_meter else None,
            "phases": phase_dict,
            "key_frames": key_frames,
            "jump_height": jump_height,
            "ball_contact": ball_contact,
            "reach_height": reach_height,
            "approach": approach,
            "arm_swing": arm_swing,
            "arm_swing_range": arm_swing_range,
            "ball_detections": ball_overlay,
            "track_bboxes": track_bboxes,
        })

    return {
        "video_id": video_id,
        "fps": fps,
        "width": vid_width,
        "height": vid_height,
        "tracks": results,
    }

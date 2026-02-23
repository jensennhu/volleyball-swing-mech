"""
Video processing pipeline orchestrator.

Runs the full pipeline: detect → track → pose → segment → auto-infer.
Called as a background job after video upload.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from spike_platform.config import settings
from spike_platform.database import SessionLocal
from spike_platform.models.db_models import (
    Video, Track, TrackFrame, Segment, BallDetection, TrainingRun,
)
from spike_platform.services.detection import (
    PersonDetector, TrackResult, FrameDetection, DetectionOutput,
)
from spike_platform.services.pose import PoseService
from spike_platform.services.segmentation import create_segments_for_track, SegmentData
from spike_platform.services.reid import ReIDEncoder
from spike_platform.services.track_postprocess import (
    detect_id_switches,
    extract_track_embeddings,
    merge_broken_tracks,
)
from spike_platform.services.track_classifier import classify_tracks
from spike_platform.services.ball_tracking import track_balls
from spike_platform.services.context_features import (
    compute_vertical_motion_features,
    compute_ball_context_features,
)

logger = logging.getLogger(__name__)


def process_video_pipeline(
    video_id: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """
    Full processing pipeline for an uploaded video.

    Steps:
        1. Run person detection + tracking (YOLOv8 + ByteTrack)
        2. For each tracked person, extract pose features per frame
        3. Create sliding window segments
        4. Save segments and features to DB + disk

    Called by BackgroundWorker.submit().
    """
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            return

        # Verify video file exists before starting
        if not Path(video.filepath).exists():
            video.status = "error"
            video.error_message = f"Video file not found: {video.filepath}"
            db.commit()
            return

        video.status = "processing"
        db.commit()

        if progress_callback:
            progress_callback(5.0, "Starting detection and tracking...")

        # Step 1: Person detection + tracking (+ ball detection)
        logger.info(f"[{video_id[:8]}] Starting detection...")
        detector = PersonDetector()
        detection_output = detector.process_video(
            video.filepath,
            progress_callback=lambda pct, msg: (
                progress_callback(5 + pct * 0.35, msg) if progress_callback else None
            ),
        )
        track_results = detection_output.person_tracks
        ball_raw = detection_output.ball_detections

        # Post-process: extract ReID embeddings and split tracks at ID switches
        if progress_callback:
            progress_callback(38.0, "Extracting appearance embeddings for tracks...")
        encoder = ReIDEncoder()
        embeddings = extract_track_embeddings(video.filepath, track_results, encoder)
        track_results = detect_id_switches(track_results, embeddings)
        track_results = merge_broken_tracks(track_results, embeddings)

        # Step 1b: Ball tracking (nearest-neighbor linking)
        if progress_callback:
            progress_callback(40.0, f"Tracking ball across {len(ball_raw)} detections...")
        tracked_balls = track_balls(ball_raw)
        # Save ball detections to DB
        for ball_det, ball_tid in tracked_balls:
            db.add(BallDetection(
                video_id=video_id,
                frame_number=ball_det.frame_number,
                bbox_x1=ball_det.bbox[0],
                bbox_y1=ball_det.bbox[1],
                bbox_x2=ball_det.bbox[2],
                bbox_y2=ball_det.bbox[3],
                confidence=ball_det.confidence,
                ball_track_id=ball_tid,
            ))
        db.commit()
        logger.info(f"[{video_id[:8]}] Ball tracking: {len(tracked_balls)} detections across frames.")

        # Filter to tracks with enough frames
        long_tracks = [t for t in track_results if t.frame_count >= settings.MIN_TRACK_FRAMES]
        total_frames = sum(t.frame_count for t in long_tracks)
        logger.info(
            f"[{video_id[:8]}] Detection done. "
            f"{len(track_results)} total tracks, {len(long_tracks)} with >= {settings.MIN_TRACK_FRAMES} frames, "
            f"{total_frames} frames to process."
        )

        if progress_callback:
            progress_callback(45.0, f"Found {len(long_tracks)} tracks ({total_frames} frames). Extracting poses...")

        # Step 2: Extract pose features via single sequential video read.
        # Build frame→detections index so we read each frame exactly once,
        # processing all tracks that need it in one pass.
        pose_service = PoseService()

        db_tracks = []
        for track_result in long_tracks:
            db_track = Track(
                video_id=video_id,
                track_id=track_result.track_id,
                start_frame=track_result.start_frame,
                end_frame=track_result.end_frame,
                frame_count=track_result.frame_count,
                avg_confidence=track_result.avg_confidence,
            )
            db.add(db_track)
            db_tracks.append(db_track)
        db.flush()  # assign IDs to all tracks

        # Map frame_number → list of (track_index, FrameDetection)
        frame_to_dets: dict[int, list[tuple[int, FrameDetection]]] = {}
        for i, track_result in enumerate(long_tracks):
            for det in track_result.frames:
                frame_to_dets.setdefault(det.frame_number, []).append((i, det))
        needed_frames = sorted(frame_to_dets.keys())

        cap = cv2.VideoCapture(video.filepath)
        current_frame = 0
        frames_done = 0
        needed_idx = 0

        while needed_idx < len(needed_frames):
            target = needed_frames[needed_idx]

            # Skip unneeded frames sequentially (grab is fast — no pixel decode)
            while current_frame < target:
                cap.grab()
                current_frame += 1

            ret, frame = cap.read()
            current_frame += 1
            if not ret:
                break

            # Process all tracks that need this frame
            for track_idx, det in frame_to_dets[target]:
                features, pose_conf, wrist_r, wrist_l = pose_service.extract_features_and_wrists(
                    frame, det.bbox
                )
                tf = TrackFrame(
                    track_id=db_tracks[track_idx].id,
                    frame_number=det.frame_number,
                    bbox_x1=det.bbox[0],
                    bbox_y1=det.bbox[1],
                    bbox_x2=det.bbox[2],
                    bbox_y2=det.bbox[3],
                    detection_confidence=det.confidence,
                    pose_features=json.dumps(features.tolist()) if features is not None else None,
                    pose_confidence=pose_conf if features is not None else None,
                    wrist_r_x=wrist_r[0] if features is not None else None,
                    wrist_r_y=wrist_r[1] if features is not None else None,
                    wrist_l_x=wrist_l[0] if features is not None else None,
                    wrist_l_y=wrist_l[1] if features is not None else None,
                )
                db.add(tf)
                frames_done += 1

            needed_idx += 1

            # Commit and report progress periodically
            if needed_idx % 100 == 0 or needed_idx == len(needed_frames):
                db.commit()
                if progress_callback:
                    pose_pct = (frames_done / total_frames) * 30 if total_frames > 0 else 0
                    progress_callback(
                        45 + pose_pct,
                        f"Pose extraction: {frames_done}/{total_frames} frames ({needed_idx}/{len(needed_frames)} unique)",
                    )

        db.commit()
        cap.release()
        pose_service.close()

        logger.info(f"[{video_id[:8]}] Pose extraction done. Classifying tracks...")

        # Step 2.5: Classify tracks as player/non_player
        role_counts = classify_tracks(video_id, db)
        logger.info(
            f"[{video_id[:8]}] Track roles: {role_counts['player']} players, "
            f"{role_counts['non_player']} non-players"
        )

        if progress_callback:
            progress_callback(75.0, f"Creating segments for {len(db_tracks)} tracks...")

        # Get video dimensions for context feature normalization
        cap2 = cv2.VideoCapture(video.filepath)
        video_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap2.release()

        # Load ball detections indexed by frame for context features
        ball_db_rows = (
            db.query(BallDetection)
            .filter(BallDetection.video_id == video_id)
            .order_by(BallDetection.frame_number)
            .all()
        )
        ball_by_frame: dict[int, BallDetection] = {}
        for bd in ball_db_rows:
            # Keep highest-confidence detection per frame
            if bd.frame_number not in ball_by_frame or bd.confidence > ball_by_frame[bd.frame_number].confidence:
                ball_by_frame[bd.frame_number] = bd

        # Step 3: Create segments via sliding window with v2 context features
        feature_version = settings.FEATURE_VERSION
        total_segments = 0
        for db_track in db_tracks:
            track_frames = (
                db.query(TrackFrame)
                .filter(TrackFrame.track_id == db_track.id)
                .order_by(TrackFrame.frame_number)
                .all()
            )

            # Compute context features for this track
            vertical_feats = compute_vertical_motion_features(track_frames, video_height)
            ball_feats = compute_ball_context_features(
                track_frames, ball_by_frame, video_width, video_height,
            )

            segment_datas = create_segments_for_track(
                track_frames,
                vertical_features=vertical_feats,
                ball_features=ball_feats,
            )

            for seg_data in segment_datas:
                features_path = _save_segment_features(video_id, db_track.id, seg_data)

                segment = Segment(
                    track_id=db_track.id,
                    video_id=video_id,
                    start_frame=seg_data.start_frame,
                    end_frame=seg_data.end_frame,
                    window_size=settings.WINDOW_SIZE,
                    features_path=str(features_path),
                    feature_version=feature_version,
                )
                db.add(segment)
                total_segments += 1

        db.commit()

        video.status = "processed"
        db.commit()

        logger.info(f"[{video_id[:8]}] Processing complete. {len(db_tracks)} tracks, {total_segments} segments.")

        if progress_callback:
            progress_callback(80.0, f"Processed: {len(db_tracks)} tracks, {total_segments} segments. Running inference...")

        # Step 4: Auto-inference with latest trained models
        _auto_infer(video_id, progress_callback)

        if progress_callback:
            progress_callback(100.0, f"Done. {len(db_tracks)} tracks, {total_segments} segments.")

    except Exception as e:
        logger.exception(f"[{video_id[:8]}] Processing failed: {e}")
        try:
            db.rollback()
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.status = "error"
                video.error_message = str(e)[:500]
                db.commit()
        except Exception:
            logger.exception(f"[{video_id[:8]}] Failed to update error status")
    finally:
        db.close()


def _auto_infer(
    video_id: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """Run spike detection + phase classification using latest trained models."""
    db = SessionLocal()
    try:
        # Spike detection inference
        spike_run = (
            db.query(TrainingRun)
            .filter(
                TrainingRun.task_type == "spike_detection",
                TrainingRun.status == "completed",
            )
            .order_by(TrainingRun.completed_at.desc())
            .first()
        )
        if spike_run:
            if progress_callback:
                progress_callback(82.0, "Auto-inference: spike detection...")
            try:
                from spike_platform.services.inference import run_inference_on_video
                run_inference_on_video(video_id, spike_run.id)
                logger.info(f"[{video_id[:8]}] Auto spike inference done (run {spike_run.id}).")
            except Exception as e:
                logger.warning(f"[{video_id[:8]}] Auto spike inference failed: {e}")
        else:
            logger.info(f"[{video_id[:8]}] No spike detection model available, skipping auto-inference.")

        # Phase classification inference
        phase_run = (
            db.query(TrainingRun)
            .filter(
                TrainingRun.task_type == "phase_classification",
                TrainingRun.status == "completed",
            )
            .order_by(TrainingRun.completed_at.desc())
            .first()
        )
        if phase_run:
            if progress_callback:
                progress_callback(92.0, "Auto-inference: phase classification...")
            try:
                from spike_platform.services.phase_inference import run_phase_inference
                run_phase_inference(video_id, phase_run.id)
                logger.info(f"[{video_id[:8]}] Auto phase inference done (run {phase_run.id}).")
            except Exception as e:
                logger.warning(f"[{video_id[:8]}] Auto phase inference failed: {e}")
        else:
            logger.info(f"[{video_id[:8]}] No phase classification model available, skipping.")

    finally:
        db.close()


def _save_segment_features(
    video_id: str, track_id: int, seg_data: SegmentData
) -> Path:
    """Save segment feature matrix as .npy file."""
    features_dir = settings.FEATURES_DIR / video_id
    features_dir.mkdir(parents=True, exist_ok=True)

    filename = f"t{track_id}_f{seg_data.start_frame}-{seg_data.end_frame}.npy"
    filepath = features_dir / filename
    np.save(str(filepath), seg_data.features)
    return filepath

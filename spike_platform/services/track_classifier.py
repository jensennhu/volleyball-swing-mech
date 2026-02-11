"""
Heuristic track role classification: player vs non-player.

Uses bounding box area and pose confidence to distinguish active players
from referees, audience members, and background people.
"""

import logging
from statistics import median

import numpy as np
from sqlalchemy.orm import Session

from spike_platform.config import settings
from spike_platform.models.db_models import Track, TrackFrame, Video

logger = logging.getLogger(__name__)


def compute_track_stats(track_id: int, db: Session, video_width: int, video_height: int) -> dict:
    """Compute per-track statistics for classification.

    Returns dict with median_bbox_area (normalized), median_pose_confidence,
    median_y_position (normalized).
    """
    frames = (
        db.query(TrackFrame)
        .filter(TrackFrame.track_id == track_id)
        .all()
    )

    if not frames:
        return {"median_bbox_area": 0.0, "median_pose_confidence": 0.0, "median_y_position": 0.5}

    frame_area = video_width * video_height if video_width and video_height else 1.0

    bbox_areas = []
    pose_confs = []
    y_positions = []

    for f in frames:
        w = f.bbox_x2 - f.bbox_x1
        h = f.bbox_y2 - f.bbox_y1
        bbox_areas.append((w * h) / frame_area)
        y_positions.append(((f.bbox_y1 + f.bbox_y2) / 2) / video_height if video_height else 0.5)
        if f.pose_confidence is not None:
            pose_confs.append(f.pose_confidence)

    return {
        "median_bbox_area": float(median(bbox_areas)) if bbox_areas else 0.0,
        "median_pose_confidence": float(median(pose_confs)) if pose_confs else 0.0,
        "median_y_position": float(median(y_positions)) if y_positions else 0.5,
    }


def classify_tracks(video_id: str, db: Session) -> dict:
    """Auto-classify tracks in a video as player or non_player.

    Uses per-video relative thresholds (camera angles vary) combined
    with absolute pose confidence thresholds.

    Preserves tracks where role_source='human' (manual overrides).

    Returns dict with classification counts.
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        return {"player": 0, "non_player": 0, "skipped": 0}

    tracks = db.query(Track).filter(Track.video_id == video_id).all()
    if not tracks:
        return {"player": 0, "non_player": 0, "skipped": 0}

    video_w = video.width or 1920
    video_h = video.height or 1080

    # Compute stats for all tracks
    track_stats = {}
    for track in tracks:
        track_stats[track.id] = compute_track_stats(track.id, db, video_w, video_h)

    # Compute video-wide median bbox area for relative thresholds
    all_areas = [s["median_bbox_area"] for s in track_stats.values() if s["median_bbox_area"] > 0]
    video_median_area = float(median(all_areas)) if all_areas else 0.0

    area_threshold = video_median_area * settings.TRACK_ROLE_BBOX_AREA_RATIO
    pose_threshold = settings.TRACK_ROLE_POSE_CONF_THRESHOLD

    counts = {"player": 0, "non_player": 0, "skipped": 0}

    for track in tracks:
        # Preserve human overrides
        if track.role_source == "human":
            counts["skipped"] += 1
            continue

        stats = track_stats[track.id]
        is_small = stats["median_bbox_area"] < area_threshold
        is_low_pose = stats["median_pose_confidence"] < pose_threshold

        if is_small and is_low_pose:
            track.role = "non_player"
        elif stats["median_bbox_area"] < video_median_area * 0.3:
            # Very small relative to video — almost certainly not a player
            track.role = "non_player"
        else:
            track.role = "player"

        track.role_source = "heuristic"
        counts[track.role] += 1

    db.commit()

    logger.info(
        f"Track classification for video {video_id}: "
        f"{counts['player']} players, {counts['non_player']} non-players, "
        f"{counts['skipped']} human-labeled (preserved)"
    )

    return counts

"""
Sliding window segmentation over tracked person timelines.

Creates fixed-length feature windows (default 40 frames)
as the unit of labeling and prediction.
"""

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from spike_platform.config import settings
from spike_platform.models.db_models import TrackFrame


@dataclass
class SegmentData:
    """A proposed segment with its feature matrix."""
    start_frame: int
    end_frame: int
    features: np.ndarray  # shape (window_size, feature_dim)


def create_segments_for_track(
    track_frames: list[TrackFrame],
    window_size: int = None,
    stride: int = None,
    max_gap: int = 8,
    vertical_features: dict[int, np.ndarray] = None,
    ball_features: dict[int, np.ndarray] = None,
) -> list[SegmentData]:
    """
    Create overlapping sliding window segments from a track's frame data.

    Args:
        track_frames: Ordered list of TrackFrame ORM objects for one track.
        window_size: Number of frames per segment (default: settings.WINDOW_SIZE).
        stride: Step between windows (default: settings.WINDOW_STRIDE).
        max_gap: Maximum allowed gap in frame numbers within a window.
        vertical_features: Optional {frame_number: np.array(3,)} from context_features.
        ball_features: Optional {frame_number: np.array(4,)} from context_features.

    Returns:
        List of SegmentData. Feature dim is 33 (v1) or 40 (v2) depending on
        whether context feature dicts are provided.
    """
    window_size = window_size or settings.WINDOW_SIZE
    stride = stride or settings.WINDOW_STRIDE

    use_context = vertical_features is not None and ball_features is not None

    # Filter to frames that have valid pose features
    valid_frames = []
    for tf in track_frames:
        if tf.pose_features is None:
            continue
        try:
            feats = json.loads(tf.pose_features)
            if len(feats) == settings.FEATURE_DIM:
                valid_frames.append((tf.frame_number, feats))
        except (json.JSONDecodeError, TypeError):
            continue

    if len(valid_frames) < window_size:
        return []

    # Sort by frame number
    valid_frames.sort(key=lambda x: x[0])

    # Default context vectors when missing for a frame
    default_vertical = np.zeros(3, dtype=np.float32)
    default_ball = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    segments = []
    for i in range(0, len(valid_frames) - window_size + 1, stride):
        window = valid_frames[i : i + window_size]

        # Check for large gaps in frame continuity
        frame_nums = [f[0] for f in window]
        max_frame_gap = max(
            frame_nums[j + 1] - frame_nums[j] for j in range(len(frame_nums) - 1)
        )
        if max_frame_gap > max_gap:
            continue

        if use_context:
            # Build 40-dim features: [pose_33 | vertical_3 | ball_4]
            rows = []
            for fn, pose_feats in window:
                pose = np.array(pose_feats, dtype=np.float32)
                vert = vertical_features.get(fn, default_vertical)
                ball = ball_features.get(fn, default_ball)
                rows.append(np.concatenate([pose, vert, ball]))
            features = np.stack(rows)
        else:
            features = np.array([f[1] for f in window], dtype=np.float32)

        segments.append(
            SegmentData(
                start_frame=frame_nums[0],
                end_frame=frame_nums[-1],
                features=features,
            )
        )

    return segments

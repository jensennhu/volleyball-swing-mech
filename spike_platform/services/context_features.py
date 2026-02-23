"""
Context feature computation for v2 feature vectors.

Computes per-frame features that provide scene context beyond pure pose:
  - Vertical motion (3 dims): bbox center Y, Y velocity, bbox height ratio
  - Ball context (4 dims): ball present, ball-hand distance, ball velocity X/Y
"""

import math

import numpy as np

from spike_platform.models.db_models import TrackFrame, BallDetection


def compute_vertical_motion_features(
    track_frames: list[TrackFrame],
    video_height: int,
) -> dict[int, np.ndarray]:
    """
    Compute per-frame vertical motion features from bbox data.

    Returns:
        {frame_number: np.array([center_y_norm, y_velocity, height_ratio])}
    """
    if video_height <= 0:
        video_height = 1

    # Sort by frame number
    sorted_frames = sorted(track_frames, key=lambda tf: tf.frame_number)

    result: dict[int, np.ndarray] = {}
    prev_center_y = None

    for tf in sorted_frames:
        center_y = (tf.bbox_y1 + tf.bbox_y2) / 2.0
        center_y_norm = center_y / video_height
        height_ratio = (tf.bbox_y2 - tf.bbox_y1) / video_height

        if prev_center_y is not None:
            y_velocity = center_y_norm - prev_center_y
        else:
            y_velocity = 0.0

        result[tf.frame_number] = np.array(
            [center_y_norm, y_velocity, height_ratio], dtype=np.float32
        )
        prev_center_y = center_y_norm

    return result


def compute_ball_context_features(
    track_frames: list[TrackFrame],
    ball_detections_by_frame: dict[int, BallDetection],
    video_width: int,
    video_height: int,
) -> dict[int, np.ndarray]:
    """
    Compute per-frame ball context features.

    Returns:
        {frame_number: np.array([ball_present, ball_dist_hands, ball_vx, ball_vy])}

    When no ball is detected: [0, 1.0, 0, 0]
    """
    if video_width <= 0:
        video_width = 1
    if video_height <= 0:
        video_height = 1

    frame_diagonal = math.hypot(video_width, video_height)

    sorted_frames = sorted(track_frames, key=lambda tf: tf.frame_number)

    result: dict[int, np.ndarray] = {}
    prev_ball_cx = None
    prev_ball_cy = None
    prev_ball_frame = None

    for tf in sorted_frames:
        ball = ball_detections_by_frame.get(tf.frame_number)

        if ball is None:
            result[tf.frame_number] = np.array(
                [0.0, 1.0, 0.0, 0.0], dtype=np.float32
            )
            continue

        ball_cx = (ball.bbox_x1 + ball.bbox_x2) / 2.0
        ball_cy = (ball.bbox_y1 + ball.bbox_y2) / 2.0

        # Ball-hand distance (min distance to either wrist)
        wrist_r = (tf.wrist_r_x, tf.wrist_r_y)
        wrist_l = (tf.wrist_l_x, tf.wrist_l_y)

        dist_r = math.hypot(ball_cx - wrist_r[0], ball_cy - wrist_r[1]) if wrist_r[0] else frame_diagonal
        dist_l = math.hypot(ball_cx - wrist_l[0], ball_cy - wrist_l[1]) if wrist_l[0] else frame_diagonal
        min_dist = min(dist_r, dist_l)
        ball_dist_norm = min(min_dist / frame_diagonal, 1.0)

        # Ball velocity
        if prev_ball_cx is not None and prev_ball_frame is not None:
            frame_gap = tf.frame_number - prev_ball_frame
            if frame_gap > 0 and frame_gap <= 5:
                ball_vx = (ball_cx - prev_ball_cx) / video_width / frame_gap
                ball_vy = (ball_cy - prev_ball_cy) / video_height / frame_gap
            else:
                ball_vx = 0.0
                ball_vy = 0.0
        else:
            ball_vx = 0.0
            ball_vy = 0.0

        result[tf.frame_number] = np.array(
            [1.0, ball_dist_norm, ball_vx, ball_vy], dtype=np.float32
        )

        prev_ball_cx = ball_cx
        prev_ball_cy = ball_cy
        prev_ball_frame = tf.frame_number

    return result

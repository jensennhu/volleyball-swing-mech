"""
Simple nearest-neighbor ball tracker.

Links ball detections across frames by proximity (bbox center distance).
Typically only 1 ball visible at a time, so this degenerates to simple
frame-to-frame linking.
"""

import math
from dataclasses import dataclass

from spike_platform.services.detection import BallFrameDetection


@dataclass
class _ActiveTrack:
    track_id: int
    last_frame: int
    last_cx: float
    last_cy: float


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def track_balls(
    raw_detections: list[BallFrameDetection],
    max_distance: float = 150.0,
    max_gap_frames: int = 10,
) -> list[tuple[BallFrameDetection, int]]:
    """
    Assign a ball_track_id to each detection via nearest-neighbor linking.

    Args:
        raw_detections: Ball detections sorted by frame_number.
        max_distance: Max pixel distance to link to an existing track.
        max_gap_frames: Max frames a track can be inactive before closing.

    Returns:
        List of (detection, ball_track_id) tuples.
    """
    if not raw_detections:
        return []

    active_tracks: list[_ActiveTrack] = []
    next_track_id = 0
    results: list[tuple[BallFrameDetection, int]] = []

    for det in raw_detections:
        cx, cy = _bbox_center(det.bbox)

        # Expire old tracks
        active_tracks = [
            t for t in active_tracks
            if det.frame_number - t.last_frame <= max_gap_frames
        ]

        # Find closest active track
        best_track = None
        best_dist = float("inf")
        for t in active_tracks:
            dist = math.hypot(cx - t.last_cx, cy - t.last_cy)
            if dist < best_dist:
                best_dist = dist
                best_track = t

        if best_track is not None and best_dist <= max_distance:
            # Link to existing track
            best_track.last_frame = det.frame_number
            best_track.last_cx = cx
            best_track.last_cy = cy
            results.append((det, best_track.track_id))
        else:
            # Start new track
            new_track = _ActiveTrack(
                track_id=next_track_id,
                last_frame=det.frame_number,
                last_cx=cx,
                last_cy=cy,
            )
            active_tracks.append(new_track)
            results.append((det, next_track_id))
            next_track_id += 1

    return results

"""
Post-processing for ByteTrack results: merge fragmented tracks and detect ID switches.

Operates on in-memory TrackResult objects (before DB storage).
"""

import logging

from spike_platform.config import settings
from spike_platform.services.detection import TrackResult, FrameDetection

logger = logging.getLogger(__name__)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _bbox_height(bbox: tuple[float, float, float, float]) -> float:
    return bbox[3] - bbox[1]


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _avg_bbox(frames: list[FrameDetection]) -> tuple[float, float, float, float]:
    """Average bbox across a list of frame detections."""
    n = len(frames)
    return (
        sum(f.bbox[0] for f in frames) / n,
        sum(f.bbox[1] for f in frames) / n,
        sum(f.bbox[2] for f in frames) / n,
        sum(f.bbox[3] for f in frames) / n,
    )


def detect_id_switches(tracks: list[TrackResult]) -> list[TrackResult]:
    """Detect and split tracks where the bbox jumps to a different person.

    A jump is flagged when bbox center displacement between consecutive frames
    exceeds TRACK_SWITCH_MAX_JUMP * bbox_height.
    """
    max_jump = settings.TRACK_SWITCH_MAX_JUMP
    result = []
    next_id = max((t.track_id for t in tracks), default=0) + 1000

    for track in tracks:
        if len(track.frames) < 2:
            result.append(track)
            continue

        split_points = []
        for i in range(1, len(track.frames)):
            prev = track.frames[i - 1]
            curr = track.frames[i]

            cx_prev, cy_prev = _bbox_center(prev.bbox)
            cx_curr, cy_curr = _bbox_center(curr.bbox)
            dist = ((cx_curr - cx_prev) ** 2 + (cy_curr - cy_prev) ** 2) ** 0.5
            h = _bbox_height(prev.bbox)

            if h > 0 and dist > max_jump * h:
                split_points.append(i)
                logger.warning(
                    f"ID switch detected in track {track.track_id} at frame "
                    f"{curr.frame_number} (jump={dist:.0f}px, {dist/h:.1f}x bbox height)"
                )

        if not split_points:
            result.append(track)
        else:
            # Split track at each switch point
            prev_idx = 0
            for sp in split_points:
                chunk = TrackResult(
                    track_id=track.track_id if prev_idx == 0 else next_id,
                    frames=track.frames[prev_idx:sp],
                )
                if prev_idx > 0:
                    next_id += 1
                result.append(chunk)
                prev_idx = sp
            # Last chunk
            result.append(TrackResult(
                track_id=next_id,
                frames=track.frames[prev_idx:],
            ))
            next_id += 1

    return result


def merge_fragmented_tracks(tracks: list[TrackResult]) -> list[TrackResult]:
    """Merge tracks that are likely fragments of the same person.

    Two tracks are merged if they are temporally close and spatially similar
    at their boundary frames.
    """
    if len(tracks) <= 1:
        return tracks

    max_gap = settings.TRACK_MERGE_MAX_GAP
    min_iou = settings.TRACK_MERGE_MIN_IOU

    # Sort by start frame
    sorted_tracks = sorted(tracks, key=lambda t: t.start_frame)

    # Union-Find for clustering
    parent = list(range(len(sorted_tracks)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Compare all candidate pairs
    for i in range(len(sorted_tracks)):
        track_a = sorted_tracks[i]
        for j in range(i + 1, len(sorted_tracks)):
            track_b = sorted_tracks[j]

            gap = track_b.start_frame - track_a.end_frame
            if gap > max_gap:
                break  # sorted, no more candidates
            if gap < 1:
                continue  # overlapping tracks, skip

            # Compare boundary bboxes (last 5 of A vs first 5 of B)
            tail_a = track_a.frames[-5:]
            head_b = track_b.frames[:5]
            avg_a = _avg_bbox(tail_a)
            avg_b = _avg_bbox(head_b)

            iou = _bbox_iou(avg_a, avg_b)

            # Also check center distance relative to bbox height
            cx_a, cy_a = _bbox_center(avg_a)
            cx_b, cy_b = _bbox_center(avg_b)
            dist = ((cx_b - cx_a) ** 2 + (cy_b - cy_a) ** 2) ** 0.5
            h = _bbox_height(avg_a)

            # Check bbox size ratio
            area_a = _bbox_area(avg_a)
            area_b = _bbox_area(avg_b)
            size_ratio = min(area_a, area_b) / max(area_a, area_b) if max(area_a, area_b) > 0 else 0

            should_merge = False
            if iou > min_iou:
                should_merge = True
            elif h > 0 and dist < 1.5 * h and size_ratio > 0.6:
                should_merge = True

            if should_merge:
                union(i, j)
                logger.info(
                    f"Merging tracks {track_a.track_id} and {track_b.track_id} "
                    f"(gap={gap} frames, IoU={iou:.2f}, dist={dist:.0f}px, size_ratio={size_ratio:.2f})"
                )

    # Build merged tracks from clusters
    clusters: dict[int, list[int]] = {}
    for i in range(len(sorted_tracks)):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    result = []
    for root, members in clusters.items():
        if len(members) == 1:
            result.append(sorted_tracks[members[0]])
        else:
            # Merge all tracks in the cluster
            base = sorted_tracks[members[0]]
            all_frames = list(base.frames)
            for m in members[1:]:
                all_frames.extend(sorted_tracks[m].frames)
            all_frames.sort(key=lambda f: f.frame_number)
            result.append(TrackResult(track_id=base.track_id, frames=all_frames))

    return result

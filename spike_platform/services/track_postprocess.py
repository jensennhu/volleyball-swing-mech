"""
Post-processing for ByteTrack results: merge fragmented tracks and detect ID switches.

Operates on in-memory TrackResult objects (before DB storage).
Optionally uses ReID embeddings (from OSNet) for appearance-based decisions.
"""

import logging
from typing import Optional

import cv2
import numpy as np

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


def extract_track_embeddings(
    video_path: str,
    tracks: list[TrackResult],
    encoder,
    boundary_frames: int = 5,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Extract ReID embeddings for each track's head and tail boundary frames.

    Does a single sequential video read, only decoding frames that are needed.
    Returns per-track embeddings for merge/split decisions.

    Args:
        video_path: Path to video file.
        tracks: List of TrackResult objects.
        encoder: ReIDEncoder instance.
        boundary_frames: Number of frames to sample from head/tail of each track.

    Returns:
        Dict mapping track_id -> {"head": embedding, "tail": embedding, "mean": embedding}
        where each embedding is shape (512,).
    """
    if not tracks:
        return {}

    # Collect which frames we need and what crops to extract
    # frame_number -> list of (track_id, "head"/"tail", bbox)
    frame_requests: dict[int, list[tuple[int, str, tuple]]] = {}

    for track in tracks:
        n = len(track.frames)
        head_frames = track.frames[:boundary_frames]
        tail_frames = track.frames[max(0, n - boundary_frames):]

        for det in head_frames:
            frame_requests.setdefault(det.frame_number, []).append(
                (track.track_id, "head", det.bbox)
            )
        for det in tail_frames:
            frame_requests.setdefault(det.frame_number, []).append(
                (track.track_id, "tail", det.bbox)
            )

    needed_frames = sorted(frame_requests.keys())
    if not needed_frames:
        return {}

    # Single sequential video read
    cap = cv2.VideoCapture(video_path)
    current_frame = 0
    needed_idx = 0

    # Collect crops: track_id -> {"head": [crops], "tail": [crops]}
    track_crops: dict[int, dict[str, list[np.ndarray]]] = {}

    while needed_idx < len(needed_frames):
        target = needed_frames[needed_idx]

        while current_frame < target:
            cap.grab()
            current_frame += 1

        ret, frame = cap.read()
        current_frame += 1
        if not ret:
            break

        h, w = frame.shape[:2]
        for track_id, boundary, bbox in frame_requests[target]:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            track_crops.setdefault(track_id, {"head": [], "tail": []})
            track_crops[track_id][boundary].append(crop)

        needed_idx += 1

    cap.release()

    # Encode all crops in batches per track
    result: dict[int, dict[str, np.ndarray]] = {}

    for track_id, boundaries in track_crops.items():
        head_crops = boundaries["head"]
        tail_crops = boundaries["tail"]

        head_emb = None
        tail_emb = None

        if head_crops:
            head_embs = encoder.encode_batch(head_crops)
            head_emb = head_embs.mean(axis=0)
            head_emb = head_emb / (np.linalg.norm(head_emb) + 1e-8)

        if tail_crops:
            tail_embs = encoder.encode_batch(tail_crops)
            tail_emb = tail_embs.mean(axis=0)
            tail_emb = tail_emb / (np.linalg.norm(tail_emb) + 1e-8)

        # Mean of head and tail for overall track embedding
        if head_emb is not None and tail_emb is not None:
            mean_emb = (head_emb + tail_emb) / 2
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        elif head_emb is not None:
            mean_emb = head_emb
        elif tail_emb is not None:
            mean_emb = tail_emb
        else:
            continue

        result[track_id] = {
            "head": head_emb if head_emb is not None else mean_emb,
            "tail": tail_emb if tail_emb is not None else mean_emb,
            "mean": mean_emb,
        }

    logger.info(f"Extracted ReID embeddings for {len(result)}/{len(tracks)} tracks")
    return result


def detect_id_switches(
    tracks: list[TrackResult],
    embeddings: Optional[dict[int, dict[str, np.ndarray]]] = None,
) -> list[TrackResult]:
    """Detect and split tracks where the bbox jumps to a different person.

    A jump is flagged when bbox center displacement between consecutive frames
    exceeds TRACK_SWITCH_MAX_JUMP * bbox_height.

    If ReID embeddings are provided, the head vs tail embedding similarity
    is used to confirm or suppress splits:
    - cosine_sim > 0.7: suppress split (same person moved fast)
    - cosine_sim < 0.4: confirm split (different person)
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

        # If we have embeddings, use head/tail similarity to confirm or suppress
        if split_points and embeddings and track.track_id in embeddings:
            emb = embeddings[track.track_id]
            head_emb = emb["head"]
            tail_emb = emb["tail"]
            sim = float(np.dot(head_emb, tail_emb))
            logger.info(
                f"Track {track.track_id} head-tail cosine sim: {sim:.3f}"
            )
            if sim > 0.7:
                logger.info(
                    f"Suppressing split for track {track.track_id} (high appearance similarity {sim:.3f})"
                )
                split_points = []

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


def merge_fragmented_tracks(
    tracks: list[TrackResult],
    embeddings: Optional[dict[int, dict[str, np.ndarray]]] = None,
) -> list[TrackResult]:
    """Merge tracks that are likely fragments of the same person.

    Two tracks are merged if they are temporally close and spatially similar
    at their boundary frames.

    If ReID embeddings are provided:
    - Block merge if cosine_sim < 0.3 (clearly different people)
    - Allow merge with relaxed spatial constraints if cosine_sim > 0.6
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

            # Compute ReID cosine similarity if available
            reid_sim = None
            if embeddings:
                emb_a = embeddings.get(track_a.track_id)
                emb_b = embeddings.get(track_b.track_id)
                if emb_a is not None and emb_b is not None:
                    # Compare tail of A with head of B
                    reid_sim = float(np.dot(emb_a["tail"], emb_b["head"]))

            # Decision logic
            should_merge = False

            if reid_sim is not None:
                if reid_sim < 0.3:
                    # Clearly different people — block merge regardless of spatial
                    logger.debug(
                        f"Blocking merge of tracks {track_a.track_id} and {track_b.track_id} "
                        f"(reid_sim={reid_sim:.3f} < 0.3)"
                    )
                    continue
                elif reid_sim > 0.6:
                    # High appearance similarity — relax spatial constraints
                    if h > 0 and dist < 3.0 * h:
                        should_merge = True
                    elif iou > min_iou:
                        should_merge = True

            # Standard spatial merge criteria (if not already decided)
            if not should_merge:
                if iou > min_iou:
                    should_merge = True
                elif h > 0 and dist < 1.5 * h and size_ratio > 0.6:
                    should_merge = True

            if should_merge:
                union(i, j)
                sim_str = f", reid_sim={reid_sim:.3f}" if reid_sim is not None else ""
                logger.info(
                    f"Merging tracks {track_a.track_id} and {track_b.track_id} "
                    f"(gap={gap} frames, IoU={iou:.2f}, dist={dist:.0f}px, "
                    f"size_ratio={size_ratio:.2f}{sim_str})"
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

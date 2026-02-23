"""
Post-processing for ByteTrack results: detect and split ID switches.

Operates on in-memory TrackResult objects (before DB storage).
Uses ReID embeddings (from OSNet) sampled throughout each track to detect
appearance changes that indicate the tracker switched to a different person.
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


def extract_track_embeddings(
    video_path: str,
    tracks: list[TrackResult],
    encoder,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """
    Extract ReID embeddings sampled throughout each track.

    Samples one crop every REID_SAMPLE_INTERVAL frames per track,
    plus always the first and last frame. Does a single sequential
    video read for efficiency.

    Args:
        video_path: Path to video file.
        tracks: List of TrackResult objects.
        encoder: ReIDEncoder instance.

    Returns:
        Dict mapping track_id -> list of (frame_number, embedding_512d) tuples,
        sorted by frame_number.
    """
    if not tracks:
        return {}

    sample_interval = settings.REID_SAMPLE_INTERVAL

    # Determine which frames to sample per track
    # frame_number -> list of (track_id, bbox)
    frame_requests: dict[int, list[tuple[int, tuple]]] = {}

    for track in tracks:
        if not track.frames:
            continue

        # Always sample first and last frame
        sample_indices = {0, len(track.frames) - 1}

        # Add evenly spaced samples throughout
        for i in range(sample_interval, len(track.frames), sample_interval):
            sample_indices.add(i)

        for idx in sample_indices:
            det = track.frames[idx]
            frame_requests.setdefault(det.frame_number, []).append(
                (track.track_id, det.bbox)
            )

    needed_frames = sorted(frame_requests.keys())
    if not needed_frames:
        return {}

    # Single sequential video read
    cap = cv2.VideoCapture(video_path)
    current_frame = 0
    needed_idx = 0

    # Collect: track_id -> list of (frame_number, crop)
    track_samples: dict[int, list[tuple[int, np.ndarray]]] = {}

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
        for track_id, bbox in frame_requests[target]:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            track_samples.setdefault(track_id, []).append((target, crop))

        needed_idx += 1

    cap.release()

    # Encode all crops per track
    result: dict[int, list[tuple[int, np.ndarray]]] = {}

    for track_id, samples in track_samples.items():
        samples.sort(key=lambda s: s[0])
        frame_numbers = [s[0] for s in samples]
        crops = [s[1] for s in samples]

        embeddings = encoder.encode_batch(crops)  # (N, 512)

        result[track_id] = list(zip(frame_numbers, embeddings))

    total_samples = sum(len(v) for v in result.values())
    logger.info(
        f"Extracted {total_samples} ReID samples across {len(result)}/{len(tracks)} tracks"
    )
    return result


def detect_id_switches(
    tracks: list[TrackResult],
    embeddings: Optional[dict[int, list[tuple[int, np.ndarray]]]] = None,
) -> list[TrackResult]:
    """Detect and split tracks where the person identity changes.

    Uses two signals:
    1. Spatial: bbox center jumps > TRACK_SWITCH_MAX_JUMP * bbox_height
    2. Appearance: cosine similarity between consecutive ReID samples
       drops below REID_SWITCH_THRESHOLD

    If both signals agree, the split is confirmed. If only appearance
    triggers, the split still happens (catches smooth ID switches).
    If only spatial triggers but appearance is high (>0.7), the split
    is suppressed (same person moved fast).
    """
    max_jump = settings.TRACK_SWITCH_MAX_JUMP
    reid_threshold = settings.REID_SWITCH_THRESHOLD
    result = []
    next_id = max((t.track_id for t in tracks), default=0) + 1000

    for track in tracks:
        if len(track.frames) < 2:
            result.append(track)
            continue

        # --- Signal 1: Spatial bbox jumps (sustained displacement) ---
        spatial_splits: set[int] = set()  # frame indices in track.frames
        SUSTAINED_FRAMES = 3    # must stay displaced for N frames after jump
        DECAY_RATIO = 0.85      # if dist drops below this fraction, it's motion

        for i in range(1, len(track.frames)):
            prev = track.frames[i - 1]
            curr = track.frames[i]

            cx_prev, cy_prev = _bbox_center(prev.bbox)
            cx_curr, cy_curr = _bbox_center(curr.bbox)
            dist = ((cx_curr - cx_prev) ** 2 + (cy_curr - cy_prev) ** 2) ** 0.5
            h = _bbox_height(prev.bbox)

            if h > 0 and dist > max_jump * h:
                # Large jump detected — check if displacement persists
                sustained = True
                for j in range(1, min(SUSTAINED_FRAMES + 1, len(track.frames) - i)):
                    future = track.frames[i + j]
                    cx_fut, cy_fut = _bbox_center(future.bbox)
                    dist_from_origin = (
                        (cx_fut - cx_prev) ** 2 + (cy_fut - cy_prev) ** 2
                    ) ** 0.5
                    if dist_from_origin < dist * DECAY_RATIO:
                        sustained = False
                        break

                if sustained:
                    spatial_splits.add(i)
                else:
                    logger.info(
                        f"Transient spatial jump in track {track.track_id} at frame "
                        f"{track.frames[i].frame_number} (displacement decayed, likely motion)"
                    )

        # --- Signal 3: Bbox height changes (different-sized person) ---
        height_splits: set[int] = set()
        HEIGHT_SWITCH_RATIO = 2.0  # height changes by more than 2x

        for i in range(1, len(track.frames)):
            prev = track.frames[i - 1]
            curr = track.frames[i]
            h_prev = _bbox_height(prev.bbox)
            h_curr = _bbox_height(curr.bbox)
            if h_prev > 0 and h_curr > 0:
                ratio = max(h_prev, h_curr) / min(h_prev, h_curr)
                if ratio > HEIGHT_SWITCH_RATIO:
                    height_splits.add(i)
                    logger.info(
                        f"Height change in track {track.track_id} at frame "
                        f"{curr.frame_number} (ratio={ratio:.2f}, {h_prev:.0f}→{h_curr:.0f})"
                    )

        # --- Signal 2: ReID appearance drops ---
        reid_split_frames: set[int] = set()  # video frame numbers where appearance changes
        reid_high_sim_frames: set[int] = set()  # frames with high similarity (suppress spatial)
        reid_low_sim_frames: set[int] = set()   # frames with low similarity (confirm spatial)

        track_embs = embeddings.get(track.track_id, []) if embeddings else []

        if len(track_embs) >= 2:
            for k in range(1, len(track_embs)):
                frame_a, emb_a = track_embs[k - 1]
                frame_b, emb_b = track_embs[k]
                sim = float(np.dot(emb_a, emb_b))

                if sim < reid_threshold:
                    reid_split_frames.add(frame_b)
                    for fn in range(frame_a, frame_b + 1):
                        reid_low_sim_frames.add(fn)
                    logger.info(
                        f"ReID appearance change in track {track.track_id} "
                        f"between frames {frame_a}-{frame_b} (sim={sim:.3f})"
                    )
                elif sim > 0.7:
                    # Mark all frames in this range as high-similarity
                    for fn in range(frame_a, frame_b + 1):
                        reid_high_sim_frames.add(fn)

        # --- Combine signals into final split points ---
        split_points: list[int] = []  # indices into track.frames

        # Spatial splits: require ReID confirmation (low sim) or absence of ReID data
        for i in spatial_splits:
            frame_num = track.frames[i].frame_number
            if frame_num in reid_high_sim_frames:
                logger.info(
                    f"Suppressing spatial split in track {track.track_id} at frame "
                    f"{frame_num} (high ReID similarity)"
                )
            elif frame_num in reid_low_sim_frames:
                split_points.append(i)
                logger.warning(
                    f"Confirmed spatial+ReID split in track {track.track_id} "
                    f"at frame {frame_num}"
                )
            elif not track_embs:
                # No ReID data at all — trust spatial signal
                split_points.append(i)
                logger.warning(
                    f"Spatial ID switch in track {track.track_id} at frame "
                    f"{frame_num} (no ReID data)"
                )
            else:
                # ReID data exists but doesn't cover this frame — default to same person
                logger.info(
                    f"Suppressing spatial split in track {track.track_id} at frame "
                    f"{frame_num} (no ReID evidence of identity change)"
                )

        # ReID splits: require spatial support for moderate appearance drops
        REID_STRICT_THRESHOLD = 0.15  # appearance-only splits need much lower sim
        if reid_split_frames:
            frame_num_to_idx = {
                det.frame_number: i for i, det in enumerate(track.frames)
            }
            for k in range(1, len(track_embs)):
                frame_a, emb_a = track_embs[k - 1]
                frame_b, emb_b = track_embs[k]
                sim = float(np.dot(emb_a, emb_b))

                if frame_b not in reid_split_frames:
                    continue

                # Find the track frame index closest to this frame
                if frame_b in frame_num_to_idx:
                    idx = frame_num_to_idx[frame_b]
                else:
                    idx = min(
                        range(len(track.frames)),
                        key=lambda i: abs(track.frames[i].frame_number - frame_b),
                    )

                if idx <= 0 or idx in split_points:
                    continue

                # Check if there's spatial support nearby (within 5 indices)
                has_spatial = any(abs(idx - sp) <= 5 for sp in spatial_splits)

                if has_spatial:
                    split_points.append(idx)
                    logger.warning(
                        f"Confirmed ReID+spatial split in track {track.track_id} "
                        f"at frame {track.frames[idx].frame_number} (sim={sim:.3f})"
                    )
                elif sim < REID_STRICT_THRESHOLD:
                    split_points.append(idx)
                    logger.warning(
                        f"Appearance-only split in track {track.track_id} "
                        f"at frame {track.frames[idx].frame_number} (sim={sim:.3f})"
                    )
                else:
                    logger.info(
                        f"Suppressing ReID split in track {track.track_id} at frame "
                        f"{track.frames[idx].frame_number} (sim={sim:.3f}, no spatial support)"
                    )

        # Height splits: dramatically different bbox size = different person
        for i in height_splits:
            if i not in split_points:
                frame_num = track.frames[i].frame_number
                if frame_num in reid_high_sim_frames:
                    logger.info(
                        f"Suppressing height split in track {track.track_id} at frame "
                        f"{frame_num} (high ReID similarity — same person changed pose)"
                    )
                else:
                    split_points.append(i)
                    logger.warning(
                        f"Height-based ID switch in track {track.track_id} at frame {frame_num}"
                    )

        # Sort and deduplicate split points
        split_points = sorted(set(split_points))

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
                if chunk.frames:
                    if prev_idx > 0:
                        next_id += 1
                    result.append(chunk)
                prev_idx = sp
            # Last chunk
            last_chunk = TrackResult(
                track_id=next_id,
                frames=track.frames[prev_idx:],
            )
            if last_chunk.frames:
                result.append(last_chunk)
                next_id += 1

    logger.info(
        f"ID switch detection: {len(tracks)} tracks in → {len(result)} tracks out"
    )
    return result


def merge_broken_tracks(
    tracks: list[TrackResult],
    embeddings: Optional[dict[int, list[tuple[int, np.ndarray]]]] = None,
    max_gap: int = 30,
    max_center_dist_ratio: float = 3.0,
) -> list[TrackResult]:
    """Merge tracks that are likely the same person split by tracker loss.

    Finds pairs where Track A ends within max_gap frames of Track B starting,
    and the spatial distance between A's last bbox and B's first bbox is small.
    """
    if len(tracks) <= 1:
        return tracks

    # Sort tracks by start frame
    tracks_sorted = sorted(
        tracks, key=lambda t: t.frames[0].frame_number if t.frames else float("inf")
    )

    merged: set[int] = set()  # indices that have been absorbed
    result = []

    for i, track_a in enumerate(tracks_sorted):
        if i in merged or not track_a.frames:
            continue

        current = track_a
        current_emb_ids = {track_a.track_id}  # track IDs whose embeddings belong to current

        # Try to extend by merging subsequent tracks
        while True:
            best_j = None
            best_dist = float("inf")

            last_frame = current.frames[-1]
            last_fn = last_frame.frame_number
            cx_last, cy_last = _bbox_center(last_frame.bbox)
            h_last = _bbox_height(last_frame.bbox)

            for j, track_b in enumerate(tracks_sorted):
                if j in merged or j == i or not track_b.frames:
                    continue
                if track_b is current:
                    continue

                first_frame = track_b.frames[0]
                first_fn = first_frame.frame_number

                # Check temporal gap
                gap = first_fn - last_fn
                if gap < 0 or gap > max_gap:
                    continue

                # Check spatial proximity
                cx_first, cy_first = _bbox_center(first_frame.bbox)
                dist = ((cx_first - cx_last) ** 2 + (cy_first - cy_last) ** 2) ** 0.5

                if h_last > 0 and dist / h_last > max_center_dist_ratio:
                    continue

                # Check bbox size consistency
                h_first = _bbox_height(first_frame.bbox)
                if h_last > 0 and h_first > 0:
                    height_ratio = min(h_last, h_first) / max(h_last, h_first)
                    if height_ratio < 0.5:
                        logger.debug(
                            f"Rejecting merge of track {track_b.track_id} into {current.track_id}: "
                            f"height ratio {height_ratio:.2f} (last_h={h_last:.0f}, first_h={h_first:.0f})"
                        )
                        continue

                # Check ReID similarity if available
                if embeddings:
                    embs_a = []
                    for tid in current_emb_ids:
                        embs_a.extend(embeddings.get(tid, []))
                    embs_a.sort(key=lambda x: x[0])
                    embs_b = embeddings.get(track_b.track_id, [])
                    if embs_a and embs_b:
                        sim = float(np.dot(embs_a[-1][1], embs_b[0][1]))
                        if sim < 0.2:
                            continue

                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j is not None:
                merged.add(best_j)
                merge_target = tracks_sorted[best_j]
                logger.info(
                    f"Merging track {merge_target.track_id} into {current.track_id} "
                    f"(gap={merge_target.frames[0].frame_number - current.frames[-1].frame_number} frames, "
                    f"dist={best_dist:.1f}px)"
                )
                current = TrackResult(
                    track_id=current.track_id,
                    frames=current.frames + merge_target.frames,
                )
                current_emb_ids.add(merge_target.track_id)
            else:
                break

        result.append(current)

    logger.info(f"Track merging: {len(tracks)} tracks in → {len(result)} tracks out")
    return result

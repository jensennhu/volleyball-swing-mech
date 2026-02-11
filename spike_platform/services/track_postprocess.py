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

        # --- Signal 1: Spatial bbox jumps ---
        spatial_splits: set[int] = set()  # frame indices in track.frames
        for i in range(1, len(track.frames)):
            prev = track.frames[i - 1]
            curr = track.frames[i]

            cx_prev, cy_prev = _bbox_center(prev.bbox)
            cx_curr, cy_curr = _bbox_center(curr.bbox)
            dist = ((cx_curr - cx_prev) ** 2 + (cy_curr - cy_prev) ** 2) ** 0.5
            h = _bbox_height(prev.bbox)

            if h > 0 and dist > max_jump * h:
                spatial_splits.add(i)

        # --- Signal 2: ReID appearance drops ---
        reid_split_frames: set[int] = set()  # video frame numbers where appearance changes
        reid_high_sim_frames: set[int] = set()  # frames with high similarity (suppress spatial)

        track_embs = embeddings.get(track.track_id, []) if embeddings else []

        if len(track_embs) >= 2:
            for k in range(1, len(track_embs)):
                frame_a, emb_a = track_embs[k - 1]
                frame_b, emb_b = track_embs[k]
                sim = float(np.dot(emb_a, emb_b))

                if sim < reid_threshold:
                    reid_split_frames.add(frame_b)
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

        # Spatial splits: confirm unless ReID says high similarity
        for i in spatial_splits:
            frame_num = track.frames[i].frame_number
            if frame_num in reid_high_sim_frames:
                logger.info(
                    f"Suppressing spatial split in track {track.track_id} at frame "
                    f"{frame_num} (high ReID similarity)"
                )
            else:
                split_points.append(i)
                logger.warning(
                    f"Spatial ID switch in track {track.track_id} at frame {frame_num}"
                )

        # ReID splits: find closest frame index for each appearance change
        if reid_split_frames:
            frame_num_to_idx = {
                det.frame_number: i for i, det in enumerate(track.frames)
            }
            for frame_num in reid_split_frames:
                # Find the track frame index closest to this frame number
                if frame_num in frame_num_to_idx:
                    idx = frame_num_to_idx[frame_num]
                else:
                    # Find nearest frame index
                    idx = min(
                        range(len(track.frames)),
                        key=lambda i: abs(track.frames[i].frame_number - frame_num),
                    )
                if idx > 0 and idx not in split_points:
                    split_points.append(idx)
                    logger.warning(
                        f"ReID ID switch in track {track.track_id} at frame "
                        f"{track.frames[idx].frame_number}"
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

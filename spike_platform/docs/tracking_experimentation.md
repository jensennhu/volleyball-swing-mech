# Tracking Experimentation Log

History of changes to person detection, tracking, and post-processing in the spike platform. Documented for future reference when revisiting tracking quality.

## Original Problem

Two issues observed during spike review:

1. **Track fragmentation**: A single person's spike motion split across multiple tracks. The approach phase might be in track A, while the swing/land phase ends up in track B. This makes phase inference incomplete since neither track captures the full spike sequence.

2. **ID switching**: A track's bounding box jumps from one person to another mid-track, typically during occlusion (e.g., one player walks in front of another). Example: track 837 starts on a player in a red jersey, then the bbox hops to a nearby player in a blue jersey with a different height.

## Iteration 1: ByteTrack Tuning + Spatial Post-Processing

**Approach**: Tune ByteTrack config and add spatial heuristics for merge/split.

**Changes**:
- Created `spike_platform/cfg/bytetrack.yaml` with tuned params:
  - `track_buffer: 50` (up from default 30) — keeps lost tracks alive longer
  - `match_thresh: 0.75` — stricter IoU matching
  - `track_high_thresh: 0.25`, `track_low_thresh: 0.1`, `new_track_thresh: 0.25`
- Created `spike_platform/services/track_postprocess.py` with two functions:
  - `detect_id_switches()` — splits tracks where bbox center jumps > 2x bbox height between consecutive frames
  - `merge_fragmented_tracks()` — Union-Find clustering to merge temporally close (gap < 40 frames), spatially similar tracks (IoU > 0.2 or center distance < 1.5x height with size ratio > 0.6)
- Added config settings: `TRACK_MERGE_MAX_GAP=40`, `TRACK_MERGE_MIN_IOU=0.2`, `TRACK_SWITCH_MAX_JUMP=2.0`
- Added reprocess endpoint + UI button

**Result**: Partial improvement. Some fragmentation fixed by merging. But ID switches between visually different people (different jersey colors, different heights) were NOT caught — spatial heuristics can't distinguish people by appearance.

**Key Lesson**: Spatial-only post-processing is insufficient. Need appearance-based features to distinguish between people.

## Iteration 2: BoTSORT with Built-in ReID (`model: auto`)

**Approach**: Switch from ByteTrack to BoTSORT tracker which has native ReID support via `model: auto` (uses YOLO nano features for appearance matching).

**Changes**:
- Created `spike_platform/cfg/botsort.yaml`:
  - `with_reid: true`, `model: auto`
  - `gmc_method: sparseOptFlow` (camera motion compensation)
  - `proximity_thresh: 0.5`, `appearance_thresh: 0.25`
- Updated `detection.py` to use `botsort.yaml`
- Added `torchreid>=0.2.5` to requirements

**Result**: **Worse than Iteration 1.** The YOLO nano features used by `model: auto` are not discriminative enough for person ReID. Still confused red/blue jerseys and different-height people. The appearance features from a detection model aren't designed for re-identification.

**Key Lesson**: YOLO detection features ≠ ReID features. Need a purpose-built person ReID model (trained on person re-identification datasets, not object detection).

## Iteration 3: ByteTrack + Standalone OSNet ReID (Boundary-Only)

**Approach**: Revert to ByteTrack. Add standalone OSNet x0.25 (a lightweight person ReID model, ~0.5M params, 512-dim embeddings, pretrained on market1501/msmt17/cuhk03) in the post-processing layer.

**Changes**:
- Reverted `detection.py` back to `bytetrack.yaml`
- Created `spike_platform/services/reid.py` — `ReIDEncoder` class wrapping torchreid's OSNet x0.25
  - Input: person crop (BGR) → resize 128x256 → ImageNet normalization → OSNet → 512-dim L2-normalized embedding
  - `encode()` for single crops, `encode_batch()` for batched inference
- Updated `extract_track_embeddings()` — samples first/last 5 frames per track (boundary-only), single sequential video read
- Updated `merge_fragmented_tracks()` with ReID:
  - Block merge if cosine_sim < 0.3 (different people)
  - Relax spatial constraints if cosine_sim > 0.6 (same person, allow distance < 3x height)
  - Standard spatial rules in 0.3-0.6 range
- Updated `detect_id_switches()` with ReID:
  - Suppress split if head-tail cosine_sim > 0.7 (same person moved fast)

**Result**: **Worse than all previous iterations.** The merge function was far too aggressive — collapsed multiple distinct players into a single mega-track. The spatial merge thresholds (IoU > 0.2, distance < 1.5x height) were too loose, and the ReID only blocked merges at very low similarity (< 0.3). In the 0.3-0.6 ambiguous range, the old aggressive spatial rules still fired. Result: "over majority of the video focused on a single player."

**Key Lesson**: The merge function was solving a minor problem (occasional fragmentation) while creating a major one (collapsing distinct players). ByteTrack's `track_buffer: 50` already handles most fragmentation via its internal re-association. Post-processing merging does more harm than good.

## Iteration 4: ByteTrack + OSNet ReID (Dense Sampling, Split-Only) — Current

**Approach**: Remove merging entirely. Enhance split detection with dense ReID sampling throughout each track to catch smooth ID switches that don't produce large bbox jumps.

**Changes**:
- **Removed `merge_fragmented_tracks()` from pipeline** (function kept in code but not called)
- Removed config: `TRACK_MERGE_MAX_GAP`, `TRACK_MERGE_MIN_IOU`
- Rewrote `extract_track_embeddings()` for dense sampling:
  - Samples every `REID_SAMPLE_INTERVAL` (30) frames throughout track, plus first and last frame
  - Returns list of `(frame_number, embedding)` tuples per track
  - Still uses single sequential video read for efficiency
- Rewrote `detect_id_switches()` with two-signal approach:
  - **Signal 1 (spatial)**: bbox center jump > 2x height (same as before)
  - **Signal 2 (appearance)**: cosine similarity between consecutive ReID samples drops below `REID_SWITCH_THRESHOLD` (0.4)
  - Combined logic: spatial split suppressed if ReID says high similarity (>0.7); ReID split confirmed even without bbox jump
- Added config: `REID_SAMPLE_INTERVAL=30`, `REID_SWITCH_THRESHOLD=0.4`

**Result**: **Best tracking accuracy so far.** Tracks correctly maintain person identity — red/blue jersey switches are caught, different-height people stay separate. Tradeoff: more tracks (due to splitting) and fewer segments per track (shorter tracks after splitting produce fewer 40-frame windows).

**Key Lesson**: Less is more with post-processing. Don't try to merge — let the tracker handle re-association. Focus post-processing on splitting incorrectly-merged tracks using appearance features. The tradeoff (more tracks, fewer segments) is acceptable because:
- Spike detection works on individual 40-frame segments, not full tracks
- Phase classification runs per-frame on whatever the track contains
- Phase annotation UI allows expanding frame range beyond segment boundaries
- Clean, identity-consistent tracks > long, identity-confused tracks

## Current Configuration

```yaml
# bytetrack.yaml
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 50
match_thresh: 0.75
fuse_score: true
```

```python
# config.py
TRACK_SWITCH_MAX_JUMP: float = 2.0   # spatial split threshold
REID_SAMPLE_INTERVAL: int = 30       # frames between embedding samples
REID_SWITCH_THRESHOLD: float = 0.4   # cosine sim below this triggers split
MIN_TRACK_FRAMES: int = 40           # discard tracks shorter than this
```

## Architecture

```
Video -> YOLOv8 + ByteTrack -> Raw tracks
  -> OSNet ReID embedding extraction (every 30 frames)
  -> ID switch detection (spatial jumps + appearance drops)
  -> Filter tracks < 40 frames
  -> Pose extraction -> Segmentation -> DB
```

## Future Considerations

- **Threshold tuning**: `REID_SWITCH_THRESHOLD=0.4` may need adjustment per video quality/lighting. Could expose as a UI setting.
- **Re-enabling merge**: If fragmentation becomes a problem again, merge could be re-added but ONLY with strict ReID requirement (e.g., require cosine_sim > 0.7 AND spatial proximity, no spatial-only merges).
- **MIN_TRACK_FRAMES**: Could lower from 40 to 20 to recover more short track fragments, at the cost of more noise.
- **Per-frame ReID**: Current sampling every 30 frames could miss brief ID switches. Denser sampling increases accuracy but also processing time.
- **BoTSORT with proper ReID model**: If ultralytics adds support for custom ReID model paths (not just `model: auto`), could try BoTSORT with OSNet directly — this would be cleaner than post-processing splits.

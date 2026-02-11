"""
Phase inference service: load trained PhaseLSTM, predict phases on spike segments.

For each spike segment's track, loads per-frame pose features, runs the model,
converts per-frame predictions into contiguous phase regions, and stores them
in segment_phases with human_label=NULL (distinguishing from human annotations).
"""

import json
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import torch

from spike_platform.config import settings
from spike_platform.database import SessionLocal
from spike_platform.models.db_models import Segment, SegmentPhase, Track, TrackFrame, TrainingRun
from spike_platform.ml.phase_lstm import PhaseLSTM

IDX_TO_PHASE = {0: "approach", 1: "jump", 2: "swing", 3: "land"}


def _frames_to_regions(frame_numbers: list[int], predictions: list[int], probabilities: np.ndarray):
    """Convert per-frame predictions into contiguous phase regions.

    Returns list of dicts: {phase, start_frame, end_frame, confidence}
    """
    if not frame_numbers:
        return []

    regions = []
    current_phase = predictions[0]
    start_idx = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_phase:
            # Close current region
            phase_name = IDX_TO_PHASE[current_phase]
            region_probs = probabilities[start_idx:i, current_phase]
            regions.append({
                "phase": phase_name,
                "start_frame": frame_numbers[start_idx],
                "end_frame": frame_numbers[i - 1],
                "confidence": float(np.mean(region_probs)),
            })
            current_phase = predictions[i]
            start_idx = i

    # Close last region
    phase_name = IDX_TO_PHASE[current_phase]
    region_probs = probabilities[start_idx:, current_phase]
    regions.append({
        "phase": phase_name,
        "start_frame": frame_numbers[start_idx],
        "end_frame": frame_numbers[-1],
        "confidence": float(np.mean(region_probs)),
    })

    return regions


def run_phase_inference(
    video_id: str,
    training_run_id: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """
    Run phase predictions on all spike segments of a video.

    Works per-track: loads full track frame features, runs PhaseLSTM,
    then clips predicted regions to each segment on the track.
    """
    db = SessionLocal()
    try:
        # Find model checkpoint
        if training_run_id:
            run = db.query(TrainingRun).filter(
                TrainingRun.id == training_run_id,
                TrainingRun.task_type == "phase_classification",
            ).first()
        else:
            run = (
                db.query(TrainingRun)
                .filter(
                    TrainingRun.status == "completed",
                    TrainingRun.task_type == "phase_classification",
                )
                .order_by(TrainingRun.completed_at.desc())
                .first()
            )

        if not run or not run.checkpoint_dir:
            if progress_callback:
                progress_callback(100.0, "No trained phase model available.")
            return

        if progress_callback:
            progress_callback(5.0, "Loading phase model...")

        # Load model and scaler
        checkpoint_dir = Path(run.checkpoint_dir)
        config = json.loads((checkpoint_dir / "config.json").read_text())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_crf = config.get("use_crf", False)
        model = PhaseLSTM(
            input_dim=config["input_dim"],
            lstm_units=config["lstm_units"],
            dropout=config.get("dropout", 0.3),
            use_crf=use_crf,
        ).to(device)
        state_dict = torch.load(
            str(checkpoint_dir / "model.pt"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

        scaler = joblib.load(str(checkpoint_dir / "scaler.pkl"))

        # Find all spike segments for this video, grouped by track (exclude non-player tracks)
        spike_segments = (
            db.query(Segment)
            .join(Track, Segment.track_id == Track.id)
            .filter(
                Segment.video_id == video_id,
                Segment.human_label == 1,
                Track.role != "non_player",
            )
            .all()
        )

        if not spike_segments:
            if progress_callback:
                progress_callback(100.0, "No spike segments to predict phases for.")
            return

        # Group by track_id
        tracks_map: dict[int, list[Segment]] = {}
        for seg in spike_segments:
            tracks_map.setdefault(seg.track_id, []).append(seg)

        total_tracks = len(tracks_map)
        predicted_count = 0

        if progress_callback:
            progress_callback(10.0, f"Predicting phases for {total_tracks} tracks...")

        for track_idx, (track_id, segs) in enumerate(tracks_map.items()):
            # Check if any segment on this track already has human annotations
            has_human = (
                db.query(SegmentPhase)
                .filter(
                    SegmentPhase.segment_id.in_([s.id for s in segs]),
                    SegmentPhase.human_label.isnot(None),
                )
                .first()
            )
            if has_human:
                # Skip tracks with human annotations
                continue

            # Get track info for frame range
            track = db.query(Track).filter(Track.id == track_id).first()
            if not track:
                continue

            # Load all track frames with pose features
            track_frames = (
                db.query(TrackFrame)
                .filter(
                    TrackFrame.track_id == track_id,
                    TrackFrame.frame_number >= track.start_frame,
                    TrackFrame.frame_number <= track.end_frame,
                )
                .order_by(TrackFrame.frame_number)
                .all()
            )

            # Build feature matrix
            frame_numbers = []
            features = []
            for tf in track_frames:
                if tf.pose_features is None:
                    continue
                feat = json.loads(tf.pose_features)
                if len(feat) != settings.FEATURE_DIM:
                    continue
                frame_numbers.append(tf.frame_number)
                features.append(feat)

            if len(features) < 10:
                continue

            # Scale and predict
            feat_array = np.array(features, dtype=np.float32)
            feat_scaled = scaler.transform(feat_array)

            # Run model: (1, T, 33) -> (1, T, 4)
            input_tensor = torch.FloatTensor(feat_scaled).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)  # (1, T, 4)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (T, 4)
                if use_crf:
                    mask = torch.ones(1, len(features), dtype=torch.bool, device=device)
                    preds = np.array(model.crf_decode(logits, mask)[0])  # (T,)
                else:
                    preds = np.argmax(probs, axis=-1)  # (T,)

            # Convert to contiguous phase regions
            regions = _frames_to_regions(frame_numbers, preds.tolist(), probs)

            # For each segment on this track, clip regions and store
            for seg in segs:
                # Delete existing predicted phases (preserve human annotations)
                db.query(SegmentPhase).filter(
                    SegmentPhase.segment_id == seg.id,
                    SegmentPhase.human_label.is_(None),
                ).delete(synchronize_session="fetch")

                for region in regions:
                    clipped_start = max(region["start_frame"], seg.start_frame)
                    clipped_end = min(region["end_frame"], seg.end_frame)
                    if clipped_start > clipped_end:
                        continue
                    phase = SegmentPhase(
                        segment_id=seg.id,
                        phase=region["phase"],
                        start_frame=clipped_start,
                        end_frame=clipped_end,
                        confidence=region["confidence"],
                        human_label=None,
                        model_run_id=run.id,
                    )
                    db.add(phase)

            db.commit()
            predicted_count += 1

            pct = 10 + (track_idx / total_tracks) * 85
            if progress_callback:
                progress_callback(pct, f"Predicted track {track_idx + 1}/{total_tracks}")

        if progress_callback:
            progress_callback(
                100.0,
                f"Phase inference complete. Predicted {predicted_count}/{total_tracks} tracks.",
            )

    finally:
        db.close()

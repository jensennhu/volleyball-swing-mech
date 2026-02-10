"""
Inference service: load trained model, predict on segments.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from spike_platform.config import settings
from spike_platform.database import SessionLocal
from spike_platform.models.db_models import Segment, TrainingRun
from spike_platform.ml.trainer import SpikeTrainer


def run_inference_on_video(
    video_id: str,
    training_run_id: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """
    Run spike predictions on all segments of a video.

    If training_run_id is not specified, uses the latest completed run.
    """
    db = SessionLocal()
    try:
        # Find model checkpoint
        if training_run_id:
            run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        else:
            run = (
                db.query(TrainingRun)
                .filter(TrainingRun.status == "completed")
                .order_by(TrainingRun.completed_at.desc())
                .first()
            )

        if not run or not run.checkpoint_dir:
            if progress_callback:
                progress_callback(100.0, "No trained model available.")
            return

        if progress_callback:
            progress_callback(10.0, "Loading model...")

        trainer = SpikeTrainer.load_checkpoint(run.checkpoint_dir)
        trainer.model.eval()

        # Get all segments for this video
        segments = (
            db.query(Segment)
            .filter(Segment.video_id == video_id)
            .all()
        )

        if not segments:
            if progress_callback:
                progress_callback(100.0, "No segments to predict.")
            return

        if progress_callback:
            progress_callback(20.0, f"Predicting {len(segments)} segments...")

        # Batch predict
        features_list = []
        valid_segments = []
        for seg in segments:
            if not seg.features_path:
                continue
            feat_path = Path(seg.features_path)
            if not feat_path.exists():
                continue
            feat = np.load(str(feat_path))
            if feat.shape == (settings.WINDOW_SIZE, settings.FEATURE_DIM):
                features_list.append(feat)
                valid_segments.append(seg)

        if not features_list:
            if progress_callback:
                progress_callback(100.0, "No valid features found.")
            return

        features = np.stack(features_list)  # (N, 40, 33)

        # Scale features
        n, t, f = features.shape
        features_scaled = trainer.scaler.transform(features.reshape(-1, f)).reshape(n, t, f)

        # Run inference
        batch_tensor = torch.FloatTensor(features_scaled).to(trainer.device)

        with torch.no_grad():
            logits = trainer.model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        # Update segments
        for seg, prob in zip(valid_segments, probs):
            seg.prediction = int(prob >= 0.5)
            seg.confidence = float(prob)
            seg.model_run_id = run.id

        db.commit()

        # Update video status
        from spike_platform.models.db_models import Video
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = "predicted"
            db.commit()

        if progress_callback:
            spike_count = sum(1 for p in probs if p >= 0.5)
            progress_callback(
                100.0,
                f"Done. {spike_count}/{len(probs)} segments predicted as spikes.",
            )

    finally:
        db.close()

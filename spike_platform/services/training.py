"""
Training service: collects labeled data, trains model, saves checkpoint.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from spike_platform.config import settings
from spike_platform.database import SessionLocal
from spike_platform.models.db_models import Segment, TrainingRun
from spike_platform.ml.trainer import SpikeTrainer


def run_training(
    training_run_id: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """
    Collect labeled segments, train SpikeLSTM, save checkpoint, update DB.
    Called as a background job.
    """
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if not run:
            return

        if progress_callback:
            progress_callback(5.0, "Collecting labeled data...")

        # Collect all labeled segments
        labeled_segments = (
            db.query(Segment)
            .filter(Segment.human_label.isnot(None))
            .filter(Segment.features_path.isnot(None))
            .all()
        )

        features_list = []
        labels_list = []
        for seg in labeled_segments:
            feat_path = Path(seg.features_path)
            if not feat_path.exists():
                continue
            feat = np.load(str(feat_path))
            if feat.shape != (settings.WINDOW_SIZE, settings.FEATURE_DIM):
                continue
            features_list.append(feat)
            labels_list.append(seg.human_label)

        if len(features_list) < 10:
            run.status = "failed"
            db.commit()
            return

        features = np.stack(features_list)  # (N, 40, 33)
        labels = np.array(labels_list)      # (N,)

        # Stratified split: 70/15/15
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.3, stratify=labels, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        # Update dataset stats
        run.train_count = len(X_train)
        run.val_count = len(X_val)
        run.test_count = len(X_test)
        run.positive_count = int(np.sum(labels == 1))
        run.negative_count = int(np.sum(labels == 0))
        db.commit()

        if progress_callback:
            progress_callback(15.0, f"Training on {len(X_train)} samples...")

        # Train
        lstm_units = json.loads(run.lstm_units)
        trainer = SpikeTrainer(
            input_dim=settings.FEATURE_DIM,
            lstm_units=lstm_units,
            dropout=run.dropout,
            learning_rate=run.learning_rate,
            batch_size=run.batch_size,
            epochs=run.epochs,
        )

        def on_epoch(epoch, train_loss, val_loss):
            pct = 15 + (epoch / run.epochs) * 70
            if progress_callback:
                progress_callback(pct, f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            # Update DB with latest losses
            run.train_loss = train_loss
            run.val_loss = val_loss
            run.best_epoch = trainer.best_epoch
            db.commit()

        result = trainer.train(X_train, y_train, X_val, y_val, on_epoch=on_epoch)

        if progress_callback:
            progress_callback(90.0, "Evaluating on test set...")

        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)

        # Save checkpoint
        checkpoint_dir = str(settings.CHECKPOINT_DIR / f"run_{run.id:03d}")
        trainer.save_checkpoint(checkpoint_dir)

        # Update training run
        run.status = "completed"
        run.best_epoch = result["best_epoch"]
        run.train_loss = result["train_loss"]
        run.val_loss = result["val_loss"]
        run.test_accuracy = metrics["accuracy"]
        run.test_precision = metrics["precision"]
        run.test_recall = metrics["recall"]
        run.test_f1 = metrics["f1"]
        run.test_auc = metrics.get("auc")
        run.checkpoint_dir = checkpoint_dir
        run.completed_at = datetime.now(timezone.utc)
        db.commit()

        if progress_callback:
            progress_callback(100.0, f"Training complete. Test F1: {metrics['f1']:.3f}")

    except Exception as e:
        run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if run:
            run.status = "failed"
            db.commit()
        raise
    finally:
        db.close()

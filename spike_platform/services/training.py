"""
Training service: collects labeled data, trains model, saves checkpoint.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from spike_platform.config import settings
from spike_platform.database import SessionLocal
from spike_platform.models.db_models import Segment, Track, TrainingRun
from spike_platform.ml.trainer import SpikeTrainer

logger = logging.getLogger(__name__)


def split_by_video(
    video_ids: list[str],
    video_has_positive: list[bool],
) -> tuple[list[str], list[str], list[str]]:
    """Split video IDs into train/val/test (70/15/15) at the video level.

    Stratifies by whether each video contains positive labels when possible.
    Falls back gracefully for small datasets.

    Returns:
        (train_vids, val_vids, test_vids)
    """
    n = len(video_ids)

    if n < 3:
        logger.warning(
            f"Only {n} video(s) with labeled data — using all for training "
            f"(no val/test holdout). Add more videos for honest evaluation."
        )
        return list(video_ids), [], []

    strat = video_has_positive if len(set(video_has_positive)) > 1 else None

    try:
        train_vids, temp_vids = train_test_split(
            video_ids, test_size=0.3, stratify=strat, random_state=42,
        )
    except ValueError:
        # Stratification failed (e.g., too few samples per class)
        train_vids, temp_vids = train_test_split(
            video_ids, test_size=0.3, random_state=42,
        )

    if len(temp_vids) < 2:
        return list(train_vids), list(temp_vids), []

    temp_has_pos = [video_has_positive[video_ids.index(v)] for v in temp_vids]
    temp_strat = temp_has_pos if len(set(temp_has_pos)) > 1 else None

    try:
        val_vids, test_vids = train_test_split(
            temp_vids, test_size=0.5, stratify=temp_strat, random_state=42,
        )
    except ValueError:
        val_vids, test_vids = train_test_split(
            temp_vids, test_size=0.5, random_state=42,
        )

    return list(train_vids), list(val_vids), list(test_vids)


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

        # Collect all labeled segments, grouped by video (exclude non-player tracks)
        labeled_segments = (
            db.query(Segment)
            .join(Track, Segment.track_id == Track.id)
            .filter(Segment.human_label.isnot(None))
            .filter(Segment.features_path.isnot(None))
            .filter(Track.role != "non_player")
            .all()
        )

        # Group by video: {video_id: [(features, label), ...]}
        video_data: dict[str, list[tuple[np.ndarray, int]]] = {}
        for seg in labeled_segments:
            feat_path = Path(seg.features_path)
            if not feat_path.exists():
                continue
            feat = np.load(str(feat_path))
            if feat.shape != (settings.WINDOW_SIZE, settings.FEATURE_DIM):
                continue
            video_data.setdefault(seg.video_id, []).append((feat, seg.human_label))

        total_samples = sum(len(v) for v in video_data.values())
        if total_samples < 10:
            run.status = "failed"
            db.commit()
            return

        # Video-level split
        video_ids = list(video_data.keys())
        video_has_positive = [
            any(label == 1 for _, label in video_data[v]) for v in video_ids
        ]

        train_vids, val_vids, test_vids = split_by_video(video_ids, video_has_positive)

        def collect_split(vids):
            feats, labs = [], []
            for v in vids:
                for feat, label in video_data[v]:
                    feats.append(feat)
                    labs.append(label)
            if not feats:
                return np.empty((0, settings.WINDOW_SIZE, settings.FEATURE_DIM)), np.array([])
            return np.stack(feats), np.array(labs)

        X_train, y_train = collect_split(train_vids)
        X_val, y_val = collect_split(val_vids)
        X_test, y_test = collect_split(test_vids)

        # Log split composition
        for name, vids, X, y in [
            ("Train", train_vids, X_train, y_train),
            ("Val", val_vids, X_val, y_val),
            ("Test", test_vids, X_test, y_test),
        ]:
            pos = int(np.sum(y == 1)) if len(y) > 0 else 0
            logger.info(f"  {name}: {len(vids)} videos, {len(X)} segments ({pos} spike)")

        # Update dataset stats
        labels = np.concatenate([y_train, y_val, y_test]) if len(y_val) > 0 else y_train
        run.train_count = len(X_train)
        run.val_count = len(X_val)
        run.test_count = len(X_test)
        run.positive_count = int(np.sum(labels == 1))
        run.negative_count = int(np.sum(labels == 0))
        db.commit()

        if progress_callback:
            progress_callback(15.0, f"Training on {len(X_train)} samples ({len(train_vids)} videos)...")

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

        # Evaluate (skip if no test data)
        if len(X_test) > 0:
            metrics = trainer.evaluate(X_test, y_test)
        else:
            logger.warning("No test set — skipping evaluation metrics.")
            metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None, "auc": None}

        # Save checkpoint
        checkpoint_dir = str(settings.CHECKPOINT_DIR / f"run_{run.id:03d}")
        trainer.save_checkpoint(checkpoint_dir)

        # Append split metadata to checkpoint config
        config_path = Path(checkpoint_dir) / "config.json"
        config = json.loads(config_path.read_text())
        config["split"] = {
            "method": "video_level",
            "train_videos": train_vids,
            "val_videos": val_vids,
            "test_videos": test_vids,
        }
        config_path.write_text(json.dumps(config, indent=2))

        # Update training run
        run.status = "completed"
        run.best_epoch = result["best_epoch"]
        run.train_loss = result["train_loss"]
        run.val_loss = result["val_loss"]
        run.test_accuracy = metrics.get("accuracy")
        run.test_precision = metrics.get("precision")
        run.test_recall = metrics.get("recall")
        run.test_f1 = metrics.get("f1")
        run.test_auc = metrics.get("auc")
        run.checkpoint_dir = checkpoint_dir
        run.completed_at = datetime.now(timezone.utc)
        db.commit()

        f1_str = f"{metrics['f1']:.3f}" if metrics.get("f1") is not None else "N/A (no test set)"
        if progress_callback:
            progress_callback(100.0, f"Training complete. Test F1: {f1_str}")

    except Exception as e:
        run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if run:
            run.status = "failed"
            db.commit()
        raise
    finally:
        db.close()

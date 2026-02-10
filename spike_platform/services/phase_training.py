"""
Phase classification training service.

Collects phase-annotated spike segments, builds per-frame labeled sequences
from track_frames, trains a PhaseLSTM, and saves checkpoint.
"""

import json
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import torch
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from spike_platform.config import settings
from spike_platform.database import SessionLocal
from spike_platform.models.db_models import Segment, SegmentPhase, TrackFrame, TrainingRun
from spike_platform.ml.phase_lstm import PhaseLSTM

PHASE_TO_IDX = {"approach": 0, "jump": 1, "swing": 2, "land": 3}


class PhaseDataset(Dataset):
    """Dataset of variable-length sequences with per-frame phase labels."""

    def __init__(self, features_list: list[np.ndarray], labels_list: list[np.ndarray], scaler: StandardScaler = None):
        # Pad all sequences to the same length
        max_len = max(f.shape[0] for f in features_list)
        self.features = []
        self.labels = []
        self.lengths = []

        for feat, lab in zip(features_list, labels_list):
            t, f_dim = feat.shape
            self.lengths.append(t)
            # Pad with zeros
            padded_feat = np.zeros((max_len, f_dim), dtype=np.float32)
            padded_lab = np.full(max_len, -1, dtype=np.int64)  # -1 = ignore
            if scaler is not None:
                feat = scaler.transform(feat)
            padded_feat[:t] = feat
            padded_lab[:t] = lab
            self.features.append(padded_feat)
            self.labels.append(padded_lab)

        self.features = torch.FloatTensor(np.stack(self.features))
        self.labels = torch.LongTensor(np.stack(self.labels))
        self.lengths = torch.LongTensor(self.lengths)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.lengths[idx]


def run_phase_training(
    training_run_id: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """
    Collect phase-annotated segments, train PhaseLSTM, save checkpoint.
    Called as a background job.
    """
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if not run:
            return

        if progress_callback:
            progress_callback(5.0, "Collecting phase-annotated data...")

        # Find segments with phase annotations
        annotated_seg_ids = (
            db.query(SegmentPhase.segment_id)
            .filter(SegmentPhase.human_label.isnot(None))
            .distinct()
            .all()
        )
        annotated_seg_ids = [row[0] for row in annotated_seg_ids]

        if len(annotated_seg_ids) < 5:
            run.status = "failed"
            db.commit()
            return

        features_list = []
        labels_list = []

        for seg_id in annotated_seg_ids:
            seg = db.query(Segment).filter(Segment.id == seg_id).first()
            if not seg:
                continue

            # Get phase labels for this segment
            phases = (
                db.query(SegmentPhase)
                .filter(SegmentPhase.segment_id == seg_id, SegmentPhase.human_label.isnot(None))
                .order_by(SegmentPhase.start_frame)
                .all()
            )
            if not phases:
                continue

            # Determine overall frame range from phases
            min_frame = min(p.start_frame for p in phases)
            max_frame = max(p.end_frame for p in phases)

            # Get track frames for the range
            track_frames = (
                db.query(TrackFrame)
                .filter(
                    TrackFrame.track_id == seg.track_id,
                    TrackFrame.frame_number >= min_frame,
                    TrackFrame.frame_number <= max_frame,
                )
                .order_by(TrackFrame.frame_number)
                .all()
            )

            if len(track_frames) < 10:
                continue

            # Build feature matrix and label array
            frame_features = []
            frame_labels = []
            for tf in track_frames:
                if tf.pose_features is None:
                    continue
                feat = json.loads(tf.pose_features)
                if len(feat) != settings.FEATURE_DIM:
                    continue

                # Determine which phase this frame belongs to
                frame_phase = None
                for p in phases:
                    if p.start_frame <= tf.frame_number <= p.end_frame:
                        frame_phase = PHASE_TO_IDX.get(p.phase)
                        break

                if frame_phase is None:
                    continue

                frame_features.append(feat)
                frame_labels.append(frame_phase)

            if len(frame_features) < 10:
                continue

            features_list.append(np.array(frame_features, dtype=np.float32))
            labels_list.append(np.array(frame_labels, dtype=np.int64))

        if len(features_list) < 5:
            run.status = "failed"
            db.commit()
            return

        # Count samples
        total_frames = sum(len(f) for f in features_list)
        run.positive_count = total_frames  # total labeled frames
        run.negative_count = 0

        # Split into train/val/test (by segment, not by frame)
        indices = list(range(len(features_list)))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        train_feats = [features_list[i] for i in train_idx]
        train_labs = [labels_list[i] for i in train_idx]
        val_feats = [features_list[i] for i in val_idx]
        val_labs = [labels_list[i] for i in val_idx]
        test_feats = [features_list[i] for i in test_idx]
        test_labs = [labels_list[i] for i in test_idx]

        run.train_count = sum(len(f) for f in train_feats)
        run.val_count = sum(len(f) for f in val_feats)
        run.test_count = sum(len(f) for f in test_feats)
        db.commit()

        if progress_callback:
            progress_callback(15.0, f"Training on {run.train_count} frames from {len(train_feats)} segments...")

        # Fit scaler on training data
        scaler = StandardScaler()
        all_train = np.concatenate(train_feats)
        scaler.fit(all_train)

        # Create datasets
        train_ds = PhaseDataset(train_feats, train_labs, scaler)
        val_ds = PhaseDataset(val_feats, val_labs, scaler)

        batch_size = run.batch_size
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Create model
        lstm_units = json.loads(run.lstm_units)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PhaseLSTM(
            input_dim=settings.FEATURE_DIM,
            lstm_units=lstm_units,
            dropout=run.dropout,
            use_crf=True,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=run.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(run.epochs):
            # Train
            model.train()
            train_losses = []
            for batch_x, batch_y, batch_len in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)  # (B, T, 4)
                # Build mask: True for valid frames, False for padding
                max_t = batch_x.size(1)
                mask = torch.arange(max_t, device=device).unsqueeze(0) < batch_len.unsqueeze(1).to(device)
                loss = model.crf_loss(logits, batch_y.clamp(min=0), mask)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))

            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y, batch_len in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_x)
                    max_t = batch_x.size(1)
                    mask = torch.arange(max_t, device=device).unsqueeze(0) < batch_len.unsqueeze(1).to(device)
                    loss = model.crf_loss(logits, batch_y.clamp(min=0), mask)
                    val_losses.append(loss.item())

            val_loss = float(np.mean(val_losses))
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Update DB
            run.train_loss = train_loss
            run.val_loss = val_loss
            run.best_epoch = best_epoch
            db.commit()

            pct = 15 + (epoch / run.epochs) * 70
            if progress_callback:
                progress_callback(pct, f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            if patience_counter >= 15:
                break

        # Restore best model
        if best_state:
            model.load_state_dict(best_state)
            model.to(device)

        if progress_callback:
            progress_callback(90.0, "Evaluating on test set...")

        # Evaluate
        test_ds = PhaseDataset(test_feats, test_labs, scaler)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch_x, batch_y, batch_len in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)  # (B, T, 4)
                max_t = batch_x.size(1)
                mask = torch.arange(max_t, device=device).unsqueeze(0) < batch_len.unsqueeze(1).to(device)
                decoded = model.crf_decode(logits, mask)  # list of lists
                labels = batch_y.numpy()

                for i in range(len(batch_len)):
                    length = batch_len[i].item()
                    valid_preds = decoded[i][:length]
                    valid_labels = labels[i, :length]
                    label_mask = valid_labels >= 0
                    all_preds.extend(np.array(valid_preds)[label_mask].tolist())
                    all_true.extend(valid_labels[label_mask].tolist())

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)

        accuracy = float(accuracy_score(all_true, all_preds))
        precision = float(precision_score(all_true, all_preds, average="macro", zero_division=0))
        recall = float(recall_score(all_true, all_preds, average="macro", zero_division=0))
        f1 = float(f1_score(all_true, all_preds, average="macro", zero_division=0))

        # Save checkpoint
        checkpoint_dir = Path(settings.CHECKPOINT_DIR / f"run_{run.id:03d}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(checkpoint_dir / "model.pt"))
        joblib.dump(scaler, str(checkpoint_dir / "scaler.pkl"))
        config = {
            "model_type": "phase_lstm",
            "input_dim": settings.FEATURE_DIM,
            "lstm_units": lstm_units,
            "dropout": run.dropout,
            "num_classes": PhaseLSTM.NUM_CLASSES,
            "use_crf": True,
            "best_epoch": best_epoch,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (checkpoint_dir / "config.json").write_text(json.dumps(config, indent=2))

        # Update run
        run.status = "completed"
        run.best_epoch = best_epoch
        run.train_loss = float(np.mean(train_losses))
        run.val_loss = best_val_loss
        run.test_accuracy = accuracy
        run.test_precision = precision
        run.test_recall = recall
        run.test_f1 = f1
        run.test_auc = None  # Not applicable for multi-class
        run.checkpoint_dir = str(checkpoint_dir)
        run.completed_at = datetime.now(timezone.utc)
        db.commit()

        if progress_callback:
            progress_callback(100.0, f"Phase training complete. Test F1 (macro): {f1:.3f}")

    except Exception:
        run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if run:
            run.status = "failed"
            db.commit()
        raise
    finally:
        db.close()

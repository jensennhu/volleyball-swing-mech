"""
PyTorch training loop for SpikeLSTM.

Handles: data splitting, StandardScaler fitting, training with early stopping,
evaluation, and checkpoint saving.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from spike_platform.ml.spike_lstm import SpikeLSTM
from spike_platform.ml.dataset import SpikeDataset


class SpikeTrainer:
    """Train and evaluate a SpikeLSTM model."""

    def __init__(
        self,
        input_dim: int = 33,
        lstm_units: list[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 16,
        epochs: int = 100,
        class_weight_positive: float = 2.0,
        early_stopping_patience: int = 15,
        lr_reduce_patience: int = 7,
    ):
        self.input_dim = input_dim
        self.lstm_units = lstm_units or [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight_positive = class_weight_positive
        self.early_stopping_patience = early_stopping_patience
        self.lr_reduce_patience = lr_reduce_patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpikeLSTM(
            input_dim=input_dim,
            lstm_units=self.lstm_units,
            dropout=dropout,
        ).to(self.device)
        self.scaler = StandardScaler()

        # Filled after training
        self.best_epoch = None
        self.best_val_loss = float("inf")
        self.best_state_dict = None

    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        on_epoch: Optional[Callable[[int, float, float], None]] = None,
    ) -> dict:
        """
        Full training loop with early stopping.

        Args:
            train_features: (N_train, window_size, feature_dim)
            train_labels: (N_train,)
            val_features: (N_val, window_size, feature_dim)
            val_labels: (N_val,)
            on_epoch: Callback(epoch, train_loss, val_loss)

        Returns:
            Dict with training results.
        """
        # Fit scaler on training data
        n, t, f = train_features.shape
        self.scaler.fit(train_features.reshape(-1, f))

        # Create datasets and loaders
        train_ds = SpikeDataset(train_features, train_labels, self.scaler)
        has_val = len(val_features) > 0

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = None
        if has_val:
            val_ds = SpikeDataset(val_features, val_labels, self.scaler)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Loss with class weighting
        pos_weight = torch.tensor([self.class_weight_positive], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.lr_reduce_patience
        )

        # Training loop
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)

            # Validate
            self.model.eval()
            if has_val:
                val_losses = []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        logits = self.model(batch_x)
                        loss = criterion(logits, batch_y)
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)
            else:
                val_loss = train_loss  # no val set — use train loss for early stopping
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if on_epoch:
                on_epoch(epoch + 1, train_loss, val_loss)

            if patience_counter >= self.early_stopping_patience:
                break

        # Restore best model
        if self.best_state_dict:
            self.model.load_state_dict(self.best_state_dict)
            self.model.to(self.device)

        return {
            "best_epoch": self.best_epoch,
            "train_loss": float(train_loss),
            "val_loss": float(self.best_val_loss),
            "epochs_run": epoch + 1,
        }

    def evaluate(
        self,
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict:
        """Evaluate model on test set. Returns metrics dict."""
        test_ds = SpikeDataset(test_features, test_labels, self.scaler)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(batch_y.numpy())

        all_probs = np.concatenate(all_probs).flatten()
        all_labels = np.concatenate(all_labels).flatten()
        all_preds = (all_probs >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
            "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        }

        # AUC requires both classes present
        if len(np.unique(all_labels)) > 1:
            metrics["auc"] = float(roc_auc_score(all_labels, all_probs))
        else:
            metrics["auc"] = None

        return metrics

    def save_checkpoint(self, checkpoint_dir: str):
        """Save model weights, scaler, and config to disk."""
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Model weights
        torch.save(self.model.state_dict(), str(path / "model.pt"))

        # Fitted scaler
        joblib.dump(self.scaler, str(path / "scaler.pkl"))

        # Config
        config = {
            "input_dim": self.input_dim,
            "lstm_units": self.lstm_units,
            "dropout": self.dropout,
            "best_epoch": self.best_epoch,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str) -> "SpikeTrainer":
        """Load a trained model from checkpoint."""
        path = Path(checkpoint_dir)

        config = json.loads((path / "config.json").read_text())
        trainer = cls(
            input_dim=config["input_dim"],
            lstm_units=config["lstm_units"],
            dropout=config["dropout"],
        )

        state_dict = torch.load(
            str(path / "model.pt"),
            map_location=trainer.device,
            weights_only=True,
        )
        trainer.model.load_state_dict(state_dict)
        trainer.scaler = joblib.load(str(path / "scaler.pkl"))

        return trainer

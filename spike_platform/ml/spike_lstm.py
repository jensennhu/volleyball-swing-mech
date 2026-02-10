"""
PyTorch LSTM model for binary spike detection.

Architecture ported from scripts/02b_train_spike_detector.py (Keras):
    Input(40, 33) → LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3)
        → Linear(16, ReLU) → Dropout(0.15) → Linear(1, Sigmoid)
"""

import torch
import torch.nn as nn


class SpikeLSTM(nn.Module):
    """Binary spike detector LSTM."""

    def __init__(
        self,
        input_dim: int = 33,
        lstm_units: list[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        lstm_units = lstm_units or [64, 32]

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        prev_dim = input_dim
        for units in lstm_units:
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=prev_dim,
                    hidden_size=units,
                    batch_first=True,
                )
            )
            self.dropout_layers.append(nn.Dropout(dropout))
            prev_dim = units

        # Fully connected head
        self.fc1 = nn.Linear(lstm_units[-1], 16)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout / 2)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) e.g. (N, 40, 33)
        Returns:
            (batch_size, 1) spike probability (pre-sigmoid logits for BCEWithLogitsLoss,
             or apply sigmoid manually for inference)
        """
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)
            x = dropout(x)

        # Take the last timestep output
        x = x[:, -1, :]

        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)  # raw logits
        return x

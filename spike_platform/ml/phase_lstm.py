"""
PyTorch LSTM model for per-frame spike phase classification.

Classifies each frame in a sequence as one of 4 phases:
    0=approach, 1=jump, 2=swing, 3=land

Architecture:
    Input(T, 33) -> LSTM(64) -> Dropout -> LSTM(32) -> Dropout
        -> Linear(32, 16, ReLU) -> Dropout -> Linear(16, 4)

Optional CRF layer enforces valid phase transition ordering.
"""

import torch
import torch.nn as nn

from spike_platform.ml.crf import LinearChainCRF


class PhaseLSTM(nn.Module):
    """Per-frame phase classifier LSTM with optional CRF."""

    NUM_CLASSES = 4  # approach, jump, swing, land

    def __init__(
        self,
        input_dim: int = 33,
        lstm_units: list[int] = None,
        dropout: float = 0.3,
        use_crf: bool = False,
    ):
        super().__init__()
        lstm_units = lstm_units or [64, 32]
        self.use_crf = use_crf

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

        # Per-frame classification head
        self.fc1 = nn.Linear(lstm_units[-1], 16)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout / 2)
        self.fc2 = nn.Linear(16, self.NUM_CLASSES)

        # Optional CRF layer
        if use_crf:
            self.crf = LinearChainCRF(self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            (batch_size, seq_len, NUM_CLASSES) — raw logits per frame
        """
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)
            x = dropout(x)

        # Apply FC to every timestep
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

    def crf_loss(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute CRF negative log-likelihood loss.

        Args:
            emissions: (B, T, NUM_CLASSES) — logits from forward()
            tags: (B, T) — ground truth labels
            mask: (B, T) — boolean mask (True=valid, False=padding)
        """
        return self.crf(emissions, tags, mask)

    def crf_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decode using CRF transition constraints.

        Args:
            emissions: (B, T, NUM_CLASSES) — logits from forward()
            mask: (B, T) — boolean mask
        Returns:
            List of B tag sequences
        """
        return self.crf.decode(emissions, mask)

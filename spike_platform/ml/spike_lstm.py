"""
PyTorch LSTM model for binary spike detection.

Architecture:
    Input(40, input_dim) → BiLSTM(64) → Dropout → BiLSTM(32) → Dropout
        → Temporal Attention → Linear(16, ReLU) → Dropout → Linear(1)

Supports bidirectional LSTM and temporal attention (both configurable
for backward compatibility with older checkpoints).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeLSTM(nn.Module):
    """Binary spike detector with bidirectional LSTM and temporal attention."""

    def __init__(
        self,
        input_dim: int = 33,
        lstm_units: list[int] = None,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()
        lstm_units = lstm_units or [64, 32]
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        num_directions = 2 if bidirectional else 1

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
                    bidirectional=bidirectional,
                )
            )
            self.dropout_layers.append(nn.Dropout(dropout))
            prev_dim = units * num_directions

        # Last LSTM output dim (used by attention and FC head)
        last_dim = lstm_units[-1] * num_directions

        # Temporal attention: learns per-frame importance weights
        if use_attention:
            self.attention_fc = nn.Linear(last_dim, 1)

        # Fully connected head
        self.fc1 = nn.Linear(last_dim, 16)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout / 2)
        self.fc2 = nn.Linear(16, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim) e.g. (N, 40, 33)
            return_attention: if True, also return attention weights

        Returns:
            logits: (batch_size, 1) pre-sigmoid logits
            attn_weights: (batch_size, seq_len) attention weights (only if return_attention=True)
        """
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)
            x = dropout(x)

        if self.use_attention:
            # x: (batch, seq_len, last_dim)
            attn_scores = self.attention_fc(x).squeeze(-1)  # (batch, seq_len)
            attn_weights = F.softmax(attn_scores, dim=-1)   # (batch, seq_len)
            # Weighted sum: (batch, seq_len, last_dim) * (batch, seq_len, 1) → sum → (batch, last_dim)
            x = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        else:
            # Fallback: take last timestep (original behavior)
            attn_weights = None
            x = x[:, -1, :]

        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)  # raw logits

        if return_attention and attn_weights is not None:
            return x, attn_weights
        return x

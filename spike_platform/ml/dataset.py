"""
PyTorch Dataset for spike detection segments.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class SpikeDataset(Dataset):
    """Dataset wrapping segment feature matrices and binary labels."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        scaler: StandardScaler = None,
    ):
        """
        Args:
            features: (N, window_size, feature_dim) e.g. (N, 40, 33)
            labels: (N,) binary labels (0 or 1)
            scaler: If provided, transform features. Applied per-feature across time.
        """
        if scaler is not None:
            n, t, f = features.shape
            features = scaler.transform(features.reshape(-1, f)).reshape(n, t, f)

        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

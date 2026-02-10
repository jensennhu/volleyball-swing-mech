"""
Standalone OSNet-based person re-identification encoder.

Extracts 512-dim appearance embeddings from person crops.
Used in track post-processing to improve merge/split decisions
via cosine similarity between track embeddings.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import torchreid


# Standard person ReID input dimensions
_REID_HEIGHT = 256
_REID_WIDTH = 128


class ReIDEncoder:
    """OSNet x0.25 person re-identification encoder (~0.5M params, 512-dim)."""

    def __init__(self):
        self.model = torchreid.models.build_model(
            name="osnet_x0_25",
            num_classes=1,
            pretrained=True,
            loss="softmax",
        )
        self.model.eval()

        from torchreid.reid.data.transforms import build_transforms
        _, test_transform = build_transforms(
            is_train=False,
            height=_REID_HEIGHT,
            width=_REID_WIDTH,
        )
        self.transform = test_transform

    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Extract 512-dim L2-normalized embedding from a person crop.

        Args:
            crop_bgr: Person crop in BGR format (HWC).

        Returns:
            L2-normalized embedding of shape (512,).
        """
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = self.transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(tensor)

        return F.normalize(embedding, dim=1).squeeze().cpu().numpy()

    def encode_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for multiple crops in one forward pass.

        Args:
            crops_bgr: List of person crops in BGR format.

        Returns:
            L2-normalized embeddings of shape (N, 512).
        """
        if not crops_bgr:
            return np.empty((0, 512), dtype=np.float32)

        tensors = []
        for crop in crops_bgr:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensors.append(self.transform(pil_img))

        batch = torch.stack(tensors)
        with torch.no_grad():
            embeddings = self.model(batch)

        return F.normalize(embeddings, dim=1).cpu().numpy()

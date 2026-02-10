"""
Central configuration for the volleyball spike detector platform.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = field(default=None)
    FEATURES_DIR: Path = field(default=None)
    CHECKPOINT_DIR: Path = field(default=None)
    DB_PATH: Path = field(default=None)

    # Detection
    YOLO_MODEL: str = "yolov8n.pt"
    DETECTION_CONFIDENCE: float = 0.3
    DETECTION_IOU: float = 0.5
    PERSON_CLASS_ID: int = 0  # COCO class ID for person

    # Pose
    POSE_MIN_DETECTION_CONF: float = 0.5
    POSE_MIN_TRACKING_CONF: float = 0.5

    # Segmentation
    WINDOW_SIZE: int = 40
    WINDOW_STRIDE: int = 10
    MIN_TRACK_FRAMES: int = 40  # Skip tracks shorter than one window

    # Training defaults
    DEFAULT_LSTM_UNITS: list = field(default_factory=lambda: [64, 32])
    DEFAULT_DROPOUT: float = 0.3
    DEFAULT_LR: float = 0.001
    DEFAULT_EPOCHS: int = 100
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_CLASS_WEIGHT_POSITIVE: float = 2.0
    EARLY_STOPPING_PATIENCE: int = 15
    LR_REDUCE_PATIENCE: int = 7

    # Track post-processing
    TRACK_MERGE_MAX_GAP: int = 40        # max frames between tracks to consider merging
    TRACK_MERGE_MIN_IOU: float = 0.2     # min bbox IoU to merge
    TRACK_SWITCH_MAX_JUMP: float = 2.0   # max bbox-center displacement (in bbox heights)

    # Features
    FEATURE_DIM: int = 33

    def __post_init__(self):
        if self.UPLOAD_DIR is None:
            self.UPLOAD_DIR = self.BASE_DIR / "data" / "uploads"
        if self.FEATURES_DIR is None:
            self.FEATURES_DIR = self.BASE_DIR / "data" / "features"
        if self.CHECKPOINT_DIR is None:
            self.CHECKPOINT_DIR = self.BASE_DIR / "checkpoints"
        if self.DB_PATH is None:
            self.DB_PATH = self.BASE_DIR / "data" / "platform.db"

        # Ensure directories exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def DB_URL(self) -> str:
        return f"sqlite:///{self.DB_PATH}"


settings = Settings()

"""Pydantic schemas for training and inference endpoints."""

from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class TrainingConfig(BaseModel):
    task_type: str = "spike_detection"  # spike_detection|phase_classification
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    lstm_units: list[int] = [64, 32]
    dropout: float = 0.3
    class_weight_positive: float = 2.0
    early_stopping_patience: int = 15


class TrainingRunResponse(BaseModel):
    id: int
    status: str
    task_type: str
    epochs: int
    batch_size: int
    learning_rate: float
    lstm_units: str  # JSON string
    window_size: int
    train_count: Optional[int] = None
    val_count: Optional[int] = None
    test_count: Optional[int] = None
    positive_count: Optional[int] = None
    negative_count: Optional[int] = None
    best_epoch: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None
    test_auc: Optional[float] = None
    checkpoint_dir: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class TrainingStatusResponse(BaseModel):
    status: str
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None


class InferenceRequest(BaseModel):
    video_id: str
    training_run_id: Optional[int] = None


class PhaseInferenceRequest(BaseModel):
    video_id: str
    training_run_id: Optional[int] = None


class InferenceResponse(BaseModel):
    updated_segments: int


class InferenceStatsResponse(BaseModel):
    total: int
    labeled: int
    spike: int
    non_spike: int
    unlabeled: int


# ─── Phase Classification Schemas ─────────────────────────────────

class PhaseLabel(BaseModel):
    phase: str  # approach|jump|swing|land
    start_frame: int
    end_frame: int


class PhaseLabelRequest(BaseModel):
    phases: list[PhaseLabel]


class PhaseLabelResponse(BaseModel):
    id: int
    segment_id: int
    phase: str
    start_frame: int
    end_frame: int
    human_label: Optional[str] = None
    confidence: Optional[float] = None
    model_run_id: Optional[int] = None

    model_config = {"from_attributes": True}


class FrameResponse(BaseModel):
    frame_number: int
    bbox: list[float]  # [x1, y1, x2, y2]

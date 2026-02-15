"""
SQLAlchemy ORM models for the spike detector platform.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Boolean, Column, String, Integer, Float, Text, DateTime, Index, ForeignKey
)
from sqlalchemy.orm import relationship

from spike_platform.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True)  # UUID
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    fps = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    frame_count = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="uploaded")  # uploaded|processing|processed|predicted|error
    error_message = Column(Text, nullable=True)
    video_group = Column(String, nullable=True)  # e.g. "hitting_lines", "game_play"
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    tracks = relationship("Track", back_populates="video", cascade="all, delete-orphan")
    segments = relationship("Segment", back_populates="video", cascade="all, delete-orphan")


class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    track_id = Column(Integer, nullable=False)  # ByteTrack assigned ID (per-video)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    frame_count = Column(Integer, nullable=False)
    avg_confidence = Column(Float, nullable=True)
    role = Column(String, nullable=True, default="unknown")  # player|non_player|unknown
    role_source = Column(String, nullable=True)  # heuristic|human
    created_at = Column(DateTime, default=_utcnow)

    video = relationship("Video", back_populates="tracks")
    frames = relationship("TrackFrame", back_populates="track", cascade="all, delete-orphan")
    segments = relationship("Segment", back_populates="track", cascade="all, delete-orphan")


class TrackFrame(Base):
    __tablename__ = "track_frames"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    detection_confidence = Column(Float, nullable=True)
    pose_features = Column(Text, nullable=True)  # JSON array of 33 floats
    pose_confidence = Column(Float, nullable=True)

    track = relationship("Track", back_populates="frames")

    __table_args__ = (
        Index("ix_track_frames_track_frame", "track_id", "frame_number"),
    )


class Segment(Base):
    __tablename__ = "segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False)
    video_id = Column(String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    window_size = Column(Integer, nullable=False)

    # Model predictions (filled after inference)
    prediction = Column(Integer, nullable=True)  # 0=non-spike, 1=spike
    confidence = Column(Float, nullable=True)
    model_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True)

    # Human labels (filled during review)
    human_label = Column(Integer, nullable=True)  # 0=non-spike, 1=spike
    labeled_at = Column(DateTime, nullable=True)

    # Cached feature blob
    features_path = Column(String, nullable=True)  # path to .npy file

    created_at = Column(DateTime, default=_utcnow)

    track = relationship("Track", back_populates="segments")
    video = relationship("Video", back_populates="segments")
    phases = relationship("SegmentPhase", back_populates="segment", cascade="all, delete-orphan")


class SegmentPhase(Base):
    """v2 extension: phase labels within spike segments. Empty in v1."""
    __tablename__ = "segment_phases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(Integer, ForeignKey("segments.id", ondelete="CASCADE"), nullable=False)
    phase = Column(String, nullable=False)  # approach|jump|swing|land
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=True)
    human_label = Column(String, nullable=True)
    model_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    segment = relationship("Segment", back_populates="phases")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String, nullable=False, default="running")  # running|completed|failed
    task_type = Column(String, nullable=False, default="spike_detection")  # spike_detection|phase_classification

    # Config
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    lstm_units = Column(String, nullable=False)  # JSON: [64, 32]
    window_size = Column(Integer, nullable=False)
    dropout = Column(Float, nullable=False, default=0.3)

    # Dataset stats
    train_count = Column(Integer, nullable=True)
    val_count = Column(Integer, nullable=True)
    test_count = Column(Integer, nullable=True)
    positive_count = Column(Integer, nullable=True)
    negative_count = Column(Integer, nullable=True)

    # Results (filled after training)
    best_epoch = Column(Integer, nullable=True)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    test_precision = Column(Float, nullable=True)
    test_recall = Column(Float, nullable=True)
    test_f1 = Column(Float, nullable=True)
    test_auc = Column(Float, nullable=True)

    # Artifacts
    checkpoint_dir = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    balance_by_group = Column(Boolean, nullable=True, default=False)

    started_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)

"""Training and inference endpoints."""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from spike_platform.config import settings
from spike_platform.database import get_db
from spike_platform.models.db_models import Segment, SegmentPhase, Track, TrainingRun
from spike_platform.schemas.training import (
    TrainingConfig,
    TrainingRunResponse,
    TrainingStatusResponse,
    InferenceRequest,
    InferenceResponse,
    InferenceStatsResponse,
    PhaseInferenceRequest,
    GroupMetricsResponse,
)
from spike_platform.worker import worker

router = APIRouter()


@router.post("/training/start", response_model=TrainingRunResponse)
def start_training(config: TrainingConfig, db: Session = Depends(get_db)):
    """Start a training run using all labeled segments."""
    valid_types = ("spike_detection", "phase_classification", "role_classification")
    if config.task_type not in valid_types:
        raise HTTPException(400, f"task_type must be one of {valid_types}")

    # Check we have enough labeled data
    if config.task_type == "spike_detection":
        labeled_count = (
            db.query(func.count(Segment.id))
            .filter(Segment.human_label.isnot(None))
            .scalar()
        )
        if labeled_count < 10:
            raise HTTPException(
                400,
                f"Need at least 10 labeled segments to train, have {labeled_count}",
            )
    elif config.task_type == "phase_classification":
        labeled_count = (
            db.query(func.count(func.distinct(SegmentPhase.segment_id)))
            .filter(SegmentPhase.human_label.isnot(None))
            .scalar()
        )
        if labeled_count < 5:
            raise HTTPException(
                400,
                f"Need at least 5 phase-annotated segments to train, have {labeled_count}",
            )
    else:  # role_classification
        labeled_count = (
            db.query(func.count(Track.id))
            .filter(Track.role_source == "human")
            .scalar()
        )
        if labeled_count < 20:
            raise HTTPException(
                400,
                f"Need at least 20 human-labeled tracks to train role classifier, have {labeled_count}",
            )

    if worker.is_busy:
        raise HTTPException(409, "A job is already running")

    # Create training run record
    run = TrainingRun(
        status="running",
        task_type=config.task_type,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        lstm_units=json.dumps(config.lstm_units),
        window_size=settings.WINDOW_SIZE,
        dropout=config.dropout,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    if config.task_type == "spike_detection":
        from spike_platform.services.training import run_training
        worker.submit(run_training, training_run_id=run.id)
    elif config.task_type == "phase_classification":
        from spike_platform.services.phase_training import run_phase_training
        worker.submit(run_phase_training, training_run_id=run.id)
    else:
        from spike_platform.services.track_classifier import train_role_classifier
        worker.submit(train_role_classifier, training_run_id=run.id)

    return run


@router.get("/training/runs", response_model=list[TrainingRunResponse])
def list_training_runs(task_type: str = None, db: Session = Depends(get_db)):
    """List all training runs, optionally filtered by task_type."""
    q = db.query(TrainingRun)
    if task_type:
        q = q.filter(TrainingRun.task_type == task_type)
    runs = q.order_by(TrainingRun.started_at.desc()).all()
    return runs


@router.get("/training/runs/{run_id}", response_model=TrainingRunResponse)
def get_training_run(run_id: int, db: Session = Depends(get_db)):
    """Get details for a specific training run."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(404, "Training run not found")
    return run


@router.get("/training/runs/{run_id}/status", response_model=TrainingStatusResponse)
def get_training_status(run_id: int, db: Session = Depends(get_db)):
    """Poll training progress."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(404, "Training run not found")
    return TrainingStatusResponse(
        status=run.status,
        best_epoch=run.best_epoch,
        total_epochs=run.epochs,
        train_loss=run.train_loss,
        val_loss=run.val_loss,
    )


@router.get("/training/group-metrics", response_model=GroupMetricsResponse)
def get_group_metrics(
    training_run_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    """Compute per-video-group metrics for a training run's predictions."""
    from spike_platform.services.group_metrics import compute_group_metrics

    return compute_group_metrics(training_run_id, db)


@router.post("/inference/run", response_model=InferenceResponse)
def run_inference(request: InferenceRequest, db: Session = Depends(get_db)):
    """Re-run inference on a video with a trained model."""
    if worker.is_busy:
        raise HTTPException(409, "A job is already running")

    from spike_platform.services.inference import run_inference_on_video
    worker.submit(
        run_inference_on_video,
        video_id=request.video_id,
        training_run_id=request.training_run_id,
    )
    return InferenceResponse(updated_segments=0)  # actual count updated async


@router.post("/inference/phase-run", response_model=InferenceResponse)
def run_phase_inference_endpoint(request: PhaseInferenceRequest, db: Session = Depends(get_db)):
    """Run phase inference on a video's spike segments with a trained phase model."""
    if worker.is_busy:
        raise HTTPException(409, "A job is already running")

    from spike_platform.services.phase_inference import run_phase_inference
    worker.submit(
        run_phase_inference,
        video_id=request.video_id,
        training_run_id=request.training_run_id,
    )
    return InferenceResponse(updated_segments=0)


@router.get("/inference/stats", response_model=InferenceStatsResponse)
def get_inference_stats(db: Session = Depends(get_db)):
    """Get label distribution stats across all segments."""
    total = db.query(func.count(Segment.id)).scalar()
    labeled = (
        db.query(func.count(Segment.id))
        .filter(Segment.human_label.isnot(None))
        .scalar()
    )
    spike = (
        db.query(func.count(Segment.id))
        .filter(Segment.human_label == 1)
        .scalar()
    )
    non_spike = (
        db.query(func.count(Segment.id))
        .filter(Segment.human_label == 0)
        .scalar()
    )
    return InferenceStatsResponse(
        total=total,
        labeled=labeled,
        spike=spike,
        non_spike=non_spike,
        unlabeled=total - labeled,
    )

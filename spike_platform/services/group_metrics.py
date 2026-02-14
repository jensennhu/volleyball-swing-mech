"""Per-video-group metrics computation for spike detection analysis."""

import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sqlalchemy.orm import Session

from spike_platform.models.db_models import Segment, Video, Track, TrainingRun
from spike_platform.schemas.training import (
    CalibrationBucket,
    VideoMetrics,
    GroupMetrics,
    GroupMetricsResponse,
)

logger = logging.getLogger(__name__)


def compute_group_metrics(
    training_run_id: int | None,
    db: Session,
) -> GroupMetricsResponse:
    """Compute per-video-group metrics for a training run's predictions.

    Args:
        training_run_id: Specific run to analyze, or None for latest completed spike_detection run.
        db: Database session.

    Returns:
        GroupMetricsResponse with per-group and overall metrics.
    """
    # Resolve training run
    if training_run_id is not None:
        run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
    else:
        run = (
            db.query(TrainingRun)
            .filter(
                TrainingRun.task_type == "spike_detection",
                TrainingRun.status == "completed",
            )
            .order_by(TrainingRun.completed_at.desc())
            .first()
        )

    if not run:
        return GroupMetricsResponse(training_run_id=training_run_id or 0, groups=[])

    # Query segments that have both human_label and prediction
    # Try filtering by model_run_id first; if empty, fall back to any predictions
    base_query = (
        db.query(Segment)
        .join(Track, Segment.track_id == Track.id)
        .filter(
            Segment.human_label.isnot(None),
            Segment.prediction.isnot(None),
            Track.role != "non_player",
        )
    )

    segments = base_query.filter(Segment.model_run_id == run.id).all()

    if not segments:
        # Fall back: show metrics for ALL segments with predictions + labels
        segments = base_query.all()

    if not segments:
        return GroupMetricsResponse(training_run_id=run.id, groups=[])

    # Cache video info
    video_cache: dict[str, Video] = {}
    for seg in segments:
        if seg.video_id not in video_cache:
            video_cache[seg.video_id] = (
                db.query(Video).filter(Video.id == seg.video_id).first()
            )

    # Group segments by video_group
    # Structure: {group_name: {video_id: [(human_label, prediction, confidence)]}}
    grouped: dict[str, dict[str, list[tuple[int, int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seg in segments:
        video = video_cache.get(seg.video_id)
        group_name = (video.video_group if video and video.video_group else "ungrouped")
        conf = seg.confidence if seg.confidence is not None else 0.5
        grouped[group_name][seg.video_id].append(
            (seg.human_label, seg.prediction, conf)
        )

    # Compute metrics per group + overall
    all_groups: list[GroupMetrics] = []

    # Add an "overall" entry by combining all data
    groups_to_compute = [("overall", _flatten_group(grouped))]
    groups_to_compute.extend(sorted(grouped.items()))

    for group_name, video_data in groups_to_compute:
        gm = _compute_single_group(group_name, video_data, video_cache)
        if gm is not None:
            all_groups.append(gm)

    return GroupMetricsResponse(training_run_id=run.id, groups=all_groups)


def _flatten_group(
    grouped: dict[str, dict[str, list[tuple[int, int, float]]]],
) -> dict[str, list[tuple[int, int, float]]]:
    """Flatten all groups into a single group keyed by video_id."""
    flat: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for video_data in grouped.values():
        for vid, entries in video_data.items():
            flat[vid].extend(entries)
    return dict(flat)


def _compute_single_group(
    group_name: str,
    video_data: dict[str, list[tuple[int, int, float]]],
    video_cache: dict[str, Video],
) -> GroupMetrics | None:
    """Compute metrics for a single group."""
    all_labels = []
    all_preds = []
    all_confs = []

    for entries in video_data.values():
        for label, pred, conf in entries:
            all_labels.append(label)
            all_preds.append(pred)
            all_confs.append(conf)

    if not all_labels:
        return None

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_conf = np.array(all_confs)

    # Confusion matrix components
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Calibration: 10 bins by confidence
    calibration = []
    bin_edges = np.linspace(0.0, 1.0, 11)
    for i in range(10):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        mask = (y_conf >= lo) & (y_conf < hi) if i < 9 else (y_conf >= lo) & (y_conf <= hi)
        count = int(np.sum(mask))
        actual_rate = float(np.mean(y_true[mask])) if count > 0 else None
        calibration.append(CalibrationBucket(
            bin_start=round(lo, 2),
            bin_end=round(hi, 2),
            count=count,
            actual_positive_rate=round(actual_rate, 4) if actual_rate is not None else None,
        ))

    # Per-video metrics
    per_video: list[VideoMetrics] = []
    video_f1s = []
    for vid, entries in sorted(video_data.items()):
        v_labels = np.array([e[0] for e in entries])
        v_preds = np.array([e[1] for e in entries])
        v_prec = float(precision_score(v_labels, v_preds, zero_division=0))
        v_rec = float(recall_score(v_labels, v_preds, zero_division=0))
        v_f1 = float(f1_score(v_labels, v_preds, zero_division=0))
        video_f1s.append(v_f1)

        video = video_cache.get(vid)
        per_video.append(VideoMetrics(
            video_id=vid,
            filename=video.filename if video else vid,
            n_samples=len(entries),
            precision=round(v_prec, 4),
            recall=round(v_rec, 4),
            f1=round(v_f1, 4),
        ))

    f1_mean = float(np.mean(video_f1s)) if video_f1s else 0.0
    f1_std = float(np.std(video_f1s)) if len(video_f1s) > 1 else 0.0

    return GroupMetrics(
        group_name=group_name,
        total=len(all_labels),
        spike_count=int(np.sum(y_true == 1)),
        non_spike_count=int(np.sum(y_true == 0)),
        tp=tp, fp=fp, tn=tn, fn=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        fpr=round(fpr, 4),
        fnr=round(fnr, 4),
        calibration=calibration,
        per_video=per_video,
        f1_mean=round(f1_mean, 4),
        f1_std=round(f1_std, 4),
    )

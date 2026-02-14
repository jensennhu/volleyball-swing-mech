"""
Track role classification: player vs non-player.

Uses heuristic signals (bbox area, pose confidence, movement, position)
and optionally a trained RandomForest classifier learned from human labels.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sqlalchemy.orm import Session

from spike_platform.config import settings
from spike_platform.database import SessionLocal
from spike_platform.models.db_models import Track, TrackFrame, TrainingRun, Video

logger = logging.getLogger(__name__)

# Feature names in the order used by the ML classifier
ROLE_FEATURE_NAMES = [
    "median_bbox_area",
    "median_pose_confidence",
    "median_y_position",
    "movement_variance",
    "bbox_aspect_ratio",
    "track_duration_ratio",
    "vertical_range",
]


def compute_track_stats(
    track_id: int, db: Session, video_width: int, video_height: int,
    video_frame_count: int = 0,
) -> dict:
    """Compute per-track statistics for classification.

    Returns dict with median_bbox_area, median_pose_confidence, median_y_position,
    movement_variance, bbox_aspect_ratio, track_duration_ratio, vertical_range.
    """
    frames = (
        db.query(TrackFrame)
        .filter(TrackFrame.track_id == track_id)
        .order_by(TrackFrame.frame_number)
        .all()
    )

    defaults = {
        "median_bbox_area": 0.0,
        "median_pose_confidence": 0.0,
        "median_y_position": 0.5,
        "movement_variance": 0.0,
        "bbox_aspect_ratio": 1.0,
        "track_duration_ratio": 0.0,
        "vertical_range": 0.0,
    }

    if not frames:
        return defaults

    frame_area = video_width * video_height if video_width and video_height else 1.0

    bbox_areas = []
    pose_confs = []
    y_positions = []
    aspect_ratios = []
    centers = []  # (cx, cy, bbox_h) for movement variance

    for f in frames:
        w = f.bbox_x2 - f.bbox_x1
        h = f.bbox_y2 - f.bbox_y1
        bbox_areas.append((w * h) / frame_area)
        cy = ((f.bbox_y1 + f.bbox_y2) / 2) / video_height if video_height else 0.5
        y_positions.append(cy)
        if f.pose_confidence is not None:
            pose_confs.append(f.pose_confidence)
        if h > 0:
            aspect_ratios.append(w / h)
        cx = ((f.bbox_x1 + f.bbox_x2) / 2) / video_width if video_width else 0.5
        centers.append((cx, cy, h / video_height if video_height else 0.01))

    # Movement variance: variance of frame-to-frame center displacement (normalized by bbox height)
    displacements = []
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        bbox_h = max(centers[i][2], 0.001)
        displacements.append((dx**2 + dy**2) ** 0.5 / bbox_h)
    movement_var = float(np.var(displacements)) if displacements else 0.0

    # Track duration ratio
    duration_ratio = len(frames) / video_frame_count if video_frame_count > 0 else 0.0

    # Vertical range (max_y - min_y)
    vert_range = max(y_positions) - min(y_positions) if y_positions else 0.0

    return {
        "median_bbox_area": float(median(bbox_areas)) if bbox_areas else 0.0,
        "median_pose_confidence": float(median(pose_confs)) if pose_confs else 0.0,
        "median_y_position": float(median(y_positions)) if y_positions else 0.5,
        "movement_variance": movement_var,
        "bbox_aspect_ratio": float(median(aspect_ratios)) if aspect_ratios else 1.0,
        "track_duration_ratio": float(duration_ratio),
        "vertical_range": float(vert_range),
    }


def _compute_all_track_stats(video: Video, tracks: list[Track], db: Session) -> dict:
    """Compute stats for all tracks in a video. Returns {track_id: stats_dict}."""
    video_w = video.width or 1920
    video_h = video.height or 1080
    video_fc = video.frame_count or 0
    return {
        track.id: compute_track_stats(track.id, db, video_w, video_h, video_fc)
        for track in tracks
    }


def _stats_to_feature_vector(stats: dict) -> list[float]:
    """Convert stats dict to ordered feature vector for ML classifier."""
    return [stats[name] for name in ROLE_FEATURE_NAMES]


def classify_tracks(video_id: str, db: Session) -> dict:
    """Auto-classify tracks in a video as player or non_player.

    Uses per-video relative thresholds (camera angles vary) combined
    with absolute pose confidence thresholds and movement/position signals.

    Preserves tracks where role_source='human' (manual overrides).

    Returns dict with classification counts.
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        return {"player": 0, "non_player": 0, "skipped": 0}

    tracks = db.query(Track).filter(Track.video_id == video_id).all()
    if not tracks:
        return {"player": 0, "non_player": 0, "skipped": 0}

    track_stats = _compute_all_track_stats(video, tracks, db)

    # Compute video-wide median bbox area for relative thresholds
    all_areas = [s["median_bbox_area"] for s in track_stats.values() if s["median_bbox_area"] > 0]
    video_median_area = float(median(all_areas)) if all_areas else 0.0

    area_threshold = video_median_area * settings.TRACK_ROLE_BBOX_AREA_RATIO
    pose_threshold = settings.TRACK_ROLE_POSE_CONF_THRESHOLD

    counts = {"player": 0, "non_player": 0, "skipped": 0}

    for track in tracks:
        # Preserve human overrides
        if track.role_source == "human":
            counts["skipped"] += 1
            continue

        stats = track_stats[track.id]
        is_small = stats["median_bbox_area"] < area_threshold
        is_low_pose = stats["median_pose_confidence"] < pose_threshold

        if is_small and is_low_pose:
            track.role = "non_player"
        elif stats["median_bbox_area"] < video_median_area * 0.3:
            # Very small relative to video — almost certainly not a player
            track.role = "non_player"
        elif stats["median_y_position"] < 0.15:
            # Top of frame — likely spectators/audience
            track.role = "non_player"
        elif stats["movement_variance"] < 0.001 and stats["track_duration_ratio"] > 0.7:
            # Nearly stationary for most of the video — likely ref or camera-visible staff
            track.role = "non_player"
        else:
            track.role = "player"

        track.role_source = "heuristic"
        counts[track.role] += 1

    db.commit()

    logger.info(
        f"Track classification for video {video_id}: "
        f"{counts['player']} players, {counts['non_player']} non-players, "
        f"{counts['skipped']} human-labeled (preserved)"
    )

    return counts


def train_role_classifier(
    training_run_id: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """Train a RandomForest role classifier from human-labeled tracks.

    Called as a background job via the worker.
    """
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if not run:
            return

        if progress_callback:
            progress_callback(10.0, "Collecting human-labeled tracks...")

        # Get all tracks with human role labels
        labeled_tracks = (
            db.query(Track)
            .filter(Track.role_source == "human")
            .all()
        )

        if len(labeled_tracks) < 20:
            run.status = "failed"
            run.notes = f"Need at least 20 human-labeled tracks, have {len(labeled_tracks)}"
            db.commit()
            if progress_callback:
                progress_callback(100.0, run.notes)
            return

        if progress_callback:
            progress_callback(20.0, f"Computing features for {len(labeled_tracks)} tracks...")

        # Build feature matrix and labels
        features = []
        labels = []
        for track in labeled_tracks:
            video = db.query(Video).filter(Video.id == track.video_id).first()
            if not video:
                continue
            video_w = video.width or 1920
            video_h = video.height or 1080
            video_fc = video.frame_count or 0
            stats = compute_track_stats(track.id, db, video_w, video_h, video_fc)
            features.append(_stats_to_feature_vector(stats))
            labels.append(1 if track.role == "player" else 0)

        X = np.array(features)
        y = np.array(labels)

        run.positive_count = int(np.sum(y == 1))  # player count
        run.negative_count = int(np.sum(y == 0))  # non-player count
        run.train_count = len(y)
        run.val_count = 0
        run.test_count = 0
        db.commit()

        if progress_callback:
            progress_callback(40.0, f"Training on {len(y)} tracks ({run.positive_count}P / {run.negative_count}NP)...")

        # Train with cross-validation for metrics
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        n_folds = min(5, len(y))
        if n_folds >= 2:
            y_pred_cv = cross_val_predict(clf, X, y, cv=n_folds)
            accuracy = float(accuracy_score(y, y_pred_cv))
            precision = float(precision_score(y, y_pred_cv, zero_division=0))
            recall = float(recall_score(y, y_pred_cv, zero_division=0))
            f1_val = float(f1_score(y, y_pred_cv, zero_division=0))
        else:
            accuracy = precision = recall = f1_val = None

        # Fit final model on all data
        clf.fit(X, y)

        if progress_callback:
            progress_callback(80.0, "Saving model...")

        # Save checkpoint
        checkpoint_dir = Path(settings.CHECKPOINT_DIR / f"run_{run.id:03d}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, str(checkpoint_dir / "role_model.pkl"))
        config = {
            "model_type": "role_classifier",
            "algorithm": "RandomForestClassifier",
            "n_estimators": 100,
            "feature_names": ROLE_FEATURE_NAMES,
            "n_samples": len(y),
            "n_players": int(np.sum(y == 1)),
            "n_non_players": int(np.sum(y == 0)),
            "cv_folds": n_folds,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (checkpoint_dir / "config.json").write_text(json.dumps(config, indent=2))

        # Update run
        run.status = "completed"
        run.best_epoch = 1
        run.train_loss = None
        run.val_loss = None
        run.test_accuracy = accuracy
        run.test_precision = precision
        run.test_recall = recall
        run.test_f1 = f1_val
        run.test_auc = None
        run.checkpoint_dir = str(checkpoint_dir)
        run.completed_at = datetime.now(timezone.utc)
        db.commit()

        f1_str = f"{f1_val:.3f}" if f1_val is not None else "N/A"
        if progress_callback:
            progress_callback(100.0, f"Role classifier trained. CV F1: {f1_str} ({len(y)} tracks)")

    except Exception:
        run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if run:
            run.status = "failed"
            db.commit()
        raise
    finally:
        db.close()


def classify_tracks_ml(video_id: str, db: Session) -> dict:
    """Classify tracks using trained ML model, falling back to heuristic.

    Looks for the latest completed role_classification TrainingRun.
    If none exists, delegates to classify_tracks() heuristic.
    """
    # Find latest completed role classifier
    run = (
        db.query(TrainingRun)
        .filter(TrainingRun.task_type == "role_classification", TrainingRun.status == "completed")
        .order_by(TrainingRun.completed_at.desc())
        .first()
    )

    if not run or not run.checkpoint_dir:
        logger.info(f"No trained role classifier found, using heuristic for video {video_id}")
        return classify_tracks(video_id, db)

    model_path = Path(run.checkpoint_dir) / "role_model.pkl"
    if not model_path.exists():
        logger.warning(f"Role model checkpoint missing at {model_path}, using heuristic")
        return classify_tracks(video_id, db)

    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        return {"player": 0, "non_player": 0, "skipped": 0, "method": "ml"}

    tracks = db.query(Track).filter(Track.video_id == video_id).all()
    if not tracks:
        return {"player": 0, "non_player": 0, "skipped": 0, "method": "ml"}

    clf = joblib.load(str(model_path))
    track_stats = _compute_all_track_stats(video, tracks, db)

    counts = {"player": 0, "non_player": 0, "skipped": 0, "method": "ml"}

    for track in tracks:
        if track.role_source == "human":
            counts["skipped"] += 1
            continue

        stats = track_stats[track.id]
        features = np.array([_stats_to_feature_vector(stats)])
        pred = clf.predict(features)[0]
        track.role = "player" if pred == 1 else "non_player"
        track.role_source = "model"
        counts[track.role] += 1

    db.commit()

    logger.info(
        f"ML track classification for video {video_id}: "
        f"{counts['player']} players, {counts['non_player']} non-players, "
        f"{counts['skipped']} human-labeled (preserved)"
    )

    return counts

"""Mobile API endpoints — simplified upload → process → result flow."""

import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from spike_platform.config import settings
from spike_platform.database import get_db
from spike_platform.models.db_models import Segment, Track, Video
from spike_platform.routers.analysis import get_spike_analysis
from spike_platform.services.event_clustering import cluster_spike_events
from spike_platform.worker import worker
from spike_platform.services.video_processing import process_video_pipeline

router = APIRouter()

# video_id → worker job_id mapping
_mobile_jobs: dict[str, str] = {}


@router.post("/mobile/analyze")
async def mobile_analyze(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a video and immediately start processing. Returns IDs for polling."""
    video_id = str(uuid.uuid4())
    filepath = settings.UPLOAD_DIR / f"{video_id}{Path(file.filename).suffix}"

    content = await file.read()
    filepath.write_bytes(content)

    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        filepath.unlink(missing_ok=True)
        raise HTTPException(400, "Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    video = Video(
        id=video_id,
        filename=file.filename,
        filepath=str(filepath),
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_seconds=duration,
        status="uploaded",
        video_group="mobile",
    )
    db.add(video)
    db.commit()

    job_id = worker.submit(process_video_pipeline, video_id=video_id)
    _mobile_jobs[video_id] = job_id

    return {"video_id": video_id, "job_id": job_id}


@router.post("/mobile/analyze/{video_id}/reprocess")
def mobile_reprocess(video_id: str, db: Session = Depends(get_db)):
    """Reset and re-submit an existing video for processing."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    if not Path(video.filepath).exists():
        raise HTTPException(410, "Video file no longer on server")

    video.status = "uploaded"
    video.error_message = None
    db.commit()

    job_id = worker.submit(process_video_pipeline, video_id=video_id)
    _mobile_jobs[video_id] = job_id

    return {"video_id": video_id, "job_id": job_id}


@router.get("/mobile/import/single-events")
def mobile_import_single_events(db: Session = Depends(get_db)):
    """Return all processed server videos that have exactly 1 spike event."""
    videos = (
        db.query(Video)
        .filter(Video.status.in_(["processed", "predicted"]))
        .order_by(Video.created_at.desc())
        .all()
    )

    results = []
    for video in videos:
        # Prefer human labels; fall back to predictions
        spike_segs = (
            db.query(Segment)
            .filter(
                Segment.video_id == video.id,
                or_(Segment.human_label == 1, Segment.prediction == 1),
            )
            .all()
        )
        if not spike_segs:
            continue

        events = cluster_spike_events(spike_segs, db)
        if len(events) != 1:
            continue

        try:
            analysis = get_spike_analysis(video.id, db, include_predicted=True)
        except Exception:
            continue

        tracks = analysis.get("tracks", [])
        if not tracks:
            continue

        # Pick best track by max spike confidence
        best_track = None
        best_confidence = -1.0
        for t in tracks:
            track_row = (
                db.query(Track)
                .filter(Track.track_id == t["track_id"], Track.video_id == video.id)
                .first()
            )
            if not track_row:
                continue
            max_conf = (
                db.query(func.max(Segment.confidence))
                .filter(Segment.track_id == track_row.id, Segment.prediction == 1)
                .scalar()
            )
            conf = max_conf or 0.0
            if conf > best_confidence:
                best_confidence = conf
                best_track = t

        if best_track is None:
            best_track = tracks[0]

        kf = best_track.get("key_frames", {})
        thumbnail_frame = kf.get("contact") or kf.get("jump_peak")

        jh = best_track.get("jump_height") or {}
        rh = best_track.get("reach_height") or {}
        arm = best_track.get("arm_swing") or {}
        arm_r = best_track.get("arm_swing_range") or {}

        results.append({
            "video_id": video.id,
            "filename": video.filename,
            "recorded_at": video.created_at.isoformat() if video.created_at else None,
            "thumbnail_frame": thumbnail_frame,
            "metrics": {
                "jump_height_m": jh.get("height_m"),
                "reach_height_m": rh.get("reach_height_m"),
                "swing_speed_ms": arm.get("peak_speed_ms"),
                "swing_range_m": arm_r.get("range_m"),
                "spike_events": 1,
            },
        })

    return results


@router.get("/mobile/analyze/{video_id}/status")
def mobile_status(video_id: str, db: Session = Depends(get_db)):
    """Poll processing progress."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    progress_pct = None
    message = None

    job_id = _mobile_jobs.get(video_id)
    if job_id:
        job_status = worker.get_status(job_id)
        if job_status:
            progress_pct = job_status["progress_pct"]
            message = job_status["message"]

    if video.status == "error":
        message = video.error_message

    return {
        "status": video.status,
        "progress_pct": progress_pct,
        "message": message,
    }


@router.get("/mobile/analyze/{video_id}/result")
def mobile_result(video_id: str, db: Session = Depends(get_db)):
    """Get analysis for the best spike track. Returns 202 if still processing."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    if video.status in ("uploaded", "processing"):
        return JSONResponse(
            status_code=202,
            content={"status": video.status, "message": "Still processing"},
        )

    if video.status == "error":
        raise HTTPException(500, f"Processing failed: {video.error_message}")

    # Check what data exists for this video
    segment_count = (
        db.query(func.count(Segment.id))
        .filter(Segment.video_id == video_id)
        .scalar()
    )
    if segment_count == 0:
        raise HTTPException(404, "No player detected in video")

    spike_count = (
        db.query(func.count(Segment.id))
        .filter(
            Segment.video_id == video_id,
            or_(Segment.human_label == 1, Segment.prediction == 1),
        )
        .scalar()
    )
    if spike_count == 0:
        raise HTTPException(404, "No spike motion detected in video")

    # Get full analysis using predicted labels
    analysis = get_spike_analysis(video_id, db, include_predicted=True)
    tracks = analysis["tracks"]

    if not tracks:
        raise HTTPException(404, "No spikes detected in video")

    # Auto-select best track: highest spike confidence among segments
    best_track = None
    best_confidence = -1.0
    for t in tracks:
        track_row = (
            db.query(Track)
            .filter(Track.track_id == t["track_id"], Track.video_id == video_id)
            .first()
        )
        if not track_row:
            continue
        max_conf = (
            db.query(func.max(Segment.confidence))
            .filter(Segment.track_id == track_row.id, Segment.prediction == 1)
            .scalar()
        )
        conf = max_conf or 0.0
        if conf > best_confidence:
            best_confidence = conf
            best_track = t

    if best_track is None:
        best_track = tracks[0]

    # Read rotation metadata from the video file so the mobile app can
    # display landscape recordings correctly.
    rotation = 0
    try:
        cap = cv2.VideoCapture(video.filepath)
        if cap.isOpened():
            prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", 48)
            rotation = int(cap.get(prop) or 0)
            cap.release()
    except Exception:
        pass

    # Count spike events from predicted segments
    spike_segs = (
        db.query(Segment)
        .filter(Segment.video_id == video_id, Segment.prediction == 1)
        .all()
    )
    spike_event_count = len(cluster_spike_events(spike_segs, db))

    return {
        "video_id": video_id,
        "fps": analysis["fps"],
        "width": analysis["width"],
        "height": analysis["height"],
        "rotation": rotation,
        "track": best_track,
        "spike_event_count": spike_event_count,
    }

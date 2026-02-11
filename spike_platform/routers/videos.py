"""Video upload, listing, and serving endpoints."""

import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import Response, FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from spike_platform.config import settings
from spike_platform.database import get_db
from spike_platform.models.db_models import Video, Track, TrackFrame, Segment
from spike_platform.schemas.video import VideoResponse, VideoStatusResponse
from spike_platform.schemas.segment import SegmentResponse, SegmentListResponse, TrackResponse, TrackRoleUpdate
from spike_platform.worker import worker
from spike_platform.services.video_processing import process_video_pipeline

router = APIRouter()

# video_id → worker job_id mapping (in-memory, ephemeral)
_video_jobs: dict[str, str] = {}


def _video_to_response(video: Video, db: Session) -> VideoResponse:
    """Convert ORM Video to response with computed counts."""
    track_count = db.query(func.count(Track.id)).filter(Track.video_id == video.id).scalar()
    segment_count = db.query(func.count(Segment.id)).filter(Segment.video_id == video.id).scalar()
    labeled_count = (
        db.query(func.count(Segment.id))
        .filter(Segment.video_id == video.id, Segment.human_label.isnot(None))
        .scalar()
    )
    return VideoResponse(
        id=video.id,
        filename=video.filename,
        fps=video.fps,
        width=video.width,
        height=video.height,
        frame_count=video.frame_count,
        duration_seconds=video.duration_seconds,
        status=video.status,
        error_message=video.error_message,
        track_count=track_count,
        segment_count=segment_count,
        labeled_count=labeled_count,
        created_at=video.created_at,
        updated_at=video.updated_at,
    )


@router.post("/videos", response_model=VideoResponse)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a video file and start processing."""
    video_id = str(uuid.uuid4())
    filepath = settings.UPLOAD_DIR / f"{video_id}{Path(file.filename).suffix}"

    # Save file to disk
    content = await file.read()
    filepath.write_bytes(content)

    # Extract video metadata with OpenCV
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
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    # Start background processing
    job_id = worker.submit(process_video_pipeline, video_id=video_id)
    _video_jobs[video_id] = job_id

    return _video_to_response(video, db)


@router.get("/videos", response_model=list[VideoResponse])
def list_videos(db: Session = Depends(get_db)):
    """List all uploaded videos."""
    videos = db.query(Video).order_by(Video.created_at.desc()).all()
    return [_video_to_response(v, db) for v in videos]


@router.get("/videos/{video_id}", response_model=VideoResponse)
def get_video(video_id: str, db: Session = Depends(get_db)):
    """Get details for a single video."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")
    return _video_to_response(video, db)


@router.get("/videos/{video_id}/status", response_model=VideoStatusResponse)
def get_video_status(video_id: str, db: Session = Depends(get_db)):
    """Poll processing status for a video."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    progress_pct = None
    message = video.error_message
    job_id = _video_jobs.get(video_id)
    if job_id:
        job_status = worker.get_status(job_id)
        if job_status:
            progress_pct = job_status["progress_pct"]
            if job_status["message"]:
                message = job_status["message"]

    return VideoStatusResponse(
        status=video.status,
        progress_pct=progress_pct,
        message=message,
    )


@router.delete("/videos/{video_id}")
def delete_video(video_id: str, db: Session = Depends(get_db)):
    """Delete a video and all associated data."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    # Delete video file
    filepath = Path(video.filepath)
    filepath.unlink(missing_ok=True)

    # Delete associated feature files
    segments = db.query(Segment).filter(Segment.video_id == video_id).all()
    for seg in segments:
        if seg.features_path:
            Path(seg.features_path).unlink(missing_ok=True)

    db.delete(video)
    db.commit()
    return {"ok": True}


@router.post("/videos/{video_id}/reprocess")
def reprocess_video(video_id: str, db: Session = Depends(get_db)):
    """Delete existing tracks/segments and re-run the full processing pipeline."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    if worker.is_busy:
        raise HTTPException(409, "A job is already running")

    # Clean up existing data for this video
    from spike_platform.models.db_models import SegmentPhase

    segments = db.query(Segment).filter(Segment.video_id == video_id).all()
    seg_ids = [s.id for s in segments]
    for seg in segments:
        if seg.features_path:
            Path(seg.features_path).unlink(missing_ok=True)

    if seg_ids:
        db.query(SegmentPhase).filter(SegmentPhase.segment_id.in_(seg_ids)).delete(synchronize_session="fetch")
    db.query(Segment).filter(Segment.video_id == video_id).delete()

    track_ids = [t.id for t in db.query(Track).filter(Track.video_id == video_id).all()]
    if track_ids:
        db.query(TrackFrame).filter(TrackFrame.track_id.in_(track_ids)).delete(synchronize_session="fetch")
    db.query(Track).filter(Track.video_id == video_id).delete()

    video.status = "uploaded"
    video.error_message = None
    db.commit()

    # Re-submit to background worker
    job_id = worker.submit(process_video_pipeline, video_id=video_id)
    _video_jobs[video_id] = job_id

    return {"ok": True, "status": "uploaded", "message": "Reprocessing started"}


@router.get("/videos/{video_id}/clip")
def get_clip(video_id: str, db: Session = Depends(get_db)):
    """Serve the video file for HTML5 player."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    filepath = Path(video.filepath)
    if not filepath.exists():
        raise HTTPException(404, "Video file not found on disk")

    return FileResponse(
        str(filepath),
        media_type="video/mp4",
        filename=video.filename,
    )


@router.get("/videos/{video_id}/frame/{frame_num}")
def get_frame(video_id: str, frame_num: int, db: Session = Depends(get_db)):
    """Serve a single video frame as JPEG."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    cap = cv2.VideoCapture(video.filepath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(404, f"Frame {frame_num} not found")

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@router.get("/videos/{video_id}/segments", response_model=SegmentListResponse)
def list_segments(
    video_id: str,
    track_id: int | None = Query(None),
    role: str | None = Query(None),
    label: int | None = Query(None),
    prediction: int | None = Query(None),
    unlabeled_only: bool = Query(False),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List segments for a video with filtering."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    query = db.query(Segment).filter(Segment.video_id == video_id)

    if track_id is not None:
        query = query.filter(Segment.track_id == track_id)
    if role is not None:
        if role == "player":
            # Include both 'player' and 'unknown' tracks (exclude only 'non_player')
            track_ids_with_role = (
                db.query(Track.id).filter(Track.video_id == video_id, Track.role != "non_player").subquery()
            )
        else:
            track_ids_with_role = (
                db.query(Track.id).filter(Track.video_id == video_id, Track.role == role).subquery()
            )
        query = query.filter(Segment.track_id.in_(track_ids_with_role))
    if label is not None:
        query = query.filter(Segment.human_label == label)
    if prediction is not None:
        query = query.filter(Segment.prediction == prediction)
    if unlabeled_only:
        query = query.filter(Segment.human_label.is_(None))

    total = query.count()
    segments = (
        query.order_by(Segment.track_id, Segment.start_frame)
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    fps = video.fps or 30.0
    result = []
    for seg in segments:
        resp = SegmentResponse(
            id=seg.id,
            track_id=seg.track_id,
            video_id=seg.video_id,
            start_frame=seg.start_frame,
            end_frame=seg.end_frame,
            window_size=seg.window_size,
            start_time=seg.start_frame / fps,
            end_time=seg.end_frame / fps,
            prediction=seg.prediction,
            confidence=seg.confidence,
            model_run_id=seg.model_run_id,
            human_label=seg.human_label,
            labeled_at=seg.labeled_at,
        )
        result.append(resp)

    return SegmentListResponse(segments=result, total=total)


@router.get("/videos/{video_id}/tracks/{track_id}/bboxes")
def get_track_bboxes(
    video_id: str,
    track_id: int,
    start_frame: int = Query(...),
    end_frame: int = Query(...),
    db: Session = Depends(get_db),
):
    """Get bounding boxes for a track within a frame range."""
    frames = (
        db.query(TrackFrame)
        .filter(
            TrackFrame.track_id == track_id,
            TrackFrame.frame_number >= start_frame,
            TrackFrame.frame_number <= end_frame,
        )
        .order_by(TrackFrame.frame_number)
        .all()
    )
    return [
        {
            "frame": tf.frame_number,
            "bbox": [tf.bbox_x1, tf.bbox_y1, tf.bbox_x2, tf.bbox_y2],
        }
        for tf in frames
    ]


@router.get("/videos/{video_id}/tracks", response_model=list[TrackResponse])
def list_tracks(video_id: str, role: str | None = Query(None), db: Session = Depends(get_db)):
    """List all tracks for a video with role info and computed stats."""
    from spike_platform.services.track_classifier import compute_track_stats

    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    query = db.query(Track).filter(Track.video_id == video_id)
    if role is not None:
        query = query.filter(Track.role == role)
    tracks = query.order_by(Track.track_id).all()

    video_w = video.width or 1920
    video_h = video.height or 1080

    result = []
    for track in tracks:
        stats = compute_track_stats(track.id, db, video_w, video_h)
        seg_count = db.query(func.count(Segment.id)).filter(Segment.track_id == track.id).scalar()
        result.append(TrackResponse(
            id=track.id,
            track_id=track.track_id,
            start_frame=track.start_frame,
            end_frame=track.end_frame,
            frame_count=track.frame_count,
            avg_confidence=track.avg_confidence,
            role=track.role,
            role_source=track.role_source,
            median_bbox_area=stats["median_bbox_area"],
            median_pose_confidence=stats["median_pose_confidence"],
            segment_count=seg_count,
        ))

    return result


@router.patch("/videos/{video_id}/tracks/{track_id}/role")
def update_track_role(
    video_id: str,
    track_id: int,
    update: TrackRoleUpdate,
    db: Session = Depends(get_db),
):
    """Manually set a track's role (player or non_player)."""
    track = db.query(Track).filter(Track.id == track_id, Track.video_id == video_id).first()
    if not track:
        raise HTTPException(404, "Track not found")

    if update.role not in ("player", "non_player"):
        raise HTTPException(400, "role must be 'player' or 'non_player'")

    track.role = update.role
    track.role_source = "human"
    db.commit()

    return {"id": track.id, "role": track.role, "role_source": track.role_source}


@router.post("/videos/{video_id}/tracks/reclassify")
def reclassify_tracks(video_id: str, db: Session = Depends(get_db)):
    """Re-run heuristic classification on non-human-labeled tracks."""
    from spike_platform.services.track_classifier import classify_tracks

    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    counts = classify_tracks(video_id, db)
    return counts

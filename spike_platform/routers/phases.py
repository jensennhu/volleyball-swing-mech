"""Phase annotation endpoints for spike segments."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from spike_platform.database import get_db
from spike_platform.models.db_models import Segment, SegmentPhase, Track, TrackFrame
from spike_platform.schemas.training import (
    PhaseLabelRequest,
    PhaseLabelResponse,
    FrameResponse,
)

router = APIRouter()

VALID_PHASES = {"approach", "jump", "swing", "land"}


@router.get(
    "/tracks/{track_id}/frames",
    response_model=list[FrameResponse],
)
def get_track_frames(
    track_id: int,
    padding: int = 0,
    start_frame: int | None = None,
    end_frame: int | None = None,
    db: Session = Depends(get_db),
):
    """
    Get all frames for a track. Use padding to extend beyond track boundaries.
    Use start_frame/end_frame for explicit range override.
    """
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(404, "Track not found")

    start = start_frame if start_frame is not None else max(0, track.start_frame - padding)
    end = end_frame if end_frame is not None else track.end_frame + padding

    frames = (
        db.query(TrackFrame)
        .filter(
            TrackFrame.track_id == track_id,
            TrackFrame.frame_number >= start,
            TrackFrame.frame_number <= end,
        )
        .order_by(TrackFrame.frame_number)
        .all()
    )

    return [
        FrameResponse(
            frame_number=f.frame_number,
            bbox=[f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2],
        )
        for f in frames
    ]


@router.get(
    "/segments/{segment_id}/frames",
    response_model=list[FrameResponse],
)
def get_segment_frames(
    segment_id: int,
    padding: int = 60,
    start_frame: int | None = None,
    end_frame: int | None = None,
    db: Session = Depends(get_db),
):
    """
    Get extended frame range for a segment's track.

    By default returns ~padding frames before and after the segment.
    Use start_frame/end_frame to request an explicit range instead.
    """
    seg = db.query(Segment).filter(Segment.id == segment_id).first()
    if not seg:
        raise HTTPException(404, "Segment not found")

    start = start_frame if start_frame is not None else max(0, seg.start_frame - padding)
    end = end_frame if end_frame is not None else seg.end_frame + padding

    frames = (
        db.query(TrackFrame)
        .filter(
            TrackFrame.track_id == seg.track_id,
            TrackFrame.frame_number >= start,
            TrackFrame.frame_number <= end,
        )
        .order_by(TrackFrame.frame_number)
        .all()
    )

    return [
        FrameResponse(
            frame_number=f.frame_number,
            bbox=[f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2],
        )
        for f in frames
    ]


@router.get(
    "/segments/{segment_id}/phases",
    response_model=list[PhaseLabelResponse],
)
def get_segment_phases(segment_id: int, db: Session = Depends(get_db)):
    """Get phase labels for a segment."""
    seg = db.query(Segment).filter(Segment.id == segment_id).first()
    if not seg:
        raise HTTPException(404, "Segment not found")

    phases = (
        db.query(SegmentPhase)
        .filter(SegmentPhase.segment_id == segment_id)
        .order_by(SegmentPhase.start_frame)
        .all()
    )
    return phases


@router.get(
    "/segments/{segment_id}/track-phases",
    response_model=list[PhaseLabelResponse],
)
def get_track_phases(segment_id: int, db: Session = Depends(get_db)):
    """Get merged (unclipped) phase boundaries across all segments on the same track.

    Returns both human annotations and model predictions. Human annotations have
    human_label set; predictions have model_run_id set and human_label=None.
    """
    seg = db.query(Segment).filter(Segment.id == segment_id).first()
    if not seg:
        raise HTTPException(404, "Segment not found")

    # Find all sibling segment IDs on the same track
    sibling_ids = [
        s.id for s in db.query(Segment.id)
        .filter(Segment.track_id == seg.track_id, Segment.video_id == seg.video_id)
        .all()
    ]

    result = []

    # Merge human-annotated phases
    human_merged = (
        db.query(
            SegmentPhase.phase,
            func.min(SegmentPhase.start_frame).label("start_frame"),
            func.max(SegmentPhase.end_frame).label("end_frame"),
        )
        .filter(
            SegmentPhase.segment_id.in_(sibling_ids),
            SegmentPhase.human_label.isnot(None),
        )
        .group_by(SegmentPhase.phase)
        .all()
    )
    for row in human_merged:
        result.append(PhaseLabelResponse(
            id=0,
            segment_id=segment_id,
            phase=row.phase,
            start_frame=row.start_frame,
            end_frame=row.end_frame,
            human_label=row.phase,
            model_run_id=None,
        ))

    # If no human annotations, merge predicted phases
    if not human_merged:
        pred_merged = (
            db.query(
                SegmentPhase.phase,
                func.min(SegmentPhase.start_frame).label("start_frame"),
                func.max(SegmentPhase.end_frame).label("end_frame"),
                func.avg(SegmentPhase.confidence).label("confidence"),
                func.max(SegmentPhase.model_run_id).label("model_run_id"),
            )
            .filter(
                SegmentPhase.segment_id.in_(sibling_ids),
                SegmentPhase.human_label.is_(None),
                SegmentPhase.model_run_id.isnot(None),
            )
            .group_by(SegmentPhase.phase)
            .all()
        )
        for row in pred_merged:
            result.append(PhaseLabelResponse(
                id=0,
                segment_id=segment_id,
                phase=row.phase,
                start_frame=row.start_frame,
                end_frame=row.end_frame,
                human_label=None,
                confidence=float(row.confidence) if row.confidence else None,
                model_run_id=row.model_run_id,
            ))

    return result


@router.put(
    "/segments/{segment_id}/phases",
    response_model=list[PhaseLabelResponse],
)
def set_segment_phases(
    segment_id: int,
    request: PhaseLabelRequest,
    db: Session = Depends(get_db),
):
    """Set/overwrite phase labels for a segment and propagate to overlapping segments on same track."""
    seg = db.query(Segment).filter(Segment.id == segment_id).first()
    if not seg:
        raise HTTPException(404, "Segment not found")

    # Validate phases
    for p in request.phases:
        if p.phase not in VALID_PHASES:
            raise HTTPException(400, f"Invalid phase '{p.phase}'. Must be one of: {VALID_PHASES}")
        if p.start_frame > p.end_frame:
            raise HTTPException(400, f"start_frame must be <= end_frame for phase '{p.phase}'")

    # Find all spike segments on the same track
    sibling_segments = (
        db.query(Segment)
        .filter(
            Segment.track_id == seg.track_id,
            Segment.video_id == seg.video_id,
            Segment.human_label == 1,
        )
        .all()
    )
    if not sibling_segments:
        sibling_segments = [seg]

    # Determine the overall annotated range
    phase_min = min(p.start_frame for p in request.phases)
    phase_max = max(p.end_frame for p in request.phases)

    new_phases = []
    for sib in sibling_segments:
        # Check if this segment overlaps with the annotated phase range
        if sib.end_frame < phase_min or sib.start_frame > phase_max:
            continue

        # Delete existing phase labels for this segment
        db.query(SegmentPhase).filter(SegmentPhase.segment_id == sib.id).delete()

        # Clip and insert phases that overlap with this segment's frame range
        for p in request.phases:
            clipped_start = max(p.start_frame, sib.start_frame)
            clipped_end = min(p.end_frame, sib.end_frame)
            if clipped_start > clipped_end:
                continue
            phase = SegmentPhase(
                segment_id=sib.id,
                phase=p.phase,
                start_frame=clipped_start,
                end_frame=clipped_end,
                human_label=p.phase,
            )
            db.add(phase)
            new_phases.append(phase)

    db.commit()
    # Return only phases for the requested segment
    result = (
        db.query(SegmentPhase)
        .filter(SegmentPhase.segment_id == segment_id)
        .order_by(SegmentPhase.start_frame)
        .all()
    )
    return result


@router.delete("/segments/{segment_id}/phases")
def delete_segment_phases(
    segment_id: int,
    propagate: bool = False,
    db: Session = Depends(get_db),
):
    """Clear all phase labels for a segment. With propagate=true, clear all segments on same track."""
    seg = db.query(Segment).filter(Segment.id == segment_id).first()
    if not seg:
        raise HTTPException(404, "Segment not found")

    if propagate:
        sibling_ids = [
            s.id for s in db.query(Segment.id)
            .filter(Segment.track_id == seg.track_id, Segment.video_id == seg.video_id)
            .all()
        ]
        count = db.query(SegmentPhase).filter(SegmentPhase.segment_id.in_(sibling_ids)).delete(synchronize_session="fetch")
    else:
        count = db.query(SegmentPhase).filter(SegmentPhase.segment_id == segment_id).delete()

    db.commit()
    return {"deleted": count}

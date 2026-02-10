"""Segment labeling endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from spike_platform.database import get_db
from spike_platform.models.db_models import Segment
from spike_platform.schemas.segment import SegmentResponse, SegmentLabelUpdate, BulkLabelUpdate

router = APIRouter()


@router.get("/segments/{segment_id}", response_model=SegmentResponse)
def get_segment(segment_id: int, db: Session = Depends(get_db)):
    """Get a single segment."""
    seg = db.query(Segment).filter(Segment.id == segment_id).first()
    if not seg:
        raise HTTPException(404, "Segment not found")
    return seg


@router.patch("/segments/{segment_id}", response_model=SegmentResponse)
def label_segment(segment_id: int, update: SegmentLabelUpdate, db: Session = Depends(get_db)):
    """Set the human label for a segment."""
    seg = db.query(Segment).filter(Segment.id == segment_id).first()
    if not seg:
        raise HTTPException(404, "Segment not found")

    if update.human_label not in (0, 1):
        raise HTTPException(400, "human_label must be 0 or 1")

    seg.human_label = update.human_label
    seg.labeled_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(seg)
    return seg


@router.patch("/segments/bulk")
def bulk_label_segments(update: BulkLabelUpdate, db: Session = Depends(get_db)):
    """Bulk-label multiple segments."""
    if update.human_label not in (0, 1):
        raise HTTPException(400, "human_label must be 0 or 1")

    now = datetime.now(timezone.utc)
    count = (
        db.query(Segment)
        .filter(Segment.id.in_(update.segment_ids))
        .update(
            {"human_label": update.human_label, "labeled_at": now},
            synchronize_session="fetch",
        )
    )
    db.commit()
    return {"updated": count}

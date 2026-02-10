"""Pydantic schemas for segment endpoints."""

from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class SegmentResponse(BaseModel):
    id: int
    track_id: int
    video_id: str
    start_frame: int
    end_frame: int
    window_size: int
    start_time: Optional[float] = None  # computed from fps
    end_time: Optional[float] = None
    prediction: Optional[int] = None
    confidence: Optional[float] = None
    model_run_id: Optional[int] = None
    human_label: Optional[int] = None
    labeled_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class SegmentLabelUpdate(BaseModel):
    human_label: int  # 0 or 1


class BulkLabelUpdate(BaseModel):
    segment_ids: list[int]
    human_label: int  # 0 or 1


class SegmentListResponse(BaseModel):
    segments: list[SegmentResponse]
    total: int

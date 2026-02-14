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


class TrackResponse(BaseModel):
    id: int
    track_id: int
    start_frame: int
    end_frame: int
    frame_count: int
    avg_confidence: Optional[float] = None
    role: Optional[str] = None
    role_source: Optional[str] = None
    median_bbox_area: Optional[float] = None
    median_pose_confidence: Optional[float] = None
    movement_variance: Optional[float] = None
    bbox_aspect_ratio: Optional[float] = None
    track_duration_ratio: Optional[float] = None
    vertical_range: Optional[float] = None
    segment_count: int = 0

    model_config = {"from_attributes": True}


class TrackRoleUpdate(BaseModel):
    role: str  # player|non_player

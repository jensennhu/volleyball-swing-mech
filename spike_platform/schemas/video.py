"""Pydantic schemas for video endpoints."""

from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class VideoResponse(BaseModel):
    id: str
    filename: str
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    status: str
    error_message: Optional[str] = None
    track_count: int = 0
    segment_count: int = 0
    labeled_count: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class VideoStatusResponse(BaseModel):
    status: str
    progress_pct: Optional[float] = None
    message: Optional[str] = None

"""
Person detection and tracking using YOLOv8 + ByteTrack.

Processes a video file and returns per-frame bounding boxes
grouped by persistent track IDs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from spike_platform.config import settings

_BYTETRACK_CFG = str(Path(__file__).resolve().parent.parent / "cfg" / "bytetrack.yaml")


@dataclass
class FrameDetection:
    """A single person detection in one frame."""
    frame_number: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


@dataclass
class TrackResult:
    """A tracked person across multiple frames."""
    track_id: int
    frames: list[FrameDetection] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.frames[0].frame_number if self.frames else 0

    @property
    def end_frame(self) -> int:
        return self.frames[-1].frame_number if self.frames else 0

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def avg_confidence(self) -> float:
        if not self.frames:
            return 0.0
        return sum(f.confidence for f in self.frames) / len(self.frames)


class PersonDetector:
    """Detect and track people across video frames using YOLOv8 + ByteTrack."""

    def __init__(self, model_name: str = None):
        model_name = model_name or settings.YOLO_MODEL
        self.model = YOLO(model_name)

    def process_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> list[TrackResult]:
        """
        Run detection + tracking on a full video.

        Args:
            video_path: Path to video file.
            progress_callback: Called with (pct, message) during processing.

        Returns:
            List of TrackResult, one per tracked person.
        """
        # Get total frame count for progress reporting
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        tracks_dict: dict[int, TrackResult] = {}
        frame_num = 0

        # Use YOLO's streaming tracker
        results = self.model.track(
            source=video_path,
            stream=True,
            tracker=_BYTETRACK_CFG,
            conf=settings.DETECTION_CONFIDENCE,
            iou=settings.DETECTION_IOU,
            classes=[settings.PERSON_CLASS_ID],
            verbose=False,
        )

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                frame_num += 1
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Skip detections without track IDs (not yet tracked)
                if boxes.id is None:
                    continue

                track_id = int(boxes.id[i].item())
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].item())

                if track_id not in tracks_dict:
                    tracks_dict[track_id] = TrackResult(track_id=track_id)

                tracks_dict[track_id].frames.append(
                    FrameDetection(
                        frame_number=frame_num,
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                    )
                )

            frame_num += 1

            # Report progress every 100 frames
            if progress_callback and frame_num % 100 == 0:
                pct = (frame_num / total_frames) * 100 if total_frames > 0 else 0
                progress_callback(pct, f"Detection: frame {frame_num}/{total_frames}")

        # Sort frames within each track by frame number
        for track in tracks_dict.values():
            track.frames.sort(key=lambda f: f.frame_number)

        return list(tracks_dict.values())

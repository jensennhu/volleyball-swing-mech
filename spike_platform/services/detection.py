"""
Person detection and tracking using YOLOv8 + BoT-SORT.

Processes a video file and returns per-frame bounding boxes
grouped by persistent track IDs, plus ball detections.
"""

import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from spike_platform.config import settings

_TRACKER_CFG = str(Path(__file__).resolve().parent.parent / "cfg" / "botsort.yaml")


def _patch_botsort_kalman():
    """Increase vertical process noise in BoT-SORT's Kalman filter.

    The default constant-velocity model assumes smooth motion equally in all
    directions. During volleyball spikes, vertical acceleration (jump/land) is
    much faster than horizontal. This patches the shared Kalman filter to use
    higher process noise for vertical components so it trusts measurements
    over predictions during fast vertical motion.
    """
    from ultralytics.trackers.bot_sort import BOTrack

    kf = BOTrack.shared_kalman

    def patched_predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3] * 2.0,   # 2x vertical position noise
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3] * 3.0,   # 3x vertical velocity noise
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    kf.predict = types.MethodType(patched_predict, kf)


_patch_botsort_kalman()


# ---------------------------------------------------------------------------
# OKS (Object Keypoint Similarity) distance for pose-assisted association
# ---------------------------------------------------------------------------

# COCO per-keypoint sigmas (17 keypoints)
_COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035,  # nose, eyes, ears
    0.079, 0.079, 0.072, 0.072,          # shoulders, elbows
    0.062, 0.062, 0.107, 0.107,          # wrists, hips
    0.087, 0.087, 0.089, 0.089,          # knees, ankles
])


def _compute_oks(kps1: np.ndarray, kps2: np.ndarray, area: float) -> float:
    """Compute Object Keypoint Similarity between two (17,3) keypoint arrays.

    Args:
        kps1: Track's smoothed keypoints (17, 3) = [x, y, conf] in pixels.
        kps2: Detection's keypoints (17, 3) = [x, y, conf] in pixels.
        area: Bounding box area of the detection (width * height).

    Returns:
        OKS score in [0, 1] where 1 = identical poses.
    """
    visible = (kps1[:, 2] > 0.3) & (kps2[:, 2] > 0.3)
    if visible.sum() == 0:
        return 0.0

    dx = kps1[visible, 0] - kps2[visible, 0]
    dy = kps1[visible, 1] - kps2[visible, 1]
    d_sq = dx ** 2 + dy ** 2
    var = (_COCO_SIGMAS[visible] * 2) ** 2
    e = d_sq / (2 * area * var + 1e-6)
    return float(np.mean(np.exp(-e)))


def _compute_oks_distance(tracks, detections) -> np.ndarray:
    """Compute OKS-based cost matrix between tracks and detections.

    Returns (N_tracks, M_dets) array where 0 = identical pose, 1 = no match.
    """
    cost = np.ones((len(tracks), len(detections)), dtype=np.float32)

    for i, track in enumerate(tracks):
        sp = getattr(track, 'smooth_pose', None)
        if sp is None:
            continue
        for j, det in enumerate(detections):
            cp = getattr(det, 'curr_pose', None)
            if cp is None:
                continue
            tlwh = det.tlwh
            area = tlwh[2] * tlwh[3]
            if area <= 0:
                continue
            cost[i, j] = 1.0 - _compute_oks(sp, cp, area)

    return cost


def _has_pose_data(tracks, detections) -> bool:
    """Check if enough tracks and detections have pose data for OKS."""
    has_track = any(getattr(t, 'smooth_pose', None) is not None for t in tracks)
    has_det = any(getattr(d, 'curr_pose', None) is not None for d in detections)
    return has_track and has_det


# ---------------------------------------------------------------------------
# Monkey-patch BoT-SORT to use pose keypoints in association
# ---------------------------------------------------------------------------

def _patch_botsort_pose_association():
    """Add pose keypoint similarity to BoT-SORT's cost matrix.

    Patches:
      A. on_predict_postprocess_end — stash keypoints on tracker before update()
      B. BOTrack — add pose storage and EMA smoothing
      C. BOTSORT.init_track — attach keypoints to new BOTrack objects
      D. BOTSORT.get_dists — fuse OKS distance with IoU + ReID
    """
    from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
    from ultralytics.trackers.utils import matching
    import ultralytics.trackers.track as _track_module

    # ---- Patch A: stash keypoints before tracker update ----
    _orig_postprocess = _track_module.on_predict_postprocess_end

    def _patched_postprocess(predictor, persist=False):
        is_stream = predictor.dataset.mode == "stream"
        for i, result in enumerate(predictor.results):
            tracker = predictor.trackers[i if is_stream else 0]
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                tracker._frame_keypoints = result.keypoints.data.cpu().numpy()  # (N,17,3)
                tracker._frame_det_bboxes = result.boxes.xyxy.cpu().numpy()     # (N,4)
            else:
                tracker._frame_keypoints = None
                tracker._frame_det_bboxes = None
        _orig_postprocess(predictor, persist)

    _track_module.on_predict_postprocess_end = _patched_postprocess

    # ---- Patch B: BOTrack pose storage ----
    _orig_botrack_init = BOTrack.__init__

    def _patched_botrack_init(self, xywh, score, cls, feat=None, feat_history=50):
        _orig_botrack_init(self, xywh, score, cls, feat, feat_history)
        self.curr_pose = None     # (17, 3) current frame keypoints
        self.smooth_pose = None   # (17, 3) EMA-smoothed keypoints

    BOTrack.__init__ = _patched_botrack_init

    def _update_pose(self, pose_kps):
        """Update pose keypoints with EMA smoothing."""
        if pose_kps is None:
            return
        self.curr_pose = pose_kps
        if self.smooth_pose is None:
            self.smooth_pose = pose_kps.copy()
        else:
            alpha = 0.9
            self.smooth_pose[:, :2] = alpha * self.smooth_pose[:, :2] + (1 - alpha) * pose_kps[:, :2]
            self.smooth_pose[:, 2] = pose_kps[:, 2]  # keep latest confidence

    BOTrack.update_pose = _update_pose

    # Propagate pose on track update (matched detection → existing track)
    _orig_botrack_update = BOTrack.update

    def _patched_botrack_update(self, new_track, frame_id):
        if getattr(new_track, 'curr_pose', None) is not None:
            self.update_pose(new_track.curr_pose)
        _orig_botrack_update(self, new_track, frame_id)

    BOTrack.update = _patched_botrack_update

    # Propagate pose on track reactivation (lost track → matched again)
    _orig_botrack_reactivate = BOTrack.re_activate

    def _patched_botrack_reactivate(self, new_track, frame_id, new_id=False):
        if getattr(new_track, 'curr_pose', None) is not None:
            self.update_pose(new_track.curr_pose)
        _orig_botrack_reactivate(self, new_track, frame_id, new_id)

    BOTrack.re_activate = _patched_botrack_reactivate

    # ---- Patch C: attach keypoints in init_track ----
    _orig_init_track = BOTSORT.init_track

    def _patched_init_track(self, results, img=None):
        tracks = _orig_init_track(self, results, img)
        frame_kps = getattr(self, '_frame_keypoints', None)
        frame_bboxes = getattr(self, '_frame_det_bboxes', None)
        if frame_kps is None or frame_bboxes is None or len(tracks) == 0:
            return tracks

        for track in tracks:
            # Match by exact bbox coordinate comparison against full frame detections
            track_xyxy = track.xyxy  # (4,)
            diffs = np.abs(frame_bboxes - track_xyxy).sum(axis=1)
            best_idx = int(np.argmin(diffs))
            if diffs[best_idx] < 1.0:  # same detection within float precision
                kps = frame_kps[best_idx]
                track.curr_pose = kps
                track.smooth_pose = kps.copy()

        return tracks

    BOTSORT.init_track = _patched_init_track

    # ---- Patch D: OKS distance in get_dists ----
    _orig_get_dists = BOTSORT.get_dists

    def _patched_get_dists(self, tracks, detections):
        dists = _orig_get_dists(self, tracks, detections)

        if _has_pose_data(tracks, detections):
            oks_dists = _compute_oks_distance(tracks, detections)
            # Cap high distances
            oks_dists[oks_dists > settings.POSE_ASSOCIATION_THRESH] = 1.0
            # Mask out spatially distant pairs (same as ReID masking)
            iou_dists = matching.iou_distance(tracks, detections)
            oks_dists[iou_dists > (1 - self.proximity_thresh)] = 1.0
            # Fuse: element-wise minimum (a match on ANY signal is enough)
            dists = np.minimum(dists, oks_dists)

        return dists

    BOTSORT.get_dists = _patched_get_dists


_patch_botsort_pose_association()


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


@dataclass
class BallFrameDetection:
    """A single ball detection in one frame."""
    frame_number: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


@dataclass
class DetectionOutput:
    """Combined detection results: person tracks + raw ball detections."""
    person_tracks: list[TrackResult]
    ball_detections: list[BallFrameDetection]


class PersonDetector:
    """Detect and track people across video frames using YOLOv8-Pose + BoT-SORT.

    Uses two models:
      - yolov8s-pose.pt for person detection + tracking (with keypoints for
        pose-assisted association)
      - yolov8s.pt for ball detection (pose model only detects persons)
    """

    def __init__(self, model_name: str = None):
        # Pose model for person tracking (provides keypoints for association)
        self.pose_model = YOLO(model_name or settings.YOLO_POSE_MODEL)
        # Detection model for ball-only pass
        self.ball_model = YOLO(settings.YOLO_MODEL)

    def process_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> DetectionOutput:
        """
        Run detection + tracking on a full video (two passes).

        Pass 1: Person tracking with pose keypoints (yolov8s-pose + BoT-SORT).
        Pass 2: Ball detection only (yolov8s, no tracking).

        Args:
            video_path: Path to video file.
            progress_callback: Called with (pct, message) during processing.

        Returns:
            DetectionOutput with person tracks and ball detections.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # -- Pass 1: Person tracking with pose-assisted association -----------
        tracks_dict: dict[int, TrackResult] = {}
        frame_num = 0

        person_results = self.pose_model.track(
            source=video_path,
            stream=True,
            tracker=_TRACKER_CFG,
            conf=settings.DETECTION_CONFIDENCE,
            iou=settings.DETECTION_IOU,
            classes=[settings.PERSON_CLASS_ID],
            verbose=False,
        )

        for result in person_results:
            if result.boxes is not None and len(result.boxes) > 0 and result.boxes.id is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
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
            if progress_callback and frame_num % 100 == 0:
                pct = (frame_num / total_frames) * 50 if total_frames > 0 else 0
                progress_callback(pct, f"Person tracking: frame {frame_num}/{total_frames}")

        # -- Pass 2: Ball detection only (no tracking needed) -----------------
        ball_detections: list[BallFrameDetection] = []
        frame_num = 0

        ball_results = self.ball_model.predict(
            source=video_path,
            stream=True,
            conf=settings.BALL_DETECTION_CONFIDENCE,
            classes=[settings.BALL_CLASS_ID],
            verbose=False,
        )

        for result in ball_results:
            if result.boxes is not None:
                for i in range(len(result.boxes)):
                    bbox = result.boxes.xyxy[i].cpu().numpy()
                    conf = float(result.boxes.conf[i].item())
                    ball_detections.append(
                        BallFrameDetection(
                            frame_number=frame_num,
                            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                            confidence=conf,
                        )
                    )

            frame_num += 1
            if progress_callback and frame_num % 100 == 0:
                pct = 50 + (frame_num / total_frames) * 50 if total_frames > 0 else 0
                progress_callback(pct, f"Ball detection: frame {frame_num}/{total_frames}")

        # Sort results
        for track in tracks_dict.values():
            track.frames.sort(key=lambda f: f.frame_number)
        ball_detections.sort(key=lambda b: b.frame_number)

        return DetectionOutput(
            person_tracks=list(tracks_dict.values()),
            ball_detections=ball_detections,
        )

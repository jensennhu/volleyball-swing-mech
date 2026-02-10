"""
Pose extraction service.

Runs MediaPipe Pose on cropped person regions and extracts
33-dim normalized features (pelvis-centered, torso-scaled).

Ported from: src/preprocessing/normalized_pose_features.py
Feature vector: 8 joint angles + 24 normalized positions (12 keypoints x,y) + 1 confidence = 33
"""

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


class PoseService:
    """Extract normalized pose features from person crops."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # process independent crops, not video stream
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_features_from_crop(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> tuple[Optional[np.ndarray], float]:
        """
        Crop person from frame, run MediaPipe Pose, extract 33-dim features.

        Args:
            frame: Full video frame (BGR, HWC).
            bbox: (x1, y1, x2, y2) bounding box in pixel coords.

        Returns:
            (features, confidence) where features is shape (33,) or None if pose not detected.
        """
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        result = self.pose.process(crop_rgb)
        if not result.pose_landmarks:
            return None, 0.0

        features = self._extract_normalized_features(result.pose_landmarks)
        confidence = features[-1] if features is not None else 0.0
        return features, confidence

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()

    # ------------------------------------------------------------------ #
    # Ported from NormalizedPoseExtractor (src/preprocessing/normalized_pose_features.py)
    # ------------------------------------------------------------------ #

    def _calculate_angle(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
    ) -> float:
        """Calculate angle at point b formed by segments ba and bc."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def _get_landmark(self, landmarks, landmark_id: int) -> tuple[float, float]:
        """Get landmark as (x, y) in MediaPipe normalized coords (0-1)."""
        lm = landmarks.landmark[landmark_id]
        return (lm.x, lm.y)

    def _normalize_pose_to_pelvis(self, landmarks) -> dict[str, tuple[float, float]]:
        """
        Center all landmarks at pelvis and scale by torso length.
        Makes features invariant to position in frame and person size.
        """
        PL = self.mp_pose.PoseLandmark

        r_hip = self._get_landmark(landmarks, PL.RIGHT_HIP)
        l_hip = self._get_landmark(landmarks, PL.LEFT_HIP)

        # Pelvis center (root)
        pelvis_x = (r_hip[0] + l_hip[0]) / 2
        pelvis_y = (r_hip[1] + l_hip[1]) / 2

        # Shoulder center for torso length
        r_shoulder = self._get_landmark(landmarks, PL.RIGHT_SHOULDER)
        l_shoulder = self._get_landmark(landmarks, PL.LEFT_SHOULDER)
        shoulder_x = (r_shoulder[0] + l_shoulder[0]) / 2
        shoulder_y = (r_shoulder[1] + l_shoulder[1]) / 2

        # Torso length for scale normalization
        torso_length = np.sqrt((shoulder_x - pelvis_x) ** 2 + (shoulder_y - pelvis_y) ** 2)
        if torso_length < 0.01:
            torso_length = 1.0

        # 12 keypoints
        keypoints = {
            "right_shoulder": r_shoulder,
            "right_elbow": self._get_landmark(landmarks, PL.RIGHT_ELBOW),
            "right_wrist": self._get_landmark(landmarks, PL.RIGHT_WRIST),
            "right_hip": r_hip,
            "right_knee": self._get_landmark(landmarks, PL.RIGHT_KNEE),
            "right_ankle": self._get_landmark(landmarks, PL.RIGHT_ANKLE),
            "left_shoulder": l_shoulder,
            "left_elbow": self._get_landmark(landmarks, PL.LEFT_ELBOW),
            "left_wrist": self._get_landmark(landmarks, PL.LEFT_WRIST),
            "left_hip": l_hip,
            "left_knee": self._get_landmark(landmarks, PL.LEFT_KNEE),
            "left_ankle": self._get_landmark(landmarks, PL.LEFT_ANKLE),
        }

        # Center at pelvis, scale by torso
        normalized = {}
        for name, (x, y) in keypoints.items():
            normalized[name] = (
                (x - pelvis_x) / torso_length,
                (y - pelvis_y) / torso_length,
            )

        return normalized

    def _extract_normalized_features(self, landmarks) -> Optional[np.ndarray]:
        """
        Extract 33-dim normalized pose feature vector.

        Layout:
            [0:8]   - 8 joint angles (right/left shoulder, elbow, hip, knee)
            [8:32]  - 24 normalized positions (12 keypoints × x,y)
            [32]    - mean visibility confidence
        """
        try:
            norm = self._normalize_pose_to_pelvis(landmarks)

            # 8 joint angles
            angles = [
                self._calculate_angle(norm["right_hip"], norm["right_shoulder"], norm["right_elbow"]),
                self._calculate_angle(norm["right_shoulder"], norm["right_elbow"], norm["right_wrist"]),
                self._calculate_angle(norm["right_shoulder"], norm["right_hip"], norm["right_knee"]),
                self._calculate_angle(norm["right_hip"], norm["right_knee"], norm["right_ankle"]),
                self._calculate_angle(norm["left_hip"], norm["left_shoulder"], norm["left_elbow"]),
                self._calculate_angle(norm["left_shoulder"], norm["left_elbow"], norm["left_wrist"]),
                self._calculate_angle(norm["left_shoulder"], norm["left_hip"], norm["left_knee"]),
                self._calculate_angle(norm["left_hip"], norm["left_knee"], norm["left_ankle"]),
            ]

            # 24 normalized positions (x,y for each of 12 keypoints)
            position_keys = [
                "right_shoulder", "right_elbow", "right_wrist",
                "right_hip", "right_knee", "right_ankle",
                "left_shoulder", "left_elbow", "left_wrist",
                "left_hip", "left_knee", "left_ankle",
            ]
            positions = []
            for key in position_keys:
                x, y = norm[key]
                positions.extend([x, y])

            # 1 confidence (mean visibility of 6 key landmarks)
            PL = self.mp_pose.PoseLandmark
            confidence = np.mean([
                landmarks.landmark[PL.RIGHT_WRIST].visibility,
                landmarks.landmark[PL.LEFT_WRIST].visibility,
                landmarks.landmark[PL.RIGHT_SHOULDER].visibility,
                landmarks.landmark[PL.LEFT_SHOULDER].visibility,
                landmarks.landmark[PL.RIGHT_HIP].visibility,
                landmarks.landmark[PL.LEFT_HIP].visibility,
            ])

            features = angles + positions + [confidence]
            return np.array(features, dtype=np.float32)

        except Exception:
            return None

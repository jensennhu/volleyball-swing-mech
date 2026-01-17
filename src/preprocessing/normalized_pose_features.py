"""
Normalized Pose Feature Extraction
===================================

Extracts pose features with proper normalization:
1. Root normalization: Centers coordinates at pelvis (hip midpoint)
2. Scale normalization: Normalizes by torso length
3. Frame-size independent: Works across different video resolutions

This makes features invariant to:
- Person's position in frame
- Person's distance from camera
- Person's height/size
- Video resolution
"""

import numpy as np
from typing import Tuple, Dict, Optional
import mediapipe as mp


class NormalizedPoseExtractor:
    """Extract normalized pose features for consistent LSTM input."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
    
    def calculate_angle(self, a: Tuple[float, float], 
                       b: Tuple[float, float], 
                       c: Tuple[float, float]) -> float:
        """Calculate angle at point b."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def get_landmark_normalized(self, landmarks, landmark_id: int) -> Tuple[float, float]:
        """
        Get landmark in normalized coordinates (0-1 range).
        
        MediaPipe already provides normalized coordinates, so we just extract them.
        """
        lm = landmarks.landmark[landmark_id]
        return (lm.x, lm.y)
    
    def normalize_pose_to_pelvis(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """
        Normalize all landmarks relative to pelvis (hip midpoint).
        
        This makes the pose invariant to the person's position in the frame.
        
        Steps:
        1. Find pelvis center (midpoint of left and right hips)
        2. Subtract pelvis position from all landmarks
        3. Scale by torso length (shoulder to hip distance)
        
        Returns:
            Dict of normalized landmark positions relative to pelvis
        """
        # Get hip positions (pelvis)
        r_hip = self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
        l_hip = self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        
        # Calculate pelvis center (root point)
        pelvis_x = (r_hip[0] + l_hip[0]) / 2
        pelvis_y = (r_hip[1] + l_hip[1]) / 2
        
        # Get shoulder positions for scale calculation
        r_shoulder = self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        l_shoulder = self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        
        # Calculate shoulder center
        shoulder_x = (r_shoulder[0] + l_shoulder[0]) / 2
        shoulder_y = (r_shoulder[1] + l_shoulder[1]) / 2
        
        # Calculate torso length (for scale normalization)
        torso_length = np.sqrt((shoulder_x - pelvis_x)**2 + (shoulder_y - pelvis_y)**2)
        
        # Avoid division by zero
        if torso_length < 0.01:
            torso_length = 1.0
        
        # Extract all key landmarks
        landmark_positions = {
            'right_shoulder': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            'right_elbow': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            'right_wrist': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            'right_hip': r_hip,
            'right_knee': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            'right_ankle': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            'left_shoulder': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER),
            'left_elbow': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            'left_wrist': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST),
            'left_hip': l_hip,
            'left_knee': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE),
            'left_ankle': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE),
        }
        
        # Normalize: center at pelvis and scale by torso length
        normalized_landmarks = {}
        for name, (x, y) in landmark_positions.items():
            # Subtract pelvis position (centering)
            centered_x = x - pelvis_x
            centered_y = y - pelvis_y
            
            # Scale by torso length (normalization)
            normalized_x = centered_x / torso_length
            normalized_y = centered_y / torso_length
            
            normalized_landmarks[name] = (normalized_x, normalized_y)
        
        return normalized_landmarks
    
    def extract_normalized_features(self, landmarks) -> Optional[np.ndarray]:
        """
        Extract normalized pose features for LSTM.
        
        Returns 33 features:
        - 8 joint angles (rotation-invariant)
        - 24 normalized positions (12 keypoints × x,y, pelvis-centered and scaled)
        - 1 confidence score
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            np.ndarray of shape (33,) or None if extraction fails
        """
        try:
            # Get normalized landmarks (pelvis-centered, torso-scaled)
            norm_landmarks = self.normalize_pose_to_pelvis(landmarks)
            
            # Calculate joint angles (angles are already invariant to position/scale)
            angles = {
                'right_shoulder': self.calculate_angle(
                    norm_landmarks['right_hip'],
                    norm_landmarks['right_shoulder'],
                    norm_landmarks['right_elbow']
                ),
                'right_elbow': self.calculate_angle(
                    norm_landmarks['right_shoulder'],
                    norm_landmarks['right_elbow'],
                    norm_landmarks['right_wrist']
                ),
                'right_hip': self.calculate_angle(
                    norm_landmarks['right_shoulder'],
                    norm_landmarks['right_hip'],
                    norm_landmarks['right_knee']
                ),
                'right_knee': self.calculate_angle(
                    norm_landmarks['right_hip'],
                    norm_landmarks['right_knee'],
                    norm_landmarks['right_ankle']
                ),
                'left_shoulder': self.calculate_angle(
                    norm_landmarks['left_hip'],
                    norm_landmarks['left_shoulder'],
                    norm_landmarks['left_elbow']
                ),
                'left_elbow': self.calculate_angle(
                    norm_landmarks['left_shoulder'],
                    norm_landmarks['left_elbow'],
                    norm_landmarks['left_wrist']
                ),
                'left_hip': self.calculate_angle(
                    norm_landmarks['left_shoulder'],
                    norm_landmarks['left_hip'],
                    norm_landmarks['left_knee']
                ),
                'left_knee': self.calculate_angle(
                    norm_landmarks['left_hip'],
                    norm_landmarks['left_knee'],
                    norm_landmarks['left_ankle']
                )
            }
            
            # Build feature vector
            features = []
            
            # 1. Joint angles (8 features)
            for key in ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                       'left_shoulder', 'left_elbow', 'left_hip', 'left_knee']:
                features.append(angles[key])
            
            # 2. Normalized landmark positions (24 features)
            # These are now relative to pelvis and scaled by torso length
            for key in ['right_shoulder', 'right_elbow', 'right_wrist',
                       'right_hip', 'right_knee', 'right_ankle',
                       'left_shoulder', 'left_elbow', 'left_wrist',
                       'left_hip', 'left_knee', 'left_ankle']:
                x, y = norm_landmarks[key]
                features.extend([x, y])
            
            # 3. Confidence (1 feature)
            confidence = np.mean([
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].visibility
            ])
            features.append(confidence)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting normalized features: {e}")
            return None
    
    def extract_features_with_metadata(self, landmarks) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Extract features and return additional metadata for debugging/visualization.
        
        Returns:
            - features: np.ndarray of shape (33,)
            - metadata: Dict with angles, normalized_landmarks, etc.
        """
        features = self.extract_normalized_features(landmarks)
        
        if features is None:
            return None, {}
        
        # Also compute metadata
        norm_landmarks = self.normalize_pose_to_pelvis(landmarks)
        
        angles = {}
        for key in ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                   'left_shoulder', 'left_elbow', 'left_hip', 'left_knee']:
            # Extract angle from features (first 8 values)
            angle_idx = ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                        'left_shoulder', 'left_elbow', 'left_hip', 'left_knee'].index(key)
            angles[key] = features[angle_idx]
        
        metadata = {
            'angles': angles,
            'normalized_landmarks': norm_landmarks,
            'confidence': features[-1]
        }
        
        return features, metadata


# Example usage function
def compare_old_vs_new_features():
    """
    Demonstration of the difference between old (pixel) and new (normalized) features.
    """
    print("=" * 70)
    print("OLD APPROACH (Pixel Coordinates)")
    print("=" * 70)
    print("Person at position (100, 200) in 1920x1080 video:")
    print("  right_shoulder: (450, 320)")
    print("  right_elbow: (480, 420)")
    print("  right_wrist: (510, 480)")
    print()
    print("Same person at position (50, 100) in 640x480 video:")
    print("  right_shoulder: (150, 107)")  # Different values!
    print("  right_elbow: (160, 140)")
    print("  right_wrist: (170, 160)")
    print()
    print("❌ Problem: Same pose, different features → model confused!")
    print()
    print("=" * 70)
    print("NEW APPROACH (Normalized Coordinates)")
    print("=" * 70)
    print("Person at ANY position in ANY resolution:")
    print("  right_shoulder: (0.0, 0.5)    ← Relative to pelvis")
    print("  right_elbow: (0.15, 0.3)     ← Scaled by torso")
    print("  right_wrist: (0.25, 0.1)")
    print()
    print("✅ Solution: Same pose, same features → model works!")
    print("=" * 70)


if __name__ == "__main__":
    compare_old_vs_new_features()

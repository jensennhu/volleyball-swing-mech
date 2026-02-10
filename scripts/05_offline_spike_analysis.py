#!/usr/bin/env python3
"""
Comprehensive Offline Spike Analysis
====================================

Analyzes a complete volleyball video to:
1. Detect all spike sequences
2. Break down each spike into phases (approach, jump, swing, land)
3. Calculate detailed biomechanical metrics:
   - Arm swing speed (m/s and mph)
   - Jump height (cm and inches)
   - Hip-shoulder separation (cm and inches)
   - Joint angles at key moments
   - Phase timing
   - Contact point estimation

Generates comprehensive reports with visualizations in both metric and imperial units.

Usage: python 05_offline_spike_analysis.py
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Pose detection
import mediapipe as mp

# LSTM model
try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("❌ Error: TensorFlow not installed")
    exit(1)

# Normalization
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# UNIT CONVERSION UTILITIES
# ============================================================================
class UnitConverter:
    """Convert between metric and imperial units."""
    
    @staticmethod
    def cm_to_inches(cm: float) -> float:
        """Convert centimeters to inches."""
        return cm / 2.54
    
    @staticmethod
    def cm_to_meters(cm: float) -> float:
        """Convert centimeters to meters."""
        return cm / 100
    
    @staticmethod
    def cm_to_feet(cm: float) -> float:
        """Convert centimeters to feet."""
        return cm / 30.48
    
    @staticmethod
    def meters_to_feet(m: float) -> float:
        """Convert meters to feet."""
        return m * 3.28084
    
    @staticmethod
    def ms_to_mph(ms: float) -> float:
        """Convert meters/second to miles/hour."""
        return ms * 2.23694
    
    @staticmethod
    def ms_to_kmh(ms: float) -> float:
        """Convert meters/second to kilometers/hour."""
        return ms * 3.6
    
    @staticmethod
    def format_height(cm: float, include_imperial: bool = True) -> str:
        """Format height with both metric and imperial."""
        if include_imperial:
            inches = UnitConverter.cm_to_inches(cm)
            return f"{cm:.1f} cm ({inches:.1f} in)"
        return f"{cm:.1f} cm"
    
    @staticmethod
    def format_speed(ms: float, include_imperial: bool = True) -> str:
        """Format speed with both metric and imperial."""
        if include_imperial:
            mph = UnitConverter.ms_to_mph(ms)
            kmh = UnitConverter.ms_to_kmh(ms)
            return f"{ms:.2f} m/s ({mph:.1f} mph / {kmh:.1f} km/h)"
        return f"{ms:.2f} m/s"


# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
VIDEO_PATH = "data/raw/videos/recorded_videos/hitting-session.mp4"
MODEL_PATH = "models/spike_phase_classifier_normalized_bidirectional.keras"
METADATA_PATH = "models/spike_phase_classifier_normalized_bidirectional_metadata.json"

# Analysis settings
BUFFER_SIZE = 10                        # Frames in sequence buffer
PREDICTION_THRESHOLD = 0.6              # Minimum confidence to accept prediction
MIN_SPIKE_DURATION = 30                 # Minimum frames for valid spike
MAX_SPIKE_GAP = 20                      # Max frames between phases

# Biomechanics calibration
PLAYER_HEIGHT_CM = 180                  # Player height for calibration
SHOULDER_HIP_RATIO = 0.27               # Typical shoulder-hip ratio

# Output settings
OUTPUT_DIR = "outputs/reports"
GENERATE_REPORT = True
GENERATE_VISUALIZATIONS = True
SAVE_ANNOTATED_VIDEO = False           # Warning: Creates large files
SHOW_IMPERIAL_UNITS = True              # Show both metric and imperial units
VERBOSE = True
# ============================================================================


class NormalizedPoseExtractor:
    """Extract normalized pose features."""
    
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
        """Get landmark in normalized coordinates."""
        lm = landmarks.landmark[landmark_id]
        return (lm.x, lm.y)
    
    def normalize_pose_to_pelvis(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """Normalize all landmarks relative to pelvis."""
        r_hip = self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
        l_hip = self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        
        pelvis_x = (r_hip[0] + l_hip[0]) / 2
        pelvis_y = (r_hip[1] + l_hip[1]) / 2
        
        r_shoulder = self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        l_shoulder = self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        
        shoulder_x = (r_shoulder[0] + l_shoulder[0]) / 2
        shoulder_y = (r_shoulder[1] + l_shoulder[1]) / 2
        
        torso_length = np.sqrt((shoulder_x - pelvis_x)**2 + (shoulder_y - pelvis_y)**2)
        
        if torso_length < 0.01:
            torso_length = 1.0
        
        landmark_positions = {
            'right_shoulder': r_shoulder,
            'right_elbow': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            'right_wrist': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            'right_hip': r_hip,
            'right_knee': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            'right_ankle': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            'left_shoulder': l_shoulder,
            'left_elbow': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            'left_wrist': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST),
            'left_hip': l_hip,
            'left_knee': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE),
            'left_ankle': self.get_landmark_normalized(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE),
        }
        
        normalized_landmarks = {}
        for name, (x, y) in landmark_positions.items():
            centered_x = x - pelvis_x
            centered_y = y - pelvis_y
            
            normalized_x = centered_x / torso_length
            normalized_y = centered_y / torso_length
            
            normalized_landmarks[name] = (normalized_x, normalized_y)
        
        return normalized_landmarks, torso_length
    
    def extract_features_with_metadata(self, landmarks, frame_width: int = None, frame_height: int = None) -> Tuple[Optional[np.ndarray], Dict]:
        """Extract features and metadata including pixel coordinates."""
        try:
            norm_landmarks, torso_length = self.normalize_pose_to_pelvis(landmarks)
            
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
            
            features = []
            
            for key in ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                       'left_shoulder', 'left_elbow', 'left_hip', 'left_knee']:
                features.append(angles[key])
            
            for key in ['right_shoulder', 'right_elbow', 'right_wrist',
                       'right_hip', 'right_knee', 'right_ankle',
                       'left_shoulder', 'left_elbow', 'left_wrist',
                       'left_hip', 'left_knee', 'left_ankle']:
                x, y = norm_landmarks[key]
                features.extend([x, y])
            
            confidence = np.mean([
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].visibility
            ])
            features.append(confidence)
            
            # Also store ORIGINAL pixel coordinates for biomechanical calculations
            pixel_landmarks = {}
            if frame_width and frame_height:
                landmark_ids = {
                    'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
                    'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
                    'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
                    'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                    'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
                    'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
                    'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
                    'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
                    'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
                }
                
                for name, landmark_id in landmark_ids.items():
                    lm = landmarks.landmark[landmark_id]
                    # Convert MediaPipe's normalized coordinates (0-1) to pixels
                    pixel_x = lm.x * frame_width
                    pixel_y = lm.y * frame_height
                    pixel_landmarks[name] = (pixel_x, pixel_y)
            
            metadata = {
                'angles': angles,
                'normalized_landmarks': norm_landmarks,
                'pixel_landmarks': pixel_landmarks,  # NEW: Actual pixel coordinates
                'confidence': confidence,
                'torso_length': torso_length
            }
            
            return np.array(features, dtype=np.float32), metadata
            
        except Exception as e:
            return None, {}


class BiomechanicsCalculator:
    """Calculate biomechanical metrics from pose sequences."""
    
    def __init__(self, player_height_cm: float, fps: float):
        self.player_height_cm = player_height_cm
        self.fps = fps
        self.shoulder_hip_ratio = 0.27
        
    def calculate_pixels_per_cm(self, torso_length_pixels: float) -> float:
        """Calculate pixel-to-cm ratio using torso length."""
        expected_torso_cm = self.player_height_cm * self.shoulder_hip_ratio
        return torso_length_pixels / expected_torso_cm
    
    def calculate_jump_height(self, hip_positions: List[float], 
                             pixels_per_cm: float) -> Dict:
        """Calculate jump height metrics."""
        hip_positions = np.array(hip_positions)
        
        # Find baseline (standing position - lowest hip during approach)
        baseline_hip = np.max(hip_positions[:len(hip_positions)//3])  # First third
        
        # Find peak (highest point)
        peak_hip = np.min(hip_positions)
        
        # Calculate height difference in pixels
        height_diff_pixels = baseline_hip - peak_hip
        
        # Convert to cm
        jump_height_cm = height_diff_pixels * pixels_per_cm
        
        peak_idx = np.argmin(hip_positions)
        
        return {
            'jump_height_cm': float(jump_height_cm),
            'jump_height_meters': float(UnitConverter.cm_to_meters(jump_height_cm)),
            'jump_height_inches': float(UnitConverter.cm_to_inches(jump_height_cm)),
            'jump_height_feet': float(UnitConverter.cm_to_feet(jump_height_cm)),
            'peak_frame': int(peak_idx),
            'baseline_hip_y': float(baseline_hip),
            'peak_hip_y': float(peak_hip),
            'hip_positions': [float(h) for h in hip_positions]  # Store for plotting
        }
    
    def calculate_arm_speed(self, wrist_positions: List[Tuple[float, float]], 
                           pixels_per_cm: float) -> Dict:
        """Calculate arm swing speed."""
        speeds = []
        
        for i in range(1, len(wrist_positions)):
            prev_x, prev_y = wrist_positions[i-1]
            curr_x, curr_y = wrist_positions[i]
            
            dist_pixels = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            dist_cm = dist_pixels * pixels_per_cm
            dist_m = dist_cm / 100
            
            # Speed in m/s
            speed = dist_m * self.fps
            speeds.append(float(speed))
        
        max_speed = max(speeds) if speeds else 0
        max_speed_idx = speeds.index(max_speed) + 1 if speeds else 0
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        
        return {
            'max_speed_ms': float(max_speed),
            'max_speed_mph': float(UnitConverter.ms_to_mph(max_speed)),
            'max_speed_kmh': float(UnitConverter.ms_to_kmh(max_speed)),
            'avg_speed_ms': float(avg_speed),
            'avg_speed_mph': float(UnitConverter.ms_to_mph(avg_speed)),
            'avg_speed_kmh': float(UnitConverter.ms_to_kmh(avg_speed)),
            'max_speed_frame': int(max_speed_idx),
            'speeds': speeds
        }
    
    def calculate_hip_shoulder_separation(self, hip_positions: List[float],
                                         shoulder_positions: List[float],
                                         pixels_per_cm: float) -> Dict:
        """Calculate hip-shoulder separation (torso extension)."""
        separations = []
        
        for hip_y, shoulder_y in zip(hip_positions, shoulder_positions):
            sep_pixels = abs(shoulder_y - hip_y)
            sep_cm = sep_pixels * pixels_per_cm
            separations.append(float(sep_cm))
        
        max_separation = max(separations) if separations else 0.0
        max_sep_idx = separations.index(max_separation) if separations else 0
        avg_separation = float(np.mean(separations)) if separations else 0.0
        
        return {
            'max_separation_cm': float(max_separation),
            'max_separation_inches': float(UnitConverter.cm_to_inches(max_separation)),
            'avg_separation_cm': float(avg_separation),
            'avg_separation_inches': float(UnitConverter.cm_to_inches(avg_separation)),
            'max_separation_frame': int(max_sep_idx),
            'separations': separations
        }


class OfflineSpikeAnalyzer:
    """Comprehensive offline spike analysis."""
    
    def __init__(self, model_path: str, metadata_path: str, 
                 buffer_size: int = 10,
                 player_height_cm: float = 180):
        self.buffer_size = buffer_size
        self.player_height_cm = player_height_cm
        
        # Load model
        self.model = keras.models.load_model(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.label_names = self.metadata['label_names']
        
        # Initialize components
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        self.feature_extractor = NormalizedPoseExtractor()
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        print(f"✅ Analyzer initialized")
        print(f"   Model classes: {self.label_names}")
        print(f"   Buffer size: {buffer_size}")
        print(f"   Player height: {player_height_cm} cm")
    
    def analyze_video(self, video_path: str, 
                     threshold: float = 0.6,
                     verbose: bool = True) -> Dict:
        """Analyze complete video to detect and characterize spikes."""
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if verbose:
            print(f"\n📹 Video info:")
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {fps:.2f}")
            print(f"   Total frames: {total_frames}")
            print(f"   Duration: {total_frames/fps:.1f}s")
        
        # Initialize biomechanics calculator
        bio_calc = BiomechanicsCalculator(self.player_height_cm, fps)
        
        # Process video
        frame_buffer = deque(maxlen=self.buffer_size)
        feature_buffer = deque(maxlen=self.buffer_size)
        
        all_predictions = []
        all_frames_data = []
        
        frame_idx = 0
        
        with self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if verbose and frame_idx % 100 == 0:
                    print(f"   Processing frame {frame_idx}/{total_frames}...")
                
                # Convert to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                frame_data = {
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'pose_detected': False
                }
                
                if results.pose_landmarks:
                    features, metadata = self.feature_extractor.extract_features_with_metadata(
                        results.pose_landmarks,
                        frame_width=width,
                        frame_height=height
                    )
                    
                    if features is not None:
                        feature_buffer.append(features)
                        frame_buffer.append(frame.copy())
                        
                        frame_data['pose_detected'] = True
                        frame_data['metadata'] = metadata
                        
                        # Make prediction if buffer full
                        if len(feature_buffer) == self.buffer_size:
                            prediction, confidence = self.predict_from_buffer(feature_buffer)
                            
                            if confidence >= threshold:
                                all_predictions.append({
                                    'frame_idx': frame_idx,
                                    'timestamp': frame_idx / fps,
                                    'phase': prediction,
                                    'confidence': confidence
                                })
                
                all_frames_data.append(frame_data)
                frame_idx += 1
        
        cap.release()
        
        if verbose:
            print(f"✅ Video processing complete")
            print(f"   Total predictions: {len(all_predictions)}")
        
        # Group predictions into spikes
        spikes = self.group_predictions_into_spikes(
            all_predictions,
            all_frames_data,
            min_duration=30,
            max_gap=20
        )
        
        if verbose:
            print(f"   Detected spikes: {len(spikes)}")
        
        # Calculate biomechanics for each spike
        for spike in spikes:
            self.calculate_spike_biomechanics(spike, bio_calc, all_frames_data, fps)
        
        return {
            'video_info': {
                'path': video_path,
                'fps': fps,
                'total_frames': total_frames,
                'resolution': f"{width}x{height}"
            },
            'spikes': spikes,
            'total_predictions': len(all_predictions),
            'frames_data': all_frames_data  # NEW: Include for frame extraction
        }
    
    def predict_from_buffer(self, feature_buffer: deque) -> Tuple[str, float]:
        """Make prediction from current buffer."""
        X = np.array(list(feature_buffer), dtype=np.float32)
        X = X.reshape(1, self.buffer_size, -1)
        
        n_frames, n_features = X.shape[1], X.shape[2]
        X_reshaped = X.reshape(-1, n_features)
        
        if not self.scaler_fitted:
            self.scaler.fit(X_reshaped)
            self.scaler_fitted = True
        
        X_normalized = self.scaler.transform(X_reshaped)
        X = X_normalized.reshape(1, n_frames, n_features)
        
        predictions_proba = self.model.predict(X, verbose=0)
        pred_idx = np.argmax(predictions_proba[0])
        confidence = float(predictions_proba[0][pred_idx])
        
        predicted_label = self.label_names[pred_idx]
        
        return predicted_label, confidence
    
    def group_predictions_into_spikes(self, predictions: List[Dict],
                                     frames_data: List[Dict],
                                     min_duration: int = 30,
                                     max_gap: int = 20) -> List[Dict]:
        """Group sequential predictions into spike sequences."""
        if not predictions:
            return []
        
        spikes = []
        current_spike = []
        last_frame = -1
        
        for pred in predictions:
            frame_idx = pred['frame_idx']
            
            if last_frame == -1 or (frame_idx - last_frame) <= max_gap:
                current_spike.append(pred)
            else:
                # Gap too large, save current spike and start new
                if len(current_spike) >= min_duration:
                    spikes.append(self.create_spike_dict(current_spike, frames_data))
                current_spike = [pred]
            
            last_frame = frame_idx
        
        # Add last spike
        if len(current_spike) >= min_duration:
            spikes.append(self.create_spike_dict(current_spike, frames_data))
        
        return spikes
    
    def create_spike_dict(self, predictions: List[Dict], 
                         frames_data: List[Dict]) -> Dict:
        """Create spike dictionary from predictions."""
        start_frame = predictions[0]['frame_idx']
        end_frame = predictions[-1]['frame_idx']
        
        # Group by phase
        phases = {}
        for pred in predictions:
            phase = pred['phase']
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(pred)
        
        # Determine phase sequence
        phase_sequence = []
        current_phase = None
        
        for pred in predictions:
            if pred['phase'] != current_phase:
                phase_sequence.append(pred['phase'])
                current_phase = pred['phase']
        
        return {
            'spike_id': len(phases),
            'start_frame': start_frame,
            'end_frame': end_frame,
            'duration_frames': end_frame - start_frame,
            'phase_sequence': phase_sequence,
            'phases': phases,
            'predictions': predictions
        }
    
    def calculate_spike_biomechanics(self, spike: Dict, 
                                    bio_calc: BiomechanicsCalculator,
                                    frames_data: List[Dict],
                                    fps: float):
        """Calculate biomechanical metrics for a spike."""
        
        # Extract pose data for spike frames
        start_idx = spike['start_frame']
        end_idx = spike['end_frame']
        
        hip_positions = []
        shoulder_positions = []
        wrist_positions = []
        torso_lengths = []
        angles_over_time = []
        
        for frame_idx in range(start_idx, end_idx + 1):
            if frame_idx < len(frames_data):
                frame_data = frames_data[frame_idx]
                
                if frame_data['pose_detected']:
                    metadata = frame_data['metadata']
                    pixel_landmarks = metadata.get('pixel_landmarks', {})
                    
                    # Use PIXEL coordinates for biomechanical calculations
                    if pixel_landmarks:
                        hip_y = (pixel_landmarks['right_hip'][1] + pixel_landmarks['left_hip'][1]) / 2
                        shoulder_y = (pixel_landmarks['right_shoulder'][1] + pixel_landmarks['left_shoulder'][1]) / 2
                        wrist_x = (pixel_landmarks['right_wrist'][0] + pixel_landmarks['left_wrist'][0]) / 2
                        wrist_y = (pixel_landmarks['right_wrist'][1] + pixel_landmarks['left_wrist'][1]) / 2
                        
                        # Calculate actual torso length in pixels
                        shoulder_x = (pixel_landmarks['right_shoulder'][0] + pixel_landmarks['left_shoulder'][0]) / 2
                        hip_x = (pixel_landmarks['right_hip'][0] + pixel_landmarks['left_hip'][0]) / 2
                        torso_length_pixels = np.sqrt((shoulder_x - hip_x)**2 + (shoulder_y - hip_y)**2)
                        
                        hip_positions.append(hip_y)
                        shoulder_positions.append(shoulder_y)
                        wrist_positions.append((wrist_x, wrist_y))
                        torso_lengths.append(torso_length_pixels)
                        angles_over_time.append(metadata['angles'])
        
        if not hip_positions:
            spike['biomechanics'] = {}
            return
        
        # Calculate pixels per cm (average torso length)
        avg_torso_length = np.mean(torso_lengths)
        pixels_per_cm = bio_calc.calculate_pixels_per_cm(avg_torso_length)
        
        # Calculate metrics
        jump_metrics = bio_calc.calculate_jump_height(hip_positions, pixels_per_cm)
        arm_metrics = bio_calc.calculate_arm_speed(wrist_positions, pixels_per_cm)
        separation_metrics = bio_calc.calculate_hip_shoulder_separation(
            hip_positions, shoulder_positions, pixels_per_cm
        )
        
        # Find peak angles
        peak_angles = {}
        for angle_name in angles_over_time[0].keys():
            angles = [frame_angles[angle_name] for frame_angles in angles_over_time]
            peak_angles[angle_name] = {
                'max': float(max(angles)),
                'min': float(min(angles)),
                'avg': float(np.mean(angles))
            }
        
        spike['biomechanics'] = {
            'jump': jump_metrics,
            'arm_speed': arm_metrics,
            'hip_shoulder_separation': separation_metrics,
            'peak_angles': peak_angles,
            'duration_seconds': float(spike['duration_frames'] / fps)
        }


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def generate_spike_report(analysis_results: Dict, output_dir: str):
    """Generate comprehensive analysis report."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate text report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("VOLLEYBALL SPIKE ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    video_info = analysis_results['video_info']
    report_lines.append(f"Video: {os.path.basename(video_info['path'])}")
    report_lines.append(f"Resolution: {video_info['resolution']}")
    report_lines.append(f"FPS: {video_info['fps']:.2f}")
    report_lines.append(f"Total frames: {video_info['total_frames']}")
    report_lines.append("")
    
    spikes = analysis_results['spikes']
    report_lines.append(f"Total spikes detected: {len(spikes)}")
    report_lines.append("")
    
    for i, spike in enumerate(spikes, 1):
        report_lines.append("=" * 70)
        report_lines.append(f"SPIKE #{i}")
        report_lines.append("=" * 70)
        report_lines.append(f"Frames: {spike['start_frame']} - {spike['end_frame']}")
        report_lines.append(f"Duration: {spike['duration_frames']} frames")
        report_lines.append(f"Phase sequence: {' → '.join(spike['phase_sequence'])}")
        report_lines.append("")
        
        if 'biomechanics' in spike and spike['biomechanics']:
            bio = spike['biomechanics']
            
            report_lines.append("BIOMECHANICAL METRICS:")
            report_lines.append("")
            
            # Jump
            if 'jump' in bio:
                jump = bio['jump']
                report_lines.append(f"🚀 JUMP METRICS:")
                report_lines.append(f"   Jump Height: {jump['jump_height_cm']:.1f} cm ({jump['jump_height_inches']:.1f} in)")
                report_lines.append(f"                {jump['jump_height_meters']:.2f} m ({jump['jump_height_feet']:.2f} ft)")
                report_lines.append(f"   Peak at frame: {jump['peak_frame']}")
                report_lines.append("")
            
            # Arm speed
            if 'arm_speed' in bio:
                arm = bio['arm_speed']
                report_lines.append(f"💪 ARM SWING METRICS:")
                report_lines.append(f"   Max Speed: {arm['max_speed_ms']:.2f} m/s ({arm['max_speed_mph']:.1f} mph / {arm['max_speed_kmh']:.1f} km/h)")
                report_lines.append(f"   Avg Speed: {arm['avg_speed_ms']:.2f} m/s ({arm['avg_speed_mph']:.1f} mph / {arm['avg_speed_kmh']:.1f} km/h)")
                report_lines.append(f"   Peak speed at frame: {arm['max_speed_frame']}")
                report_lines.append("")
            
            # Hip-shoulder separation
            if 'hip_shoulder_separation' in bio:
                sep = bio['hip_shoulder_separation']
                report_lines.append(f"📏 TORSO EXTENSION:")
                report_lines.append(f"   Max Separation: {sep['max_separation_cm']:.1f} cm ({sep['max_separation_inches']:.1f} in)")
                report_lines.append(f"   Avg Separation: {sep['avg_separation_cm']:.1f} cm ({sep['avg_separation_inches']:.1f} in)")
                report_lines.append("")
            
            # Key angles
            if 'peak_angles' in bio:
                report_lines.append(f"📐 PEAK JOINT ANGLES:")
                for angle_name, values in list(bio['peak_angles'].items())[:4]:
                    report_lines.append(f"   {angle_name}:")
                    report_lines.append(f"      Max: {values['max']:.1f}°")
                    report_lines.append(f"      Min: {values['min']:.1f}°")
                report_lines.append("")
        
        report_lines.append("")
    
    # Save report
    report_path = os.path.join(output_dir, 'spike_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Report saved to: {report_path}")
    
    # Save JSON with numpy type conversion
    json_path = os.path.join(output_dir, 'spike_analysis_data.json')
    
    # Convert numpy types to native Python types
    serializable_results = convert_numpy_types(analysis_results)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 JSON data saved to: {json_path}")


def generate_visualizations(analysis_results: Dict, output_dir: str):
    """Generate visualization plots for spike analysis."""
    
    spikes = analysis_results['spikes']
    
    for i, spike in enumerate(spikes, 1):
        if 'biomechanics' not in spike or not spike['biomechanics']:
            continue
        
        bio = spike['biomechanics']
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # Title
        fig.suptitle(f"Spike #{i} - Biomechanical Analysis", fontsize=16, fontweight='bold')
        
        # Jump height over time
        if 'jump' in bio and 'hip_positions' in bio['jump']:
            ax1 = fig.add_subplot(gs[0, 0])
            hip_positions = bio['jump']['hip_positions']
            
            # Convert to relative height (baseline = 0)
            baseline = bio['jump']['baseline_hip_y']
            relative_heights = [(baseline - h) for h in hip_positions]
            
            ax1.plot(relative_heights, linewidth=2, color='blue')
            jump_label = f"Max: {bio['jump']['jump_height_cm']:.1f} cm ({bio['jump']['jump_height_meters']:.2f} m / {bio['jump']['jump_height_feet']:.2f} ft)"
            ax1.axhline(y=bio['jump']['jump_height_cm'], color='r', linestyle='--', 
                       label=jump_label)
            ax1.axvline(x=bio['jump']['peak_frame'], color='orange', linestyle=':', 
                       alpha=0.7, label=f"Peak frame")
            ax1.set_title('Jump Height Profile', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Height (cm)')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(bottom=0)
        
        # Arm speed over time
        if 'arm_speed' in bio and 'speeds' in bio['arm_speed']:
            ax2 = fig.add_subplot(gs[0, 1])
            speeds = bio['arm_speed']['speeds']
            ax2.plot(speeds, linewidth=2)
            speed_label = f"Max: {bio['arm_speed']['max_speed_ms']:.2f} m/s ({bio['arm_speed']['max_speed_mph']:.1f} mph)"
            ax2.axhline(y=bio['arm_speed']['max_speed_ms'], color='r', linestyle='--', 
                       label=speed_label)
            ax2.set_title('Arm Swing Speed')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Speed (m/s)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Hip-shoulder separation
        if 'hip_shoulder_separation' in bio and 'separations' in bio['hip_shoulder_separation']:
            ax3 = fig.add_subplot(gs[1, 0])
            separations = bio['hip_shoulder_separation']['separations']
            ax3.plot(separations, linewidth=2, color='green')
            sep_label = f"Max: {bio['hip_shoulder_separation']['max_separation_cm']:.1f} cm ({bio['hip_shoulder_separation']['max_separation_inches']:.1f} in)"
            ax3.axhline(y=bio['hip_shoulder_separation']['max_separation_cm'], 
                       color='r', linestyle='--',
                       label=sep_label)
            ax3.set_title('Hip-Shoulder Separation (Torso Extension)')
            ax3.set_xlabel('Frame')
            ax3.set_ylabel('Separation (cm)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Joint angles
        if 'peak_angles' in bio:
            ax4 = fig.add_subplot(gs[1, 1])
            angle_names = list(bio['peak_angles'].keys())[:6]
            max_angles = [bio['peak_angles'][name]['max'] for name in angle_names]
            min_angles = [bio['peak_angles'][name]['min'] for name in angle_names]
            
            x = np.arange(len(angle_names))
            width = 0.35
            
            ax4.bar(x - width/2, max_angles, width, label='Max', alpha=0.8)
            ax4.bar(x + width/2, min_angles, width, label='Min', alpha=0.8)
            ax4.set_title('Peak Joint Angles')
            ax4.set_ylabel('Angle (degrees)')
            ax4.set_xticks(x)
            ax4.set_xticklabels([name.replace('_', '\n') for name in angle_names], 
                               rotation=45, ha='right', fontsize=8)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Phase distribution
        ax5 = fig.add_subplot(gs[2, :])
        phase_counts = {}
        for phase in spike['phase_sequence']:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        phases = list(phase_counts.keys())
        counts = list(phase_counts.values())
        colors = {'approach': 'orange', 'jump': 'green', 'swing': 'blue', 'land': 'cyan'}
        bar_colors = [colors.get(p, 'gray') for p in phases]
        
        ax5.bar(phases, counts, color=bar_colors, alpha=0.7)
        ax5.set_title('Phase Distribution')
        ax5.set_ylabel('Frame Count')
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'spike_{i:02d}_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization {i} saved to: {plot_path}")


def create_spike_frame_montage(spike: Dict, video_path: str, output_dir: str, 
                               spike_num: int, frames_data: List[Dict]):
    """
    Create a frame montage showing 5 frames per phase with peak moments highlighted.
    
    Args:
        spike: Spike dictionary with biomechanics data
        video_path: Path to source video
        output_dir: Output directory
        spike_num: Spike number
        frames_data: All frames data from analysis
    """
    
    # Validate spike has phases
    if 'phases' not in spike or not spike['phases']:
        print(f"⚠️  Spike {spike_num} has no phase data, skipping frame montage")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️  Could not open video for frame extraction")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Determine peak frames from biomechanics
    peak_frames = {}
    if 'biomechanics' in spike and spike['biomechanics']:
        bio = spike['biomechanics']
        
        # Add offset for spike start frame
        start_frame = spike['start_frame']
        
        if 'jump' in bio:
            peak_frames['jump_peak'] = start_frame + bio['jump']['peak_frame']
        
        if 'arm_speed' in bio:
            peak_frames['max_speed'] = start_frame + bio['arm_speed']['max_speed_frame']
        
        if 'hip_shoulder_separation' in bio:
            peak_frames['max_separation'] = start_frame + bio['hip_shoulder_separation']['max_separation_frame']
    
    # Get 5 frames per phase, ensuring peak frames are included
    phases = spike.get('phases', {})
    phase_order = ['approach', 'jump', 'swing', 'land']
    
    frames_to_extract = {}
    
    for phase_name in phase_order:
        if phase_name not in phases:
            continue
        
        phase_predictions = phases[phase_name]
        phase_frames = sorted([p['frame_idx'] for p in phase_predictions])
        
        if not phase_frames:
            continue
        
        # Select 5 representative frames
        selected_frames = []
        
        if len(phase_frames) <= 5:
            selected_frames = phase_frames
        else:
            # Always include first and last
            selected_frames.append(phase_frames[0])
            selected_frames.append(phase_frames[-1])
            
            # Add peak frames if they're in this phase
            for peak_type, peak_frame in peak_frames.items():
                if phase_frames[0] <= peak_frame <= phase_frames[-1]:
                    if peak_frame not in selected_frames:
                        selected_frames.append(peak_frame)
            
            # Fill remaining slots with evenly spaced frames
            remaining_slots = 5 - len(selected_frames)
            if remaining_slots > 0:
                # Get middle frames
                middle_frames = [f for f in phase_frames if f not in selected_frames]
                if middle_frames:
                    step = max(1, len(middle_frames) // (remaining_slots + 1))
                    for i in range(remaining_slots):
                        idx = min((i + 1) * step, len(middle_frames) - 1)
                        if middle_frames[idx] not in selected_frames:
                            selected_frames.append(middle_frames[idx])
            
            selected_frames = sorted(selected_frames)[:5]
        
        frames_to_extract[phase_name] = selected_frames
    
    # Extract frames
    extracted_frames = {}
    for phase_name, frame_list in frames_to_extract.items():
        extracted_frames[phase_name] = []
        
        for frame_num in frame_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Annotate frame
                frame_copy = frame.copy()
                
                # Add frame number
                cv2.putText(frame_copy, f"Frame: {frame_num}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add timestamp
                timestamp = frame_num / fps
                cv2.putText(frame_copy, f"Time: {timestamp:.2f}s", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Highlight if peak frame
                is_peak = False
                peak_label = ""
                
                for peak_type, peak_frame in peak_frames.items():
                    if frame_num == peak_frame:
                        is_peak = True
                        if peak_type == 'jump_peak':
                            peak_label = "🚀 PEAK JUMP"
                        elif peak_type == 'max_speed':
                            peak_label = "💪 MAX SPEED"
                        elif peak_type == 'max_separation':
                            peak_label = "📏 MAX EXTENSION"
                        break
                
                if is_peak:
                    # Add colored border
                    cv2.rectangle(frame_copy, (0, 0), (frame.shape[1]-1, frame.shape[0]-1),
                                (0, 255, 0), 10)
                    # Add label
                    cv2.putText(frame_copy, peak_label, (10, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                
                extracted_frames[phase_name].append(frame_copy)
    
    cap.release()
    
    # Create montage
    if not extracted_frames:
        print(f"⚠️  No frames extracted for spike {spike_num}")
        return
    
    # Determine grid size
    num_phases = len(extracted_frames)
    max_frames_per_phase = max(len(frames) for frames in extracted_frames.values())
    
    # Resize frames to consistent size
    target_width = 320
    target_height = 240
    
    # Create figure
    fig, axes = plt.subplots(num_phases, max_frames_per_phase, 
                            figsize=(max_frames_per_phase * 4, num_phases * 3))
    
    # Handle edge cases for axes indexing
    if num_phases == 1 and max_frames_per_phase == 1:
        axes = [[axes]]  # Make it 2D
    elif num_phases == 1:
        axes = [axes]  # Make it 2D (one row)
    elif max_frames_per_phase == 1:
        axes = [[ax] for ax in axes]  # Make it 2D (one column)
    
    fig.suptitle(f"Spike #{spike_num} - Frame Montage (5 frames per phase)", 
                fontsize=16, fontweight='bold')
    
    # Plot frames - only iterate through phases that exist in extracted_frames
    actual_phase_order = [p for p in phase_order if p in extracted_frames]
    
    for phase_idx, phase_name in enumerate(actual_phase_order):
        frames = extracted_frames[phase_name]
        
        for frame_idx in range(max_frames_per_phase):
            ax = axes[phase_idx][frame_idx]
            
            if frame_idx < len(frames):
                frame = frames[frame_idx]
                # Resize
                resized = cv2.resize(frame, (target_width, target_height))
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb_frame)
                
                # Add phase label on first frame
                if frame_idx == 0:
                    ax.set_ylabel(phase_name.upper(), fontsize=12, fontweight='bold')
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    
    montage_path = os.path.join(output_dir, f'spike_{spike_num:02d}_frames.png')
    plt.savefig(montage_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️  Frame montage saved to: {montage_path}")


def main():
    """Main analysis function."""
    
    print("=" * 70)
    print("🏐 COMPREHENSIVE OFFLINE SPIKE ANALYSIS")
    print("=" * 70)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video not found: {VIDEO_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found: {MODEL_PATH}")
        return
    
    # Initialize analyzer
    analyzer = OfflineSpikeAnalyzer(
        MODEL_PATH,
        METADATA_PATH,
        BUFFER_SIZE,
        PLAYER_HEIGHT_CM
    )
    
    # Analyze video
    print(f"\n🎬 Analyzing video: {os.path.basename(VIDEO_PATH)}")
    
    results = analyzer.analyze_video(
        VIDEO_PATH,
        threshold=PREDICTION_THRESHOLD,
        verbose=VERBOSE
    )
    
    # Generate outputs
    if GENERATE_REPORT:
        print(f"\n📝 Generating reports...")
        generate_spike_report(results, OUTPUT_DIR)
    
    if GENERATE_VISUALIZATIONS:
        print(f"\n📊 Generating visualizations...")
        generate_visualizations(results, OUTPUT_DIR)
        
        # Generate frame montages
        print(f"\n🖼️  Generating frame montages...")
        for i, spike in enumerate(results['spikes'], 1):
            create_spike_frame_montage(
                spike, 
                VIDEO_PATH, 
                OUTPUT_DIR, 
                i, 
                results['frames_data']
            )
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nDetected {len(results['spikes'])} spikes")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if results['spikes']:
        print(f"\nSpike Summary:")
        for i, spike in enumerate(results['spikes'], 1):
            phases = ' → '.join(spike['phase_sequence'])
            print(f"   Spike {i}: {phases}")
            
            if 'biomechanics' in spike and spike['biomechanics']:
                bio = spike['biomechanics']
                if 'jump' in bio:
                    jump_str = f"{bio['jump']['jump_height_cm']:.1f} cm ({bio['jump']['jump_height_meters']:.2f} m / {bio['jump']['jump_height_feet']:.2f} ft)"
                    print(f"      Jump: {jump_str}")
                if 'arm_speed' in bio:
                    speed_str = f"{bio['arm_speed']['max_speed_ms']:.2f} m/s ({bio['arm_speed']['max_speed_mph']:.1f} mph)"
                    print(f"      Max arm speed: {speed_str}")


if __name__ == "__main__":
    main()

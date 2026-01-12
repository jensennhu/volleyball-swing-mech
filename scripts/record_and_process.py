#!/usr/bin/env python3
"""
Record, Save, and Process Volleyball Video
===========================================

Three-stage workflow:
1. Record video from webcam/camera
2. Save video file
3. Process video with spike phase recognition

Usage: python record_and_process.py
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import deque
import time
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


# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================

# Recording settings
VIDEO_SOURCE = 0                        # 0 = webcam, or camera index
RECORDING_FPS = 30                      # Frames per second
RECORDING_WIDTH = 1280                  # Video width
RECORDING_HEIGHT = 720                  # Video height
OUTPUT_VIDEO_DIR = "recorded_videos"    # Where to save recorded videos
VIDEO_CODEC = 'mp4v'                    # 'mp4v' or 'avc1' for .mp4

# Processing settings

MODEL_PATH = "lstm_models/spike_phase_classifier.keras"
METADATA_PATH = "lstm_models/spike_phase_classifier_metadata.json"
BUFFER_SIZE = 10                        # Number of frames for LSTM
PREDICTION_THRESHOLD = 0.7              # Minimum confidence
UPDATE_EVERY_N_FRAMES = 3               # Update prediction frequency

# Output settings
SAVE_PROCESSED_VIDEO = True             # Save annotated video
SAVE_PREDICTIONS_JSON = True            # Save predictions to JSON
PROCESSED_VIDEO_DIR = "processed_videos"
# ============================================================================


class VideoRecorder:
    """Record video from camera and save to disk."""
    
    def __init__(self, video_source: int = 0, output_dir: str = "recorded_videos",
                 fps: int = 30, width: int = 1280, height: int = 720):
        self.video_source = video_source
        self.output_dir = output_dir
        self.fps = fps
        self.width = width
        self.height = height
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
        
    def start_recording(self) -> str:
        """Start recording video."""
        # Open camera
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.video_source}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Create output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_filename = f"volleyball_recording_{timestamp}.mp4"
        self.output_path = os.path.join(self.output_dir, self.output_filename)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"🎥 Recording started: {self.output_filename}")
        print(f"   Resolution: {self.width}x{self.height} @ {self.fps} FPS")
        
        return self.output_path
    
    def record_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture and record one frame."""
        if not self.is_recording:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.writer.write(frame)
            self.frame_count += 1
            
            # Add recording indicator
            frame_display = frame.copy()
            
            # Recording indicator (red circle)
            cv2.circle(frame_display, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame_display, "REC", (55, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Timer
            elapsed = time.time() - self.start_time
            timer_text = time.strftime("%M:%S", time.gmtime(elapsed))
            cv2.putText(frame_display, timer_text, (55, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Frame count
            cv2.putText(frame_display, f"Frames: {self.frame_count}", (10, self.height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Instructions
            instructions = [
                "SPACE: Stop & Save",
                "ESC: Cancel",
                "R: Reset recording"
            ]
            for i, instruction in enumerate(instructions):
                y_pos = self.height - 80 + (i * 25)
                cv2.putText(frame_display, instruction, (self.width - 250, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            return True, frame_display
        
        return False, None
    
    def stop_recording(self) -> Dict:
        """Stop recording and save video."""
        if not self.is_recording:
            return {}
        
        self.is_recording = False
        
        if self.writer:
            self.writer.release()
        
        if self.cap:
            self.cap.release()
        
        duration = time.time() - self.start_time
        
        metadata = {
            'filename': self.output_filename,
            'path': self.output_path,
            'duration': duration,
            'frames': self.frame_count,
            'fps': self.fps,
            'resolution': f"{self.width}x{self.height}",
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"\n✅ Recording saved: {self.output_path}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Frames: {self.frame_count}")
        
        return metadata
    
    def reset_recording(self):
        """Restart recording (discard current)."""
        if self.writer:
            self.writer.release()
        
        self.frame_count = 0
        self.start_time = time.time()
        
        # Create new writer
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        print("🔄 Recording reset")


class VideoProcessor:
    """Process recorded video with spike phase recognition."""
    
    def __init__(self, model_path: str, metadata_path: str, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.feature_buffer = deque(maxlen=buffer_size)
        
        # Load model
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.label_names = self.metadata['label_names']
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print(f"✅ Processor initialized")
        print(f"   Model classes: {self.label_names}")
    
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
    
    def extract_features_from_landmarks(self, landmarks, frame_width: int, frame_height: int):
        """Extract features from MediaPipe landmarks."""
        if landmarks is None:
            return None, None, None
        
        try:
            # Get landmark coordinates
            def get_coords(landmark_id):
                lm = landmarks.landmark[landmark_id]
                return (int(lm.x * frame_width), int(lm.y * frame_height))
            
            # Right side
            r_shoulder = get_coords(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
            r_elbow = get_coords(self.mp_pose.PoseLandmark.RIGHT_ELBOW)
            r_wrist = get_coords(self.mp_pose.PoseLandmark.RIGHT_WRIST)
            r_hip = get_coords(self.mp_pose.PoseLandmark.RIGHT_HIP)
            r_knee = get_coords(self.mp_pose.PoseLandmark.RIGHT_KNEE)
            r_ankle = get_coords(self.mp_pose.PoseLandmark.RIGHT_ANKLE)
            
            # Left side
            l_shoulder = get_coords(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            l_elbow = get_coords(self.mp_pose.PoseLandmark.LEFT_ELBOW)
            l_wrist = get_coords(self.mp_pose.PoseLandmark.LEFT_WRIST)
            l_hip = get_coords(self.mp_pose.PoseLandmark.LEFT_HIP)
            l_knee = get_coords(self.mp_pose.PoseLandmark.LEFT_KNEE)
            l_ankle = get_coords(self.mp_pose.PoseLandmark.LEFT_ANKLE)
            
            # Calculate angles
            angles = {
                'right_shoulder': self.calculate_angle(r_hip, r_shoulder, r_elbow),
                'right_elbow': self.calculate_angle(r_shoulder, r_elbow, r_wrist),
                'right_hip': self.calculate_angle(r_shoulder, r_hip, r_knee),
                'right_knee': self.calculate_angle(r_hip, r_knee, r_ankle),
                'left_shoulder': self.calculate_angle(l_hip, l_shoulder, l_elbow),
                'left_elbow': self.calculate_angle(l_shoulder, l_elbow, l_wrist),
                'left_hip': self.calculate_angle(l_shoulder, l_hip, l_knee),
                'left_knee': self.calculate_angle(l_hip, l_knee, l_ankle)
            }
            
            # Extract features
            features = []
            
            # Joint angles (8 features)
            for key in ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                       'left_shoulder', 'left_elbow', 'left_hip', 'left_knee']:
                features.append(angles[key])
            
            # Landmark positions (24 features)
            for coords in [r_shoulder, r_elbow, r_wrist, r_hip, r_knee, r_ankle,
                          l_shoulder, l_elbow, l_wrist, l_hip, l_knee, l_ankle]:
                features.extend([coords[0] / frame_width, coords[1] / frame_height])
            
            # Confidence (1 feature)
            confidence = np.mean([
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility,
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].visibility
            ])
            features.append(confidence)
            
            return np.array(features, dtype=np.float32), angles, confidence
            
        except Exception as e:
            return None, None, None
    
    def predict(self) -> Tuple[Optional[str], float, List[float]]:
        """Make prediction from current buffer."""
        if len(self.feature_buffer) < self.buffer_size:
            return None, 0.0, []
        
        try:
            # Prepare sequence
            X = np.array(list(self.feature_buffer), dtype=np.float32)
            X = X.reshape(1, self.buffer_size, -1)
            
            # Normalize
            n_frames, n_features = X.shape[1], X.shape[2]
            X_reshaped = X.reshape(-1, n_features)
            
            if not self.scaler_fitted:
                self.scaler.fit(X_reshaped)
                self.scaler_fitted = True
            
            X_normalized = self.scaler.transform(X_reshaped)
            X = X_normalized.reshape(1, n_frames, n_features)
            
            # Predict
            predictions_proba = self.model.predict(X, verbose=0)
            pred_idx = np.argmax(predictions_proba[0])
            confidence = float(predictions_proba[0][pred_idx])
            all_probs = predictions_proba[0].tolist()
            
            if confidence >= PREDICTION_THRESHOLD:
                predicted_label = self.label_names[pred_idx]
                return predicted_label, confidence, all_probs
            
        except Exception as e:
            print(f"Error during prediction: {e}")
        
        return None, 0.0, []
    
    def process_video(self, video_path: str, save_video: bool = True, 
                     save_json: bool = True) -> Dict:
        """Process video and return results."""
        
        print(f"\n🎬 Processing video: {os.path.basename(video_path)}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps:.1f}")
        print(f"   Total frames: {total_frames}")
        
        # Setup output video writer
        writer = None
        if save_video:
            os.makedirs(PROCESSED_VIDEO_DIR, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = os.path.join(PROCESSED_VIDEO_DIR, f"{base_name}_processed.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"   Output: {output_video_path}")
        
        # Processing
        frame_count = 0
        predictions_log = []
        current_prediction = None
        current_confidence = 0.0
        
        print(f"\n📊 Processing frames...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            # Draw skeleton
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract features
                feature_data = self.extract_features_from_landmarks(
                    results.pose_landmarks, width, height
                )
                
                if feature_data[0] is not None:
                    features, angles, confidence = feature_data
                    self.feature_buffer.append(features)
                    
                    # Make prediction
                    if len(self.feature_buffer) == self.buffer_size:
                        if frame_count % UPDATE_EVERY_N_FRAMES == 0:
                            current_prediction, current_confidence, all_probs = self.predict()
                            
                            if current_prediction:
                                predictions_log.append({
                                    'frame': frame_count,
                                    'timestamp': frame_count / fps,
                                    'prediction': current_prediction,
                                    'confidence': current_confidence,
                                    'probabilities': dict(zip(self.label_names, all_probs))
                                })
            
            # Draw prediction overlay
            if current_prediction and current_confidence >= PREDICTION_THRESHOLD:
                # Phase colors
                phase_colors = {
                    'approach': (0, 165, 255),
                    'jump': (0, 255, 0),
                    'swing': (255, 0, 0),
                    'land': (255, 255, 0)
                }
                color = phase_colors.get(current_prediction, (255, 255, 255))
                
                # Draw prediction box
                cv2.rectangle(frame, (10, height - 100), (350, height - 10), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, height - 100), (350, height - 10), color, 2)
                
                cv2.putText(frame, f"Phase: {current_prediction.upper()}", (20, height - 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Confidence: {current_confidence:.1%}", (20, height - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Frame counter
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            if writer:
                writer.write(frame)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        print(f"\n✅ Processing complete!")
        print(f"   Processed {frame_count} frames")
        print(f"   Detected {len(predictions_log)} phase predictions")
        
        # Save JSON
        results = {
            'video_path': video_path,
            'processed_frames': frame_count,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'total_predictions': len(predictions_log),
            'predictions': predictions_log
        }
        
        if save_json:
            json_path = os.path.join(PROCESSED_VIDEO_DIR, f"{base_name}_predictions.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   Predictions saved: {json_path}")
        
        # Summary
        if predictions_log:
            from collections import Counter
            phase_counts = Counter([p['prediction'] for p in predictions_log])
            print(f"\n📈 Phase Summary:")
            for phase, count in phase_counts.most_common():
                print(f"   {phase}: {count} detections")
        
        return results
    
    def close(self):
        """Clean up resources."""
        self.pose.close()


def main():
    """Main function."""
    
    print("=" * 70)
    print("🏐 RECORD, SAVE, AND PROCESS VOLLEYBALL VIDEO")
    print("=" * 70)
    print()
    
    # Stage 1: Recording
    print("STAGE 1: RECORDING")
    print("-" * 70)
    print("Controls:")
    print("  SPACE: Stop recording and save")
    print("  ESC: Cancel recording")
    print("  R: Reset recording (start over)")
    print()
    input("Press ENTER to start recording...")
    
    recorder = VideoRecorder(
        video_source=VIDEO_SOURCE,
        output_dir=OUTPUT_VIDEO_DIR,
        fps=RECORDING_FPS,
        width=RECORDING_WIDTH,
        height=RECORDING_HEIGHT
    )
    
    video_path = recorder.start_recording()
    
    try:
        while True:
            ret, frame = recorder.record_frame()
            
            if not ret:
                break
            
            cv2.imshow('Recording', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space: Stop and save
                break
            elif key == 27:  # ESC: Cancel
                print("\n❌ Recording cancelled")
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):  # R: Reset
                recorder.reset_recording()
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        metadata = recorder.stop_recording()
    
    if not metadata:
        print("❌ No video recorded")
        return
    
    video_path = metadata['path']
    
    # Stage 2: Processing
    print("\n" + "=" * 70)
    print("STAGE 2: PROCESSING")
    print("-" * 70)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found: {MODEL_PATH}")
        print("   Train a model first using train_lstm_spike_aware.py")
        print(f"   Video saved at: {video_path}")
        return
    
    if not os.path.exists(METADATA_PATH):
        print(f"❌ Error: Metadata not found: {METADATA_PATH}")
        print(f"   Video saved at: {video_path}")
        return
    
    print(f"\nProcessing recorded video...")
    
    processor = VideoProcessor(MODEL_PATH, METADATA_PATH, BUFFER_SIZE)
    
    try:
        results = processor.process_video(
            video_path,
            save_video=SAVE_PROCESSED_VIDEO,
            save_json=SAVE_PREDICTIONS_JSON
        )
    finally:
        processor.close()
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"\nOriginal video: {video_path}")
    if SAVE_PROCESSED_VIDEO:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        processed_path = os.path.join(PROCESSED_VIDEO_DIR, f"{base_name}_processed.mp4")
        print(f"Processed video: {processed_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Real-Time Volleyball Spike Phase Recognition (NORMALIZED)
==========================================================

Processes live video or webcam feed to recognize volleyball spike phases in real-time.
Uses NORMALIZED pose features for consistent recognition across different videos/cameras.

Key improvements:
- Position-invariant (works anywhere in frame)
- Scale-invariant (works at any distance)
- Resolution-invariant (works on any video size)

Usage: python realtime_spike_recognition_normalized.py

Controls:
  - Press 'q' to quit
  - Press 'r' to reset buffer
  - Press 's' to save current sequence
"""

import cv2
import numpy as np
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# Import normalized feature extractor
from src.preprocessing.normalized_pose_features import NormalizedPoseExtractor


# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
MODEL_PATH = "models/spike_phase_classifier_normalized_3layers.keras"
METADATA_PATH = "models/spike_phase_classifier_normalized_3layers_metadata.json"

# Video source
#VIDEO_SOURCE = 0                        # 0 = webcam, or path to video file
VIDEO_SOURCE = "data/raw/videos/recorded_videos/hitting-session.mp4"

# Buffer settings
BUFFER_SIZE = 10                       # Number of frames to buffer
PREDICTION_THRESHOLD = 0.7              # Minimum confidence to display prediction
UPDATE_EVERY_N_FRAMES = 5               # Update prediction every N frames (smoother)

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
SHOW_SKELETON = True
SHOW_ANGLES = True
SHOW_CONFIDENCE = True
SHOW_BUFFER_STATUS = True

# Recording
SAVE_SEQUENCES = True                  # Save recognized sequences to disk
OUTPUT_DIR = "realtime_captures"
# ============================================================================


class RealtimeSpikeRecognizerNormalized:
    """Real-time volleyball spike phase recognition with NORMALIZED features."""
    
    def __init__(self, model_path: str, metadata_path: str, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.feature_buffer = deque(maxlen=buffer_size)
        
        # Load model
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.label_names = self.metadata['label_names']
        self.pose_type = self.metadata.get('pose_type', 'mediapipe')
        
        # Scaler for normalization
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Prediction state
        self.current_prediction = None
        self.current_confidence = 0.0
        self.prediction_history = []
        self.frame_count = 0
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize NORMALIZED feature extractor
        self.feature_extractor = NormalizedPoseExtractor()
        
        print(f"✅ Model loaded: {len(self.label_names)} classes")
        print(f"   Labels: {self.label_names}")
        print(f"   Buffer size: {buffer_size} frames")
        print(f"   ✨ Using NORMALIZED features (position + scale invariant)")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str], float]:
        """
        Process a single frame and update prediction using NORMALIZED features.
        
        Returns:
            - Annotated frame
            - Current prediction (or None)
            - Confidence score
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        # Draw skeleton
        if results.pose_landmarks and SHOW_SKELETON:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        # Extract NORMALIZED features
        if results.pose_landmarks:
            # Use normalized feature extractor
            features, metadata = self.feature_extractor.extract_features_with_metadata(
                results.pose_landmarks
            )
            
            if features is not None:
                # Add to buffer
                self.feature_buffer.append(features)
                self.frame_buffer.append(frame.copy())
                
                # Display angles
                if SHOW_ANGLES and metadata.get('angles'):
                    y_pos = 30
                    for key, angle in list(metadata['angles'].items())[:4]:  # Show first 4 angles
                        cv2.putText(frame, f"{key}: {angle:.1f}°", (10, y_pos),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_pos += 20
                
                # Display confidence
                if SHOW_CONFIDENCE:
                    conf = metadata.get('confidence', 0.0)
                    cv2.putText(frame, f"Pose conf: {conf:.2f}", (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Make prediction if buffer is full
                if len(self.feature_buffer) == self.buffer_size:
                    if self.frame_count % UPDATE_EVERY_N_FRAMES == 0:
                        self.current_prediction, self.current_confidence = self.predict()
                
                self.frame_count += 1
        
        # Draw prediction overlay
        self.draw_overlay(frame)
        
        return frame, self.current_prediction, self.current_confidence
    
    def predict(self) -> Tuple[Optional[str], float]:
        """Make prediction from current buffer using NORMALIZED features."""
        if len(self.feature_buffer) < self.buffer_size:
            return None, 0.0
        
        try:
            # Prepare sequence (features are already normalized!)
            X = np.array(list(self.feature_buffer), dtype=np.float32)
            X = X.reshape(1, self.buffer_size, -1)  # (1, 10, 33)
            
            # Normalize (StandardScaler on top of geometric normalization)
            n_frames, n_features = X.shape[1], X.shape[2]
            X_reshaped = X.reshape(-1, n_features)
            
            if not self.scaler_fitted:
                # Fit scaler on first sequence
                self.scaler.fit(X_reshaped)
                self.scaler_fitted = True
            
            X_normalized = self.scaler.transform(X_reshaped)
            X = X_normalized.reshape(1, n_frames, n_features)
            
            # Predict
            predictions_proba = self.model.predict(X, verbose=0)
            pred_idx = np.argmax(predictions_proba[0])
            confidence = float(predictions_proba[0][pred_idx])
            
            if confidence >= PREDICTION_THRESHOLD:
                predicted_label = self.label_names[pred_idx]
                
                # Add to history
                self.prediction_history.append({
                    'label': predicted_label,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
                
                return predicted_label, confidence
            
        except Exception as e:
            print(f"Error during prediction: {e}")
        
        return None, 0.0
    
    def draw_overlay(self, frame: np.ndarray):
        """Draw prediction overlay on frame."""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Top bar
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "Volleyball Spike Recognition (NORMALIZED)", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Normalization indicator
        cv2.putText(frame, "✓ Position+Scale Invariant", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Buffer status
        if SHOW_BUFFER_STATUS:
            buffer_text = f"Buffer: {len(self.feature_buffer)}/{self.buffer_size}"
            buffer_color = (0, 255, 0) if len(self.feature_buffer) == self.buffer_size else (0, 165, 255)
            cv2.putText(frame, buffer_text, (width - 200, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, buffer_color, 2)
        
        # Current prediction
        if self.current_prediction and self.current_confidence >= PREDICTION_THRESHOLD:
            # Prediction box
            box_height = 120
            box_y = height - box_height - 20
            cv2.rectangle(overlay, (20, box_y), (400, box_y + box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Phase name
            phase_colors = {
                'approach': (0, 165, 255),    # Orange
                'jump': (0, 255, 0),          # Green
                'swing': (255, 0, 0),         # Blue
                'land': (255, 255, 0)         # Cyan
            }
            color = phase_colors.get(self.current_prediction, (255, 255, 255))
            
            cv2.putText(frame, f"Phase: {self.current_prediction.upper()}", 
                       (40, box_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Confidence
            confidence_text = f"Confidence: {self.current_confidence:.1%}"
            cv2.putText(frame, confidence_text, (40, box_y + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Waiting message
            cv2.putText(frame, "Waiting for motion...", (20, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        
        # Controls
        controls = "Controls: Q=Quit | R=Reset | S=Save"
        cv2.putText(frame, controls, (width - 400, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def reset_buffer(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()
        self.feature_buffer.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        print("🔄 Buffer reset")
    
    def save_sequence(self, output_dir: str = "realtime_captures"):
        """Save current sequence to disk."""
        if len(self.frame_buffer) < self.buffer_size:
            print("⚠️  Buffer not full, cannot save")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        phase = self.current_prediction or "unknown"
        
        sequence_dir = os.path.join(output_dir, f"{timestamp}_{phase}")
        os.makedirs(sequence_dir, exist_ok=True)
        
        # Save frames
        for i, frame in enumerate(self.frame_buffer):
            filename = f"frame_{i:02d}.png"
            cv2.imwrite(os.path.join(sequence_dir, filename), frame)
        
        # Save metadata (including normalized features)
        metadata = {
            'timestamp': timestamp,
            'predicted_phase': self.current_prediction,
            'confidence': float(self.current_confidence),
            'buffer_size': len(self.frame_buffer),
            'normalized_features': True,  # Flag that these are normalized
            'normalization_method': 'pelvis_centered_torso_scaled'
        }
        
        with open(os.path.join(sequence_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Sequence saved to: {sequence_dir}")
    
    def close(self):
        """Clean up resources."""
        self.pose.close()


def main():
    """Main function to run real-time recognition with NORMALIZED features."""
    
    print("=" * 70)
    print("🏐 REAL-TIME VOLLEYBALL SPIKE PHASE RECOGNITION (NORMALIZED)")
    print("=" * 70)
    print()
    print("✨ Using NORMALIZED pose features:")
    print("   - Position-invariant (works anywhere in frame)")
    print("   - Scale-invariant (works at any distance)")
    print("   - Resolution-invariant (works on any video size)")
    print()
    
    # Check files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found: {MODEL_PATH}")
        print("   Train a model first using train_lstm_spike_aware.py")
        print("   IMPORTANT: Model must be trained on NORMALIZED features!")
        return
    
    if not os.path.exists(METADATA_PATH):
        print(f"❌ Error: Metadata not found: {METADATA_PATH}")
        return
    
    # Initialize recognizer
    recognizer = RealtimeSpikeRecognizerNormalized(MODEL_PATH, METADATA_PATH, BUFFER_SIZE)
    
    # Open video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {VIDEO_SOURCE}")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    
    print(f"✅ Video source opened: {VIDEO_SOURCE}")
    print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print()
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset buffer")
    print("  - Press 's' to save current sequence")
    print()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video or error reading frame")
                break
            
            # Process frame with NORMALIZED features
            annotated_frame, prediction, confidence = recognizer.process_frame(frame)
            
            # Display
            cv2.imshow('Volleyball Spike Recognition (NORMALIZED)', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('r'):
                recognizer.reset_buffer()
            elif key == ord('s'):
                if SAVE_SEQUENCES:
                    recognizer.save_sequence(OUTPUT_DIR)
                else:
                    print("⚠️  Sequence saving is disabled. Set SAVE_SEQUENCES=True")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()
        
        print("\n" + "=" * 70)
        print("✅ Session complete")
        print("=" * 70)
        
        if recognizer.prediction_history:
            print(f"\nRecognized {len(recognizer.prediction_history)} phases:")
            from collections import Counter
            counts = Counter([p['label'] for p in recognizer.prediction_history])
            for phase, count in counts.most_common():
                print(f"  {phase}: {count} times")


if __name__ == "__main__":
    main()
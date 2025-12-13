#!/usr/bin/env python3
"""
Pure Python LSTM Inference Script
==================================

Use a trained LSTM model to make predictions - no command line needed.
Just run: python predict_lstm_pure_python.py
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("❌ Error: TensorFlow not installed. Run: pip install tensorflow --break-system-packages")
    exit(1)

from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
MODEL_PATH = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/output_lstm/spike_lstm.keras"            # Path to trained model
METADATA_PATH = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/output_lstm/spike_lstm_metadata.json" # Path to metadata
POSE_DATA_PATH = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/frames_with_pose_yol_l/pose_data_yolo.json"     # Pose data to predict on
OUTPUT_PATH = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/output_lstm/predictions.json"                        # Where to save predictions (optional)
SAVE_PREDICTIONS = True                                 # Save predictions to JSON
# ============================================================================


class LSTMInference:
    """Inference engine for trained LSTM models."""
    
    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.scaler = StandardScaler()
        
        self.load_model()
        self.load_metadata()
    
    def load_model(self):
        """Load the trained model."""
        print(f"📂 Loading model from: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully")
    
    def load_metadata(self):
        """Load model metadata."""
        print(f"📂 Loading metadata from: {self.metadata_path}")
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"✅ Metadata loaded: {self.metadata['num_classes']} classes")
        print(f"   Labels: {self.metadata['label_names']}")
    
    def extract_features_mediapipe(self, pose_entry: Dict) -> np.ndarray:
        """Extract features from MediaPipe pose data."""
        if not pose_entry.get('pose_detected', False):
            return None
        
        landmarks = pose_entry.get('landmarks', {})
        angles = pose_entry.get('angles', {})
        
        features = []
        
        for key in ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                    'left_shoulder', 'left_elbow', 'left_hip', 'left_knee']:
            features.append(angles.get(key, 0.0))
        
        for key in ['right_shoulder', 'right_elbow', 'right_wrist', 
                    'right_hip', 'right_knee', 'right_ankle',
                    'left_shoulder', 'left_elbow', 'left_wrist',
                    'left_hip', 'left_knee', 'left_ankle']:
            coords = landmarks.get(key, [0, 0])
            features.extend(coords)
        
        features.append(pose_entry.get('confidence', 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def extract_features_yolo(self, pose_entry: Dict) -> np.ndarray:
        """Extract features from YOLO pose data."""
        if not pose_entry.get('pose_detected', False):
            return None
        
        keypoints = pose_entry.get('keypoints', {})
        angles = pose_entry.get('angles', {})
        
        features = []
        
        for key in ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                    'left_shoulder', 'left_elbow', 'left_hip', 'left_knee']:
            features.append(angles.get(key, 0.0))
        
        for key in ['right_shoulder', 'right_elbow', 'right_wrist', 
                    'right_hip', 'right_knee', 'right_ankle',
                    'left_shoulder', 'left_elbow', 'left_wrist',
                    'left_hip', 'left_knee', 'left_ankle']:
            kpt = keypoints.get(key)
            if kpt:
                features.extend([kpt['x'], kpt['y'], kpt['confidence']])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        features.append(pose_entry.get('confidence', 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def load_pose_data(self, json_path: str) -> np.ndarray:
        """Load and extract features from pose data."""
        print(f"\n📂 Loading pose data from: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        pose_data = data['pose_data']
        pose_type = self.metadata['pose_type']
        
        print(f"🔍 Extracting features ({pose_type} format)...")
        
        feature_extractor = (self.extract_features_mediapipe 
                           if pose_type == 'mediapipe' 
                           else self.extract_features_yolo)
        
        all_features = []
        valid_indices = []
        
        for i, entry in enumerate(pose_data):
            features = feature_extractor(entry)
            if features is not None:
                all_features.append(features)
                valid_indices.append(i)
        
        print(f"✅ Extracted {len(all_features)} valid frames")
        
        X = np.array(all_features, dtype=np.float32)
        
        return X, valid_indices
    
    def create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences from features."""
        seq_len = self.metadata['sequence_length']
        stride = self.metadata['stride']
        
        print(f"\n🔄 Creating sequences (length={seq_len}, stride={stride})...")
        
        sequences = []
        
        for i in range(0, len(X) - seq_len + 1, stride):
            sequence = X[i:i + seq_len]
            sequences.append(sequence)
        
        X_seq = np.array(sequences, dtype=np.float32)
        
        print(f"✅ Created {len(sequences)} sequences")
        
        return X_seq
    
    def predict(self, X_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on sequences."""
        print(f"\n🔮 Making predictions on {len(X_seq)} sequences...")
        
        # Normalize
        n_samples, seq_len, n_features = X_seq.shape
        X_flat = X_seq.reshape(-1, n_features)
        X_normalized = self.scaler.fit_transform(X_flat)
        X_seq_norm = X_normalized.reshape(n_samples, seq_len, n_features)
        
        # Predict
        predictions = self.model.predict(X_seq_norm, verbose=0)
        
        if self.metadata['num_classes'] == 2:
            # Binary classification
            y_pred = (predictions > 0.5).astype(int).flatten()
            y_proba = predictions
        else:
            # Multi-class classification
            y_pred = np.argmax(predictions, axis=1)
            y_proba = predictions
        
        # Convert to label names
        label_names = self.metadata['label_names']
        predicted_labels = [label_names[idx] for idx in y_pred]
        
        print("✅ Predictions complete")
        
        return predicted_labels, y_proba
    
    def predict_from_file(self, pose_data_path: str) -> Dict:
        """Complete inference pipeline from pose data file."""
        # Load and extract features
        X, valid_indices = self.load_pose_data(pose_data_path)
        
        # Create sequences
        X_seq = self.create_sequences(X)
        
        # Predict
        predicted_labels, probabilities = self.predict(X_seq)
        
        # Compile results
        results = {
            'predictions': predicted_labels,
            'probabilities': probabilities.tolist(),
            'num_sequences': len(X_seq),
            'label_names': self.metadata['label_names'],
            'sequence_length': self.metadata['sequence_length']
        }
        
        return results
    
    def print_prediction_summary(self, results: Dict):
        """Print summary of predictions."""
        print("\n" + "=" * 60)
        print("📊 PREDICTION SUMMARY")
        print("=" * 60)
        
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        # Count predictions
        counts = Counter(predictions)
        
        print(f"\nTotal sequences: {results['num_sequences']}")
        print(f"\nPredicted phase distribution:")
        for label, count in counts.most_common():
            percentage = (count / len(predictions)) * 100
            print(f"   {label}: {count} ({percentage:.1f}%)")
        
        # Show confidence
        if len(probabilities) > 0:
            max_probs = [max(p) if isinstance(p, list) else p[0] for p in probabilities]
            avg_confidence = np.mean(max_probs)
            print(f"\nAverage prediction confidence: {avg_confidence:.2%}")
        
        # Show sequence of predictions
        print(f"\nPrediction sequence (first 20):")
        for i, (label, prob) in enumerate(zip(predictions[:20], probabilities[:20])):
            if isinstance(prob, list):
                conf = max(prob)
            else:
                conf = prob[0] if hasattr(prob, '__len__') else prob
            print(f"   Seq {i+1}: {label} (conf: {conf:.2f})")


def main():
    """Main inference function."""
    
    print("=" * 60)
    print("🔮 LSTM MODEL INFERENCE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Metadata: {METADATA_PATH}")
    print(f"  Pose data: {POSE_DATA_PATH}")
    if SAVE_PREDICTIONS:
        print(f"  Output: {OUTPUT_PATH}")
    print()
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found: {MODEL_PATH}")
        print("\nPlease train a model first using train_lstm_pure_python.py")
        return
    
    if not os.path.exists(METADATA_PATH):
        print(f"❌ ERROR: Metadata file not found: {METADATA_PATH}")
        return
    
    if not os.path.exists(POSE_DATA_PATH):
        print(f"❌ ERROR: Pose data file not found: {POSE_DATA_PATH}")
        return
    
    # Initialize inference engine
    inferencer = LSTMInference(MODEL_PATH, METADATA_PATH)
    
    # Run inference
    results = inferencer.predict_from_file(POSE_DATA_PATH)
    
    # Print summary
    inferencer.print_prediction_summary(results)
    
    # Save results if requested
    if SAVE_PREDICTIONS:
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Predictions saved to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("✅ INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

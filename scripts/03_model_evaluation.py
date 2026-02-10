#!/usr/bin/env python3
"""
Spike-Aware LSTM Prediction Script
===================================

Use a trained LSTM model to predict phases for spike sequences.
Works with models trained using train_lstm_spike_aware.py

Usage: python predict_lstm_spike_aware.py
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ Error: TensorFlow not installed")
    print("   Install with: pip install scipy==1.13.1 ml_dtypes==0.4.0 jax==0.4.23 jaxlib==0.4.23 tensorflow==2.15.0 --break-system-packages")
    exit(1)

from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
MODEL_PATH = "models/model_20260120.keras"
METADATA_PATH = "models/model_20260120_metadata.json"
POSE_DATA_PATH = "data/processed/pose_sequences/frames_with_pose/pose_data_normalized.json"
SPIKE_METADATA_PATH = "data/processed/pose_sequences/frames_downsampled_multi/spike_sequences_metadata.json"
OUTPUT_PATH = "outputs/predictions/spike_predictions.json"
SAVE_PREDICTIONS = True
VERBOSE = True
# ============================================================================


def load_model_and_metadata(model_path: str, metadata_path: str) -> Tuple[keras.Model, Dict]:
    """Load trained model and its metadata."""
    print(f"📂 Loading model...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    model = keras.models.load_model(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Metadata loaded:")
    print(f"   Classes: {metadata['num_classes']}")
    print(f"   Labels: {metadata['label_names']}")
    print(f"   Pose type: {metadata.get('pose_type', 'unknown')}")
    
    return model, metadata


def load_spike_sequences(pose_data_path: str, 
                        spike_metadata_path: str,
                        frames_per_phase: int = 10) -> Tuple[List[Dict], Dict]:
    """Load pose data organized by spike sequences."""
    print(f"\n📂 Loading spike sequence data...")
    
    # Load pose data
    with open(pose_data_path, 'r') as f:
        pose_json = json.load(f)
    
    if 'pose_data' in pose_json:
        pose_data_list = pose_json['pose_data']
    elif isinstance(pose_json, list):
        pose_data_list = pose_json
    else:
        raise ValueError(f"Unexpected pose data format")
    
    print(f"   Loaded {len(pose_data_list)} pose entries")
    
    # Create frame lookup
    pose_by_frame = {entry['frame_number']: entry for entry in pose_data_list}
    
    # Load spike metadata
    with open(spike_metadata_path, 'r') as f:
        spike_metadata = json.load(f)
    
    print(f"   Total spikes: {spike_metadata.get('total_spikes', 0)}")
    
    # Extract phase sequences
    phase_sequences = []
    
    for spike in spike_metadata['spike_sequences']:
        spike_id = spike['spike_id']
        
        for phase in spike['phases']:
            label = phase['label']
            frame_numbers = phase['selected_frame_numbers']
            
            # Get pose data for all frames
            phase_poses = []
            for frame_num in frame_numbers:
                if frame_num in pose_by_frame:
                    pose_entry = pose_by_frame[frame_num]
                    if pose_entry.get('pose_detected', False):
                        phase_poses.append(pose_entry)
            
            if len(phase_poses) == frames_per_phase:
                phase_sequences.append({
                    'spike_id': spike_id,
                    'true_label': label,
                    'poses': phase_poses,
                    'frame_numbers': frame_numbers
                })
    
    print(f"✅ Loaded {len(phase_sequences)} phase sequences for prediction")
    
    return phase_sequences, spike_metadata


def extract_features_from_pose(pose_entry: Dict, pose_type: str = 'mediapipe') -> Optional[np.ndarray]:
    """Extract feature vector from pose data."""
    
    if pose_type == 'mediapipe':
        landmarks = pose_entry.get('landmarks', {})
        angles = pose_entry.get('angles', {})
        
        features = []
        
        # Joint angles (8 features)
        angle_keys = ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                     'left_shoulder', 'left_elbow', 'left_hip', 'left_knee']
        for key in angle_keys:
            features.append(angles.get(key, 0.0))
        
        # Landmark positions (24 features)
        landmark_keys = ['right_shoulder', 'right_elbow', 'right_wrist', 
                        'right_hip', 'right_knee', 'right_ankle',
                        'left_shoulder', 'left_elbow', 'left_wrist',
                        'left_hip', 'left_knee', 'left_ankle']
        for key in landmark_keys:
            if key in landmarks:
                coords = landmarks[key]
                features.extend([coords[0], coords[1]])
            else:
                features.extend([0.0, 0.0])
        
        # Confidence (1 feature)
        features.append(pose_entry.get('confidence', 0.0))
        
        return np.array(features, dtype=np.float32)
    
    elif pose_type == 'yolo':
        keypoints = pose_entry.get('keypoints', {})
        angles = pose_entry.get('angles', {})
        
        features = []
        
        # Joint angles (8 features)
        angle_keys = ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                     'left_shoulder', 'left_elbow', 'left_hip', 'left_knee']
        for key in angle_keys:
            features.append(angles.get(key, 0.0))
        
        # YOLO keypoints
        kpt_keys = ['right_shoulder', 'right_elbow', 'right_wrist',
                   'right_hip', 'right_knee', 'right_ankle',
                   'left_shoulder', 'left_elbow', 'left_wrist',
                   'left_hip', 'left_knee', 'left_ankle']
        
        for key in kpt_keys:
            if key in keypoints:
                kpt = keypoints[key]
                features.extend([kpt['x'], kpt['y'], kpt['confidence']])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # Overall confidence
        avg_conf = np.mean([kpt.get('confidence', 0) for kpt in keypoints.values()])
        features.append(avg_conf)
        
        return np.array(features, dtype=np.float32)
    
    return None


def prepare_sequences_for_prediction(phase_sequences: List[Dict], 
                                     pose_type: str = 'mediapipe') -> np.ndarray:
    """Prepare sequences for model prediction."""
    print(f"\n🔧 Preparing sequences for prediction...")
    
    X_list = []
    
    for seq in phase_sequences:
        # Extract features for all poses in this phase
        phase_features = []
        for pose in seq['poses']:
            features = extract_features_from_pose(pose, pose_type)
            if features is not None:
                phase_features.append(features)
        
        if len(phase_features) > 0:
            X_list.append(np.array(phase_features))
    
    X = np.array(X_list, dtype=np.float32)
    
    print(f"✅ Prepared {len(X)} sequences")
    print(f"   Shape: {X.shape}")
    
    return X


def predict_spike_phases(model: keras.Model, 
                        X: np.ndarray, 
                        label_names: List[str],
                        phase_sequences: List[Dict],
                        verbose: bool = True) -> List[Dict]:
    """Make predictions for spike phases."""
    print(f"\n🔮 Making predictions...")
    
    # Normalize features
    scaler = StandardScaler()
    n_sequences, n_frames, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    X_normalized = scaler.fit_transform(X_reshaped)
    X = X_normalized.reshape(n_sequences, n_frames, n_features)
    
    # Predict
    predictions_proba = model.predict(X, verbose=0)
    predictions = np.argmax(predictions_proba, axis=1)
    
    # Organize results by spike
    results_by_spike = {}
    
    for i, (pred_idx, seq) in enumerate(zip(predictions, phase_sequences)):
        spike_id = seq['spike_id']
        true_label = seq['true_label']
        pred_label = label_names[pred_idx]
        confidence = float(predictions_proba[i][pred_idx])
        
        if spike_id not in results_by_spike:
            results_by_spike[spike_id] = {
                'spike_id': spike_id,
                'phases': []
            }
        
        results_by_spike[spike_id]['phases'].append({
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': confidence,
            'correct': true_label == pred_label,
            'all_probabilities': {
                label: float(predictions_proba[i][j]) 
                for j, label in enumerate(label_names)
            }
        })
    
    # Convert to list
    results = list(results_by_spike.values())
    
    # Print results
    if verbose:
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"{'='*70}")
        
        total_correct = 0
        total_phases = 0
        
        for spike_result in results:
            spike_id = spike_result['spike_id']
            print(f"\nSpike {spike_id}:")
            
            for phase in spike_result['phases']:
                true_label = phase['true_label']
                pred_label = phase['predicted_label']
                confidence = phase['confidence']
                correct = phase['correct']
                
                status = "✅" if correct else "❌"
                print(f"  {status} True: {true_label:8s} | Predicted: {pred_label:8s} | Confidence: {confidence:.1%}")
                
                if correct:
                    total_correct += 1
                total_phases += 1
        
        accuracy = total_correct / total_phases if total_phases > 0 else 0
        print(f"\n{'='*70}")
        print(f"Overall Accuracy: {total_correct}/{total_phases} = {accuracy:.1%}")
        print(f"{'='*70}")
    
    return results


def main():
    """Main prediction function."""
    
    print("=" * 70)
    print("🔮 SPIKE-AWARE LSTM PREDICTION")
    print("=" * 70)
    
    # Load model and metadata
    model, metadata = load_model_and_metadata(MODEL_PATH, METADATA_PATH)
    
    # Load spike sequences
    frames_per_phase = metadata.get('frames_per_phase', 10)
    pose_type = metadata.get('pose_type', 'mediapipe')
    
    phase_sequences, spike_metadata = load_spike_sequences(
        POSE_DATA_PATH,
        SPIKE_METADATA_PATH,
        frames_per_phase
    )
    
    if len(phase_sequences) == 0:
        print("\n❌ ERROR: No phase sequences found!")
        return
    
    # Prepare sequences
    X = prepare_sequences_for_prediction(phase_sequences, pose_type)
    
    # Make predictions
    results = predict_spike_phases(
        model,
        X,
        metadata['label_names'],
        phase_sequences,
        VERBOSE
    )
    
    # Save predictions
    if SAVE_PREDICTIONS:
        output_data = {
            'model_path': MODEL_PATH,
            'pose_data_path': POSE_DATA_PATH,
            'total_spikes': len(results),
            'total_phases': sum(len(s['phases']) for s in results),
            'predictions': results
        }
        
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Predictions saved to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 70)
    print("✅ PREDICTION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
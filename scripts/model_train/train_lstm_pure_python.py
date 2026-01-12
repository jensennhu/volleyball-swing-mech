#!/usr/bin/env python3
"""
Spike-Aware LSTM Training Script
=================================

Train LSTM model on complete volleyball spike sequences.
Each training example is a complete spike (approach→jump→swing→land).

Usage: python train_lstm_spike_aware.py
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ Error: TensorFlow not installed")
    print("   If you need LSTM training, install with:")
    print("   pip install scipy==1.13.1 ml_dtypes==0.4.0 jax==0.4.23 jaxlib==0.4.23 tensorflow==2.15.0 --break-system-packages")
    exit(1)


# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
POSE_DATA_PATH = "frames_with_pose/pose_data_normalized.json"  # Path to pose data JSON
SPIKE_METADATA_PATH = "downsampled_sequences/spike_sequences_metadata.json"  # Spike metadata
POSE_TYPE = "auto"                      # "auto", "mediapipe", or "yolo"
OUTPUT_DIR = "lstm_models"              # Where to save trained models
MODEL_NAME = "spike_phase_classifier_normalized_3layers"   # Model name prefix

# Training mode
TRAINING_MODE = "phase"                 # "phase" or "spike"
                                        # "phase": Classify individual phases (approach/jump/swing/land)
                                        # "spike": Classify complete spikes (not implemented yet)

# Spike filtering
USE_ONLY_COMPLETE_SPIKES = True         # Only train on complete spikes
FRAMES_PER_PHASE = 10                   # Expected frames per phase

# Train/test split
TEST_SIZE = 0.2                         # 20% for testing
VAL_SIZE = 0.2                          # 20% for validation (from remaining data)

# Model architecture
LSTM_UNITS = [128, 64, 32]                  # LSTM layer sizes
DROPOUT = 0.3                           # Dropout rate for regularization

# Training parameters
EPOCHS = 100                             # Number of training epochs
BATCH_SIZE = 8                         # Batch size (smaller for fewer samples)
LEARNING_RATE = 0.001                   # Learning rate

# Display settings
SHOW_PLOTS = True                       # Set to False to skip showing plots
VERBOSE = 1                             # 0=silent, 1=progress bar, 2=one line per epoch
# ============================================================================


def load_spike_sequences(pose_data_path: str, 
                        spike_metadata_path: str,
                        frames_per_phase: int = 10,
                        only_complete: bool = True) -> Tuple[List[Dict], Dict]:
    """
    Load pose data organized by spike sequences.
    
    Returns:
        - List of phase sequences (each is a dict with pose features and label)
        - Spike metadata
    """
    print("\n📂 Loading spike sequence data...")
    
    # Check if files exist
    if not os.path.exists(pose_data_path):
        raise FileNotFoundError(f"Pose data file not found: {pose_data_path}")
    
    if not os.path.exists(spike_metadata_path):
        raise FileNotFoundError(f"Spike metadata file not found: {spike_metadata_path}")
    
    # Load pose data
    with open(pose_data_path, 'r') as f:
        pose_json = json.load(f)
    
    # Handle different pose data formats
    if 'pose_data' in pose_json:
        pose_data_list = pose_json['pose_data']
    elif isinstance(pose_json, list):
        pose_data_list = pose_json
    else:
        raise ValueError(f"Unexpected pose data format. Expected 'pose_data' key or list, got: {type(pose_json)}")
    
    if not pose_data_list:
        raise ValueError("Pose data is empty!")
    
    print(f"   Loaded {len(pose_data_list)} pose entries")
    
    # Create frame number lookup
    pose_by_frame = {}
    for entry in pose_data_list:
        if 'frame_number' in entry:
            pose_by_frame[entry['frame_number']] = entry
        else:
            print(f"   ⚠️  Warning: Pose entry missing frame_number: {entry.keys()}")
    
    print(f"   Created lookup for {len(pose_by_frame)} frames")
    
    # Load spike metadata
    with open(spike_metadata_path, 'r') as f:
        spike_metadata = json.load(f)
    
    # Validate spike metadata structure
    if 'spike_sequences' not in spike_metadata:
        raise ValueError(f"Spike metadata missing 'spike_sequences' key. Keys found: {spike_metadata.keys()}")
    
    spike_sequences_list = spike_metadata['spike_sequences']
    
    if not spike_sequences_list:
        raise ValueError("No spike sequences found in metadata!")
    
    print(f"   Total spikes: {spike_metadata.get('total_spikes', len(spike_sequences_list))}")
    print(f"   Complete spikes: {spike_metadata.get('complete_spikes', 'unknown')}")
    
    # Extract phase sequences
    phase_sequences = []
    
    for spike in spike_sequences_list:
        if only_complete and not spike.get('complete', True):
            if VERBOSE:
                print(f"   ⚠️  Skipping incomplete spike {spike.get('spike_id', '?')}")
            continue
        
        spike_id = spike.get('spike_id', 0)
        phases = spike.get('phases', [])
        
        if not phases:
            print(f"   ⚠️  Warning: Spike {spike_id} has no phases")
            continue
        
        for phase in phases:
            label = phase.get('label', 'unknown')
            frame_numbers = phase.get('selected_frame_numbers', [])
            
            if not frame_numbers:
                print(f"   ⚠️  Warning: Spike {spike_id} phase '{label}' has no frame numbers")
                continue
            
            # Get pose data for all frames in this phase
            phase_poses = []
            missing_frames = []
            
            for frame_num in frame_numbers:
                if frame_num in pose_by_frame:
                    pose_entry = pose_by_frame[frame_num]
                    if pose_entry.get('pose_detected', False):
                        phase_poses.append(pose_entry)
                    else:
                        missing_frames.append(f"{frame_num}(no pose)")
                else:
                    missing_frames.append(f"{frame_num}(missing)")
            
            # Only add if we have the expected number of frames
            if len(phase_poses) == frames_per_phase:
                phase_sequences.append({
                    'spike_id': spike_id,
                    'label': label,
                    'poses': phase_poses,
                    'frame_numbers': frame_numbers
                })
            else:
                print(f"   ⚠️  Skipping spike {spike_id} phase '{label}': "
                      f"expected {frames_per_phase} frames, got {len(phase_poses)}")
                if missing_frames:
                    print(f"       Missing/bad frames: {', '.join(missing_frames[:5])}"
                          f"{' ...' if len(missing_frames) > 5 else ''}")
    
    print(f"\n✅ Loaded {len(phase_sequences)} phase sequences")
    
    # Count by label
    label_counts = {}
    for seq in phase_sequences:
        label = seq['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"   Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"     {label}: {count} sequences")
    
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
        
        # Landmark positions (24 features: x,y for 12 keypoints)
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
        
        # YOLO has 17 keypoints, use subset for consistency
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


def prepare_phase_sequences(phase_sequences: List[Dict], 
                           pose_type: str = 'mediapipe') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare sequences for LSTM training.
    
    Each sequence is all frames from one phase (e.g., 10 approach frames).
    
    Returns:
        X: (n_sequences, frames_per_phase, n_features)
        y: (n_sequences,) - phase labels
        labels: List of unique labels
    """
    print("\n🔧 Preparing sequences for LSTM...")
    
    X_list = []
    y_list = []
    
    for seq in phase_sequences:
        # Extract features for all poses in this phase
        phase_features = []
        for pose in seq['poses']:
            features = extract_features_from_pose(pose, pose_type)
            if features is not None:
                phase_features.append(features)
        
        if len(phase_features) > 0:
            X_list.append(np.array(phase_features))
            y_list.append(seq['label'])
    
    X = np.array(X_list, dtype=np.float32)
    
    # Encode labels
    unique_labels = sorted(list(set(y_list)))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_int[label] for label in y_list])
    
    print(f"✅ Prepared {len(X)} sequences")
    print(f"   Sequence shape: {X.shape}")
    print(f"   Labels: {unique_labels}")
    
    return X, y, unique_labels


def build_phase_classifier(input_shape: Tuple[int, int], 
                          num_classes: int,
                          lstm_units: List[int] = [64, 32],
                          dropout: float = 0.3) -> keras.Model:
    """Build LSTM model for phase classification."""
    
    print(f"\n🏗️  Building phase classifier...")
    print(f"   Input shape: {input_shape}")
    print(f"   Output classes: {num_classes}")
    
    model = models.Sequential(name='SpikePhaseClassifier')
    
    # First LSTM layer (needs input_shape)
    model.add(layers.LSTM(
        lstm_units[0],
        return_sequences=len(lstm_units) > 1,
        input_shape=input_shape,
        name='lstm_1'
    ))
    model.add(layers.Dropout(dropout, name='dropout_1'))
    
    # Additional LSTM layers (don't need input_shape)
    for i, units in enumerate(lstm_units[1:], start=2):
        return_seq = i < len(lstm_units)
        model.add(layers.LSTM(
            units,
            return_sequences=return_seq,
            name=f'lstm_{i}'
        ))
        model.add(layers.Dropout(dropout, name=f'dropout_{i}'))
    
    # Dense layers
    model.add(layers.Dense(32, activation='relu', name='dense'))
    model.add(layers.Dropout(dropout / 2, name='dropout_dense'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    print("\n📋 Model Architecture:")
    model.summary()
    
    return model


def main():
    """Main training function."""
    
    print("=" * 70)
    print("🏐 SPIKE-AWARE LSTM TRAINING")
    print("=" * 70)
    
    # Load spike sequences
    phase_sequences, spike_metadata = load_spike_sequences(
        POSE_DATA_PATH,
        SPIKE_METADATA_PATH,
        FRAMES_PER_PHASE,
        USE_ONLY_COMPLETE_SPIKES
    )
    
    if len(phase_sequences) < 10:
        print(f"\n❌ Error: Not enough sequences for training (found {len(phase_sequences)})")
        print("   Need at least 10 sequences. Check your data!")
        return
    
    # Detect pose type
    first_pose = phase_sequences[0]['poses'][0]
    pose_type = 'yolo' if 'keypoints' in first_pose else 'mediapipe'
    print(f"\n🔍 Detected pose type: {pose_type}")
    
    # Prepare sequences
    X, y, label_names = prepare_phase_sequences(phase_sequences, pose_type)
    
    # Normalize features
    print("\n📊 Normalizing features...")
    scaler = StandardScaler()
    n_sequences, n_frames, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    X_normalized = scaler.fit_transform(X_reshaped)
    X = X_normalized.reshape(n_sequences, n_frames, n_features)
    
    # Split data
    print(f"\n✂️  Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=42, stratify=y_temp
    )
    
    print(f"   Train: {len(X_train)} sequences")
    print(f"   Val:   {len(X_val)} sequences")
    print(f"   Test:  {len(X_test)} sequences")
    
    # Build model
    model = build_phase_classifier(
        input_shape=(n_frames, n_features),
        num_classes=len(label_names),
        lstm_units=LSTM_UNITS,
        dropout=DROPOUT
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
    
    # Train
    print(f"\n🎓 Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=VERBOSE
    )
    
    # Evaluate
    print(f"\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2%}")
    
    # Predictions
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    print(f"\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras")
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'model_name': MODEL_NAME,
        'pose_type': pose_type,
        'frames_per_phase': FRAMES_PER_PHASE,
        'num_classes': len(label_names),
        'label_names': label_names,
        'input_shape': [n_frames, n_features],
        'test_accuracy': float(test_acc),
        'training_mode': TRAINING_MODE
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Plot training history
    if SHOW_PLOTS:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_training.png")
        plt.savefig(plot_path)
        print(f"📊 Training plots saved to: {plot_path}")
        plt.show()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
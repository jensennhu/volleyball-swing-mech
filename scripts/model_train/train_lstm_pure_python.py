#!/usr/bin/env python3
"""
Pure Python LSTM Training Script
=================================

Train LSTM model entirely in Python without command line arguments.
Just run: python train_lstm_pure_python.py
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    print("❌ Error: TensorFlow not installed. Run: pip install tensorflow --break-system-packages")
    exit(1)


# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
POSE_DATA_PATH = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/frames_with_pose/pose_data.json"  # Path to pose data JSON
POSE_TYPE = "auto"                  # "auto", "mediapipe", or "yolo"
OUTPUT_DIR = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/output_lstm"          # Where to save trained models
MODEL_NAME = "spike_lstm"           # Model name prefix

# Sequence parameters
SEQUENCE_LENGTH = 13                # Number of frames per sequence
STRIDE = 1                          # Step between sequences (lower = more overlap)

# Train/test split
TEST_SIZE = 0.2                     # 20% for testing
VAL_SIZE = 0.2                      # 20% for validation (from remaining data)

# Model architecture
LSTM_UNITS = [64, 32]              # LSTM layer sizes
DROPOUT = 0.3                       # Dropout rate for regularization

# Training parameters
EPOCHS = 50                         # Number of training epochs
BATCH_SIZE = 32                     # Batch size
LEARNING_RATE = 0.001              # Learning rate

# Display settings
SHOW_PLOTS = True                   # Set to False to skip showing plots
VERBOSE = 1                         # 0=silent, 1=progress bar, 2=one line per epoch
# ============================================================================


class PoseDataProcessor:
    """Process pose data from MediaPipe or YOLO into LSTM-ready sequences."""
    
    def __init__(self, pose_type: str = 'auto'):
        self.pose_type = pose_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def detect_pose_format(self, pose_data: Dict) -> str:
        """Detect if data is from MediaPipe or YOLO."""
        if not pose_data or len(pose_data) == 0:
            raise ValueError("Empty pose data")
        
        first_entry = pose_data[0]
        
        # Check for YOLO format
        if 'keypoints' in first_entry and isinstance(first_entry['keypoints'], dict):
            first_kpt = next(iter(first_entry['keypoints'].values()))
            if isinstance(first_kpt, dict) and 'confidence' in first_kpt:
                return 'yolo'
        
        # Check for MediaPipe format
        if 'landmarks' in first_entry:
            return 'mediapipe'
        
        raise ValueError("Unknown pose data format")
    
    def extract_features_mediapipe(self, pose_entry: Dict) -> Optional[np.ndarray]:
        """Extract feature vector from MediaPipe pose data."""
        if not pose_entry.get('pose_detected', False):
            return None
        
        landmarks = pose_entry.get('landmarks', {})
        angles = pose_entry.get('angles', {})
        
        features = []
        
        # Joint angles (8 features)
        for key in ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                    'left_shoulder', 'left_elbow', 'left_hip', 'left_knee']:
            features.append(angles.get(key, 0.0))
        
        # Landmark positions (24 features - x,y for 12 keypoints)
        for key in ['right_shoulder', 'right_elbow', 'right_wrist', 
                    'right_hip', 'right_knee', 'right_ankle',
                    'left_shoulder', 'left_elbow', 'left_wrist',
                    'left_hip', 'left_knee', 'left_ankle']:
            coords = landmarks.get(key, [0, 0])
            features.extend(coords)
        
        # Confidence (1 feature)
        features.append(pose_entry.get('confidence', 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def extract_features_yolo(self, pose_entry: Dict) -> Optional[np.ndarray]:
        """Extract feature vector from YOLO pose data."""
        if not pose_entry.get('pose_detected', False):
            return None
        
        keypoints = pose_entry.get('keypoints', {})
        angles = pose_entry.get('angles', {})
        
        features = []
        
        # Joint angles (8 features)
        for key in ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee',
                    'left_shoulder', 'left_elbow', 'left_hip', 'left_knee']:
            features.append(angles.get(key, 0.0))
        
        # Keypoint positions and confidences (36 features - x,y,conf for 12 keypoints)
        for key in ['right_shoulder', 'right_elbow', 'right_wrist', 
                    'right_hip', 'right_knee', 'right_ankle',
                    'left_shoulder', 'left_elbow', 'left_wrist',
                    'left_hip', 'left_knee', 'left_ankle']:
            kpt = keypoints.get(key)
            if kpt:
                features.extend([kpt['x'], kpt['y'], kpt['confidence']])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # Overall confidence (1 feature)
        features.append(pose_entry.get('confidence', 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def load_pose_data(self, json_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load pose data from JSON file and convert to sequences."""
        print(f"📂 Loading pose data from: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        pose_data = data['pose_data']
        
        # Auto-detect format if needed
        if self.pose_type == 'auto':
            self.pose_type = self.detect_pose_format(pose_data)
            print(f"🔍 Detected pose format: {self.pose_type.upper()}")
        
        # Extract features
        print(f"🔄 Extracting features from {len(pose_data)} frames...")
        
        feature_extractor = (self.extract_features_mediapipe 
                           if self.pose_type == 'mediapipe' 
                           else self.extract_features_yolo)
        
        all_features = []
        all_labels = []
        
        for entry in pose_data:
            features = feature_extractor(entry)
            if features is not None:
                all_features.append(features)
                # Use first label if multiple labels exist
                labels = entry.get('labels', ['unknown'])
                all_labels.append(labels[0])
        
        print(f"✅ Extracted {len(all_features)} valid pose frames")
        print(f"📊 Labels found: {set(all_labels)}")
        
        # Convert to numpy arrays
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels)
        
        # Get unique labels
        unique_labels = sorted(set(all_labels))
        
        return X, y, unique_labels
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, 
                        sequence_length: int = 10,
                        stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create overlapping sequences for LSTM input."""
        print(f"\n🔄 Creating sequences (length={sequence_length}, stride={stride})...")
        
        sequences = []
        labels = []
        
        for i in range(0, len(X) - sequence_length + 1, stride):
            sequence = X[i:i + sequence_length]
            # Use the label from the middle of the sequence
            label = y[i + sequence_length // 2]
            
            sequences.append(sequence)
            labels.append(label)
        
        X_seq = np.array(sequences, dtype=np.float32)
        y_seq = np.array(labels)
        
        print(f"✅ Created {len(sequences)} sequences")
        print(f"   Sequence shape: {X_seq.shape}")
        
        return X_seq, y_seq


class VolleyballSpikeLSTM:
    """LSTM model for volleyball spike phase classification."""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, 
                 lstm_units: List[int] = [64, 32],
                 dropout: float = 0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """Build the LSTM architecture."""
        print(f"\n🏗️  Building LSTM model...")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Output classes: {self.num_classes}")
        print(f"   LSTM units: {self.lstm_units}")
        
        model = models.Sequential(name='VolleyballSpikeLSTM')
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=self.input_shape,
            name='lstm_1'
        ))
        model.add(layers.Dropout(self.dropout, name='dropout_1'))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)
            model.add(layers.LSTM(
                units,
                return_sequences=return_seq,
                name=f'lstm_{i}'
            ))
            model.add(layers.Dropout(self.dropout, name=f'dropout_{i}'))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout / 2, name='dropout_dense'))
        
        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
        
        self.model = model
        
        print("\n📋 Model Architecture:")
        model.summary()
        
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model."""
        if self.model is None:
            self.build_model()
        
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        print(f"\n⚙️  Model compiled with learning_rate={learning_rate}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              verbose: int = 1) -> keras.callbacks.History:
        """Train the LSTM model."""
        print(f"\n🎓 Training model...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        
        # Callbacks
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
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        print("\n✅ Training complete!")
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                label_names: List[str]) -> Dict:
        """Evaluate model on test set."""
        print(f"\n📊 Evaluating model on {len(X_test)} test samples...")
        
        # Get predictions
        if self.num_classes == 2:
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Compute metrics
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)[:2]
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=label_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        print(cm)
        print()
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
        
        return results
    
    def plot_training_history(self, save_path: Optional[str] = None, show: bool = True):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, label_names: List[str],
                             save_path: Optional[str] = None, show: bool = True):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(cm, cmap='Blues')
        
        # Labels
        ax.set_xticks(np.arange(len(label_names)))
        ax.set_yticks(np.arange(len(label_names)))
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                text = ax.text(j, i, cm[i, j],
                             ha="center", va="center", 
                             color="black" if cm[i, j] < cm.max() / 2 else "white",
                             fontsize=14, fontweight='bold')
        
        ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_model(self, filepath: str):
        """Save model to file."""
        self.model.save(filepath)
        print(f"💾 Model saved to: {filepath}")


def main():
    """Main training function."""
    
    print("=" * 60)
    print("🏐 VOLLEYBALL SPIKE LSTM CLASSIFIER")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Pose data: {POSE_DATA_PATH}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  LSTM units: {LSTM_UNITS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Output: {OUTPUT_DIR}/")
    print()
    
    # Check if file exists
    if not os.path.exists(POSE_DATA_PATH):
        print(f"❌ ERROR: Pose data file not found: {POSE_DATA_PATH}")
        print("\nPlease ensure you have run pose detection first!")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and process data
    processor = PoseDataProcessor(pose_type=POSE_TYPE)
    X, y, label_names = processor.load_pose_data(POSE_DATA_PATH)
    
    # Create sequences
    X_seq, y_seq = processor.create_sequences(X, y, SEQUENCE_LENGTH, STRIDE)
    
    # Normalize features
    print("\n🔄 Normalizing features...")
    n_samples, seq_len, n_features = X_seq.shape
    X_seq_flat = X_seq.reshape(-1, n_features)
    X_seq_normalized = processor.scaler.fit_transform(X_seq_flat)
    X_seq = X_seq_normalized.reshape(n_samples, seq_len, n_features)
    
    # Encode labels
    print("🔄 Encoding labels...")
    y_encoded = processor.label_encoder.fit_transform(y_seq)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total sequences: {len(X_seq)}")
    print(f"   Sequence shape: {X_seq.shape}")
    print(f"   Number of classes: {len(label_names)}")
    print(f"   Classes: {label_names}")
    print(f"   Class distribution: {dict(zip(*np.unique(y_seq, return_counts=True)))}")
    
    # Split data
    print(f"\n✂️  Splitting data (test={TEST_SIZE}, val={VAL_SIZE})...")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_seq, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
    )
    
    # Second split: train vs val
    val_proportion = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_proportion, random_state=42, stratify=y_temp
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Build and train model
    lstm_model = VolleyballSpikeLSTM(
        input_shape=(seq_len, n_features),
        num_classes=len(label_names),
        lstm_units=LSTM_UNITS,
        dropout=DROPOUT
    )
    
    lstm_model.compile_model(learning_rate=LEARNING_RATE)
    
    history = lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE
    )
    
    # Evaluate
    results = lstm_model.evaluate(X_test, y_test, label_names)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}.keras')
    lstm_model.save_model(model_path)
    
    # Plot and save training history
    plot_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_training.png')
    lstm_model.plot_training_history(save_path=plot_path, show=SHOW_PLOTS)
    
    # Plot and save confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_confusion_matrix.png')
    lstm_model.plot_confusion_matrix(results['confusion_matrix'], label_names, 
                                     save_path=cm_path, show=SHOW_PLOTS)
    
    # Save metadata
    metadata = {
        'pose_type': processor.pose_type,
        'sequence_length': SEQUENCE_LENGTH,
        'stride': STRIDE,
        'num_classes': len(label_names),
        'label_names': label_names,
        'test_accuracy': float(results['test_accuracy']),
        'test_loss': float(results['test_loss']),
        'lstm_units': LSTM_UNITS,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📂 All outputs saved to: {OUTPUT_DIR}/")
    print(f"   Model: {model_path}")
    print(f"   Training plot: {plot_path}")
    print(f"   Confusion matrix: {cm_path}")
    print(f"   Metadata: {metadata_path}")
    print("\n🎯 Final Test Accuracy: {:.2f}%".format(results['test_accuracy'] * 100))


if __name__ == "__main__":
    main()

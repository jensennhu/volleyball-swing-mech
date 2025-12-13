#!/usr/bin/env python3
"""
Volleyball Spike Phase Classification using LSTM
=================================================

Train and evaluate LSTM models on pose data to classify volleyball spike phases.
Supports both MediaPipe and YOLO pose data formats.

Author: AI-Generated
Date: December 2025
Dependencies: tensorflow, numpy, scikit-learn, matplotlib

Installation:
    pip install tensorflow numpy scikit-learn matplotlib --break-system-packages

Usage:
    python lstm_spike_classifier.py --pose_data frames_with_pose/pose_data.json --epochs 50
"""

import numpy as np
import json
import argparse
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


class PoseDataProcessor:
    """Process pose data from MediaPipe or YOLO into LSTM-ready sequences."""
    
    def __init__(self, pose_type: str = 'auto'):
        """
        Initialize processor.
        
        Args:
            pose_type: 'mediapipe', 'yolo', or 'auto' to detect automatically
        """
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
        """
        Extract feature vector from MediaPipe pose data.
        
        Returns:
            Feature vector or None if pose not detected
        """
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
        """
        Extract feature vector from YOLO pose data.
        
        Returns:
            Feature vector or None if pose not detected
        """
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
        """
        Load pose data from JSON file and convert to sequences.
        
        Args:
            json_path: Path to pose_data.json or pose_data_yolo.json
        
        Returns:
            Tuple of (feature_sequences, labels, label_names)
        """
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
        """
        Create overlapping sequences for LSTM input.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            sequence_length: Number of frames per sequence
            stride: Step size between sequences (1 = maximum overlap)
        
        Returns:
            Tuple of (sequences, sequence_labels)
            sequences shape: (n_sequences, sequence_length, n_features)
        """
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
        """
        Initialize LSTM model.
        
        Args:
            input_shape: (sequence_length, n_features)
            num_classes: Number of label classes
            lstm_units: List of LSTM layer sizes
            dropout: Dropout rate for regularization
        """
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
        
        # First LSTM layer (return sequences for stacking)
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
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
        else:
            # Multi-class classification
            model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
        
        self.model = model
        
        print("\n📋 Model Architecture:")
        model.summary()
        
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with optimizer and loss function."""
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
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity mode
        
        Returns:
            Training history
        """
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
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            label_names: List of label names
        
        Returns:
            Dictionary with evaluation metrics
        """
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
    
    def plot_training_history(self, save_path: Optional[str] = None):
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
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, label_names: List[str],
                             save_path: Optional[str] = None):
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
                             ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white",
                             fontsize=14, fontweight='bold')
        
        ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save model to file."""
        self.model.save(filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        print(f"📂 Model loaded from: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM model for volleyball spike phase classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with MediaPipe data
  python lstm_spike_classifier.py --pose_data frames_with_pose/pose_data.json --epochs 50

  # Train with YOLO data
  python lstm_spike_classifier.py --pose_data frames_with_pose_yolo/pose_data_yolo.json --epochs 100

  # Custom sequence length and test split
  python lstm_spike_classifier.py --pose_data pose_data.json --seq_len 15 --test_size 0.3
        """
    )
    
    parser.add_argument('--pose_data', type=str, required=True,
                       help='Path to pose_data.json or pose_data_yolo.json')
    parser.add_argument('--pose_type', type=str, default='auto',
                       choices=['auto', 'mediapipe', 'yolo'],
                       help='Pose data format (default: auto-detect)')
    parser.add_argument('--seq_len', type=int, default=10,
                       help='Sequence length (number of frames) (default: 10)')
    parser.add_argument('--stride', type=int, default=5,
                       help='Stride for sequence creation (default: 5)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set proportion (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lstm_units', type=int, nargs='+', default=[64, 32],
                       help='LSTM layer sizes (default: 64 32)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--output_dir', type=str, default='lstm_models',
                       help='Output directory for models and plots (default: lstm_models)')
    parser.add_argument('--model_name', type=str, default='spike_lstm',
                       help='Model name prefix (default: spike_lstm)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🏐 VOLLEYBALL SPIKE LSTM CLASSIFIER")
    print("=" * 60)
    
    # Load and process data
    processor = PoseDataProcessor(pose_type=args.pose_type)
    X, y, label_names = processor.load_pose_data(args.pose_data)
    
    # Create sequences
    X_seq, y_seq = processor.create_sequences(X, y, args.seq_len, args.stride)
    
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
    print(f"\n✂️  Splitting data (test={args.test_size}, val={args.val_size})...")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_seq, y_encoded, test_size=args.test_size, random_state=42, stratify=y_encoded
    )
    
    # Second split: train vs val
    val_proportion = args.val_size / (1 - args.test_size)
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
        lstm_units=args.lstm_units,
        dropout=args.dropout
    )
    
    lstm_model.compile_model(learning_rate=args.learning_rate)
    
    history = lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate
    results = lstm_model.evaluate(X_test, y_test, label_names)
    
    # Save model
    model_path = os.path.join(args.output_dir, f'{args.model_name}.keras')
    lstm_model.save_model(model_path)
    
    # Plot and save training history
    plot_path = os.path.join(args.output_dir, f'{args.model_name}_training.png')
    lstm_model.plot_training_history(save_path=plot_path)
    
    # Plot and save confusion matrix
    cm_path = os.path.join(args.output_dir, f'{args.model_name}_confusion_matrix.png')
    lstm_model.plot_confusion_matrix(results['confusion_matrix'], label_names, save_path=cm_path)
    
    # Save metadata
    metadata = {
        'pose_type': processor.pose_type,
        'sequence_length': args.seq_len,
        'stride': args.stride,
        'num_classes': len(label_names),
        'label_names': label_names,
        'test_accuracy': float(results['test_accuracy']),
        'test_loss': float(results['test_loss']),
        'lstm_units': args.lstm_units,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate
    }
    
    metadata_path = os.path.join(args.output_dir, f'{args.model_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📂 All outputs saved to: {args.output_dir}/")
    print(f"   Model: {model_path}")
    print(f"   Training plot: {plot_path}")
    print(f"   Confusion matrix: {cm_path}")
    print(f"   Metadata: {metadata_path}")
    print("\n🎯 Final Test Accuracy: {:.2f}%".format(results['test_accuracy'] * 100))


if __name__ == "__main__":
    main()

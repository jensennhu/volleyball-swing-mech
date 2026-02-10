#!/usr/bin/env python3
"""
Train Binary Spike Detector
============================

Train an LSTM model to distinguish spike windows from non-spike windows.

Usage: python 02b_train_spike_detector.py
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
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
    exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "data/processed/spike_detection_data.json"
OUTPUT_DIR = "models"
MODEL_NAME = "spike_detector_binary"

# Model architecture
LSTM_UNITS = [64, 32]  # Simpler than phase classifier
DROPOUT = 0.3
BIDIRECTIONAL = False  # Keep it fast for real-time

# Training parameters
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
CLASS_WEIGHT = {0: 1.0, 1: 2.0}  # Higher weight for spike class

# Display
SHOW_PLOTS = True
VERBOSE = 1
# ============================================================================


def load_spike_detection_data(data_path: str):
    """Load prepared spike detection dataset."""
    print(f"\n📂 Loading spike detection data from: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    
    print(f"\n📊 Dataset Info:")
    print(f"   Window size: {metadata['window_size']} frames")
    print(f"   Features per frame: {metadata['features_per_frame']}")
    print(f"   Total examples: {metadata['total_examples']}")
    print(f"   Positive (spikes): {metadata['positive_examples']}")
    print(f"   Negative (non-spikes): {metadata['negative_examples']}")
    
    # Extract features and labels
    def process_split(split_data):
        X = np.array([ex['features'] for ex in split_data], dtype=np.float32)
        y = np.array([ex['label'] for ex in split_data], dtype=np.int32)
        return X, y
    
    X_train, y_train = process_split(data['train'])
    X_val, y_val = process_split(data['val'])
    X_test, y_test = process_split(data['test'])
    
    print(f"\n📦 Data Shapes:")
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"   Test:  X={X_test.shape}, y={y_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata


def build_spike_detector(input_shape, lstm_units=[64, 32], dropout=0.3, bidirectional=False):
    """Build binary spike detector LSTM model."""
    
    print(f"\n🏗️  Building spike detector...")
    print(f"   Input shape: {input_shape}")
    print(f"   LSTM units: {lstm_units}")
    print(f"   Bidirectional: {bidirectional}")
    
    model = models.Sequential(name='SpikeDetector')
    
    # First LSTM layer
    lstm_layer_1 = layers.LSTM(
        lstm_units[0],
        return_sequences=len(lstm_units) > 1,
        name='lstm_1'
    )
    
    if bidirectional:
        model.add(layers.Bidirectional(lstm_layer_1, input_shape=input_shape, name='bidirectional_1'))
    else:
        model.add(lstm_layer_1)
        model.layers[0]._input_shape = (None,) + input_shape
    
    model.add(layers.Dropout(dropout, name='dropout_1'))
    
    # Additional LSTM layers
    for i, units in enumerate(lstm_units[1:], start=2):
        return_seq = i < len(lstm_units)
        lstm_layer = layers.LSTM(
            units,
            return_sequences=return_seq,
            name=f'lstm_{i}'
        )
        
        if bidirectional:
            model.add(layers.Bidirectional(lstm_layer, name=f'bidirectional_{i}'))
        else:
            model.add(lstm_layer)
        
        model.add(layers.Dropout(dropout, name=f'dropout_{i}'))
    
    # Dense layers
    model.add(layers.Dense(16, activation='relu', name='dense'))
    model.add(layers.Dropout(dropout / 2, name='dropout_dense'))
    
    # Binary output
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    print("\n📋 Model Architecture:")
    model.summary()
    
    return model


def main():
    """Main training function."""
    
    print("=" * 70)
    print("🏐 BINARY SPIKE DETECTOR TRAINING")
    print("=" * 70)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = load_spike_detection_data(DATA_PATH)
    
    # Normalize features
    print(f"\n📊 Normalizing features...")
    scaler = StandardScaler()
    
    n_train, n_frames, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    X_train_norm = scaler.fit_transform(X_train_flat)
    X_train = X_train_norm.reshape(n_train, n_frames, n_features)
    
    n_val = X_val.shape[0]
    X_val_flat = X_val.reshape(-1, n_features)
    X_val_norm = scaler.transform(X_val_flat)
    X_val = X_val_norm.reshape(n_val, n_frames, n_features)
    
    n_test = X_test.shape[0]
    X_test_flat = X_test.reshape(-1, n_features)
    X_test_norm = scaler.transform(X_test_flat)
    X_test = X_test_norm.reshape(n_test, n_frames, n_features)
    
    # Build model
    model = build_spike_detector(
        input_shape=(n_frames, n_features),
        lstm_units=LSTM_UNITS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    print(f"\n⚙️  Model compiled with learning_rate={LEARNING_RATE}")
    
    # Callbacks
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        verbose=1
    )
    
    # Train
    print(f"\n🎓 Training model...")
    print(f"   Using class weights: {CLASS_WEIGHT}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=CLASS_WEIGHT,
        callbacks=[early_stop, reduce_lr],
        verbose=VERBOSE
    )
    
    # Evaluate
    print(f"\n📊 Evaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📈 Test Results:")
    print(f"   Loss: {test_results[0]:.4f}")
    print(f"   Accuracy: {test_results[1]:.2%}")
    print(f"   Precision: {test_results[2]:.2%}")
    print(f"   Recall: {test_results[3]:.2%}")
    print(f"   AUC: {test_results[4]:.4f}")
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['non-spike', 'spike']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Non-Spike  Spike")
    print(f"Actual Non-Spike   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"       Spike       {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_spike = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    print(f"\n📊 Additional Metrics:")
    print(f"   True Positives: {tp}")
    print(f"   True Negatives: {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    print(f"   Specificity: {specificity:.2%}")
    print(f"   F1-Score (spike): {f1_spike:.2%}")
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}.keras')
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save metadata
    model_metadata = {
        'model_name': MODEL_NAME,
        'model_type': 'binary_spike_detector',
        'input_shape': [n_frames, n_features],
        'window_size': metadata['window_size'],
        'lstm_units': LSTM_UNITS,
        'dropout': DROPOUT,
        'bidirectional': BIDIRECTIONAL,
        'test_metrics': {
            'accuracy': float(test_results[1]),
            'precision': float(test_results[2]),
            'recall': float(test_results[3]),
            'auc': float(test_results[4]),
            'specificity': float(specificity),
            'f1_score': float(f1_spike)
        },
        'confusion_matrix': cm.tolist(),
        'training_params': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'class_weight': CLASS_WEIGHT
        }
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"💾 Metadata saved to: {metadata_path}")
    
    # Plot training history
    if SHOW_PLOTS:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision & Recall
        axes[1, 0].plot(history.history['precision'], label='Precision', linewidth=2)
        axes[1, 0].plot(history.history['recall'], label='Recall', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_training.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training plots saved to: {plot_path}")
        plt.show()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel Performance Summary:")
    print(f"  Accuracy:   {test_results[1]:.1%}")
    print(f"  Precision:  {test_results[2]:.1%} (low false positives)")
    print(f"  Recall:     {test_results[3]:.1%} (catches most spikes)")
    print(f"  F1-Score:   {f1_spike:.1%}")
    print(f"\nFiles saved:")
    print(f"  Model: {model_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"\nNext step:")
    print(f"  Use this model in two-stage pipeline for spike detection!")


if __name__ == "__main__":
    main()

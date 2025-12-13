#!/usr/bin/env python3
"""
Standalone script to train LSTM model for spike phase classification.
Configure the parameters below and run directly.
"""

import os
import sys

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
# ============================================================================


def main():
    """Run LSTM training with configured parameters."""
    
    print("=" * 60)
    print("🏐 LSTM SPIKE CLASSIFIER - TRAINING")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Pose data: {POSE_DATA_PATH}")
    print(f"  Pose type: {POSE_TYPE}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Stride: {STRIDE}")
    print(f"  Test size: {TEST_SIZE}")
    print(f"  Val size: {VAL_SIZE}")
    print(f"  LSTM units: {LSTM_UNITS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()
    
    # Check if pose data exists
    if not os.path.exists(POSE_DATA_PATH):
        print(f"❌ ERROR: Pose data file not found: {POSE_DATA_PATH}")
        print("\nPlease make sure you have:")
        print("  1. Extracted frames from video")
        print("  2. Run pose detection (MediaPipe or YOLO)")
        print("  3. Generated pose_data.json or pose_data_yolo.json")
        return
    
    # Build command
    cmd_parts = [
        "python",
        "lstm_spike_classifier.py",
        "--pose_data", POSE_DATA_PATH,
        "--pose_type", POSE_TYPE,
        "--seq_len", str(SEQUENCE_LENGTH),
        "--stride", str(STRIDE),
        "--test_size", str(TEST_SIZE),
        "--val_size", str(VAL_SIZE),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--dropout", str(DROPOUT),
        "--learning_rate", str(LEARNING_RATE),
        "--output_dir", OUTPUT_DIR,
        "--model_name", MODEL_NAME,
        "--lstm_units"
    ]
    cmd_parts.extend([str(u) for u in LSTM_UNITS])
    
    cmd = " ".join(cmd_parts)
    
    print(f"🚀 Running training command...")
    print(f"\n{cmd}\n")
    
    # Execute
    os.system(cmd)


if __name__ == "__main__":
    main()

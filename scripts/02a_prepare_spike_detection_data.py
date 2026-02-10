#!/usr/bin/env python3
"""
Prepare Spike Detection Training Data
======================================

Extract positive (spike) and negative (non-spike) examples for training
a binary spike detector.

Usage: python 02a_prepare_spike_detection_data.py
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple
import random

# ============================================================================
# CONFIGURATION
# ============================================================================
POSE_DATA_PATH = "data/processed/pose_sequences/frames_with_pose/pose_data_normalized.json"
SPIKE_METADATA_PATH = "data/processed/pose_sequences/frames_downsampled/spike_sequences_metadata.json"
OUTPUT_PATH = "data/processed/spike_detection_data.json"

# Sampling parameters
WINDOW_SIZE = 40  # Complete spike = 4 phases × 10 frames
NEGATIVE_RATIO = 2  # Sample 2× more negatives than positives
MIN_GAP_SIZE = 50  # Minimum gap size to sample from
FRAMES_PER_PHASE = 10

# Train/val/test split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VERBOSE = True
# ============================================================================


def load_data():
    """Load pose data and spike metadata."""
    print("📂 Loading data...")
    
    # Load pose data
    with open(POSE_DATA_PATH, 'r') as f:
        pose_json = json.load(f)
    
    pose_data = pose_json.get('pose_data', pose_json)
    print(f"   Loaded {len(pose_data)} pose entries")
    
    # Create frame lookup
    pose_by_frame = {entry['frame_number']: entry for entry in pose_data}
    
    # Load spike metadata
    with open(SPIKE_METADATA_PATH, 'r') as f:
        spike_metadata = json.load(f)
    
    spike_sequences = spike_metadata.get('spike_sequences', [])
    print(f"   Loaded {len(spike_sequences)} spike sequences")
    
    return pose_by_frame, spike_sequences


def extract_positive_examples(pose_by_frame: Dict, 
                              spike_sequences: List[Dict]) -> List[Dict]:
    """
    Extract positive examples (complete spikes) from annotated data.
    
    Each positive example is a 40-frame window containing a complete spike:
    approach (10) + jump (10) + swing (10) + land (10)
    """
    print("\n🟢 Extracting positive examples (spikes)...")
    
    positive_examples = []
    
    for spike in spike_sequences:
        spike_id = spike.get('spike_id', 0)
        phases = spike.get('phases', [])
        
        # Only use complete spikes
        if not spike.get('complete', False):
            if VERBOSE:
                print(f"   ⚠️  Skipping incomplete spike {spike_id}")
            continue
        
        # Collect all frame numbers for this spike (in order)
        spike_frames = []
        
        for phase in phases:
            frame_numbers = phase.get('selected_frame_numbers', [])
            
            if len(frame_numbers) != FRAMES_PER_PHASE:
                if VERBOSE:
                    print(f"   ⚠️  Skipping spike {spike_id}: phase has {len(frame_numbers)} frames (expected {FRAMES_PER_PHASE})")
                break
            
            spike_frames.extend(frame_numbers)
        
        # Should have exactly 40 frames (4 phases × 10 frames)
        if len(spike_frames) != WINDOW_SIZE:
            if VERBOSE:
                print(f"   ⚠️  Skipping spike {spike_id}: total {len(spike_frames)} frames (expected {WINDOW_SIZE})")
            continue
        
        # Extract features for all frames
        features_sequence = []
        all_frames_valid = True
        
        for frame_num in spike_frames:
            if frame_num not in pose_by_frame:
                if VERBOSE:
                    print(f"   ⚠️  Skipping spike {spike_id}: missing frame {frame_num}")
                all_frames_valid = False
                break
            
            pose_entry = pose_by_frame[frame_num]
            
            if not pose_entry.get('pose_detected', False):
                if VERBOSE:
                    print(f"   ⚠️  Skipping spike {spike_id}: no pose at frame {frame_num}")
                all_frames_valid = False
                break
            
            # Get normalized features
            features = pose_entry.get('normalized_features')
            if features is None:
                if VERBOSE:
                    print(f"   ⚠️  Skipping spike {spike_id}: no features at frame {frame_num}")
                all_frames_valid = False
                break
            
            features_sequence.append(features)
        
        if all_frames_valid and len(features_sequence) == WINDOW_SIZE:
            positive_examples.append({
                'features': features_sequence,  # Shape: (40, 33)
                'label': 1,  # Spike
                'spike_id': spike_id,
                'frame_start': spike_frames[0],
                'frame_end': spike_frames[-1],
                'type': 'spike'
            })
            
            if VERBOSE:
                print(f"   ✓ Added spike {spike_id}: frames {spike_frames[0]}-{spike_frames[-1]}")
    
    print(f"\n✅ Extracted {len(positive_examples)} positive examples")
    return positive_examples


def find_gaps(spike_sequences: List[Dict], max_frame: int) -> List[Tuple[int, int]]:
    """Find gaps between spikes where we can sample negative examples."""
    
    # Collect all spike frame ranges
    spike_ranges = []
    
    for spike in spike_sequences:
        phases = spike.get('phases', [])
        if not phases:
            continue
        
        # Find min and max frame numbers
        all_frames = []
        for phase in phases:
            all_frames.extend(phase.get('selected_frame_numbers', []))
        
        if all_frames:
            spike_ranges.append((min(all_frames), max(all_frames)))
    
    # Sort by start frame
    spike_ranges.sort()
    
    # Find gaps
    gaps = []
    
    # Gap before first spike
    if spike_ranges and spike_ranges[0][0] > MIN_GAP_SIZE:
        gaps.append((0, spike_ranges[0][0]))
    
    # Gaps between spikes
    for i in range(len(spike_ranges) - 1):
        gap_start = spike_ranges[i][1]
        gap_end = spike_ranges[i + 1][0]
        
        if gap_end - gap_start >= MIN_GAP_SIZE:
            gaps.append((gap_start, gap_end))
    
    # Gap after last spike
    if spike_ranges and max_frame - spike_ranges[-1][1] > MIN_GAP_SIZE:
        gaps.append((spike_ranges[-1][1], max_frame))
    
    return gaps


def extract_negative_examples(pose_by_frame: Dict,
                              spike_sequences: List[Dict],
                              num_positive: int) -> List[Dict]:
    """
    Extract negative examples (non-spike windows) from video gaps.
    """
    print("\n🔴 Extracting negative examples (non-spikes)...")
    
    # Find max frame number
    max_frame = max(pose_by_frame.keys())
    
    # Find gaps between spikes
    gaps = find_gaps(spike_sequences, max_frame)
    print(f"   Found {len(gaps)} gaps between spikes")
    
    if VERBOSE:
        for gap_start, gap_end in gaps:
            print(f"     Gap: frames {gap_start}-{gap_end} ({gap_end - gap_start} frames)")
    
    # Calculate how many negatives to sample
    num_negatives_target = num_positive * NEGATIVE_RATIO
    print(f"   Target: {num_negatives_target} negative examples")
    
    negative_examples = []
    attempts = 0
    max_attempts = num_negatives_target * 10  # Avoid infinite loop
    
    while len(negative_examples) < num_negatives_target and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a gap
        if not gaps:
            print("   ⚠️  No gaps available")
            break
        
        gap_start, gap_end = random.choice(gaps)
        
        # Check if gap is large enough
        if gap_end - gap_start < WINDOW_SIZE:
            continue
        
        # Sample random starting point in this gap
        window_start = random.randint(gap_start, gap_end - WINDOW_SIZE)
        window_frames = list(range(window_start, window_start + WINDOW_SIZE))
        
        # Extract features for all frames
        features_sequence = []
        all_frames_valid = True
        
        for frame_num in window_frames:
            if frame_num not in pose_by_frame:
                all_frames_valid = False
                break
            
            pose_entry = pose_by_frame[frame_num]
            
            if not pose_entry.get('pose_detected', False):
                all_frames_valid = False
                break
            
            features = pose_entry.get('normalized_features')
            if features is None:
                all_frames_valid = False
                break
            
            features_sequence.append(features)
        
        if all_frames_valid and len(features_sequence) == WINDOW_SIZE:
            negative_examples.append({
                'features': features_sequence,  # Shape: (40, 33)
                'label': 0,  # Non-spike
                'frame_start': window_frames[0],
                'frame_end': window_frames[-1],
                'type': 'non-spike',
                'gap_source': (gap_start, gap_end)
            })
            
            if VERBOSE and len(negative_examples) % 10 == 0:
                print(f"   Sampled {len(negative_examples)}/{num_negatives_target} negatives...")
    
    print(f"\n✅ Extracted {len(negative_examples)} negative examples")
    return negative_examples


def split_dataset(examples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test sets."""
    
    # Shuffle
    random.shuffle(examples)
    
    n = len(examples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]
    
    return train, val, test


def main():
    """Main function."""
    
    print("=" * 70)
    print("🏐 SPIKE DETECTION DATA PREPARATION")
    print("=" * 70)
    
    # Load data
    pose_by_frame, spike_sequences = load_data()
    
    # Extract positive examples (spikes)
    positive_examples = extract_positive_examples(pose_by_frame, spike_sequences)
    
    if len(positive_examples) == 0:
        print("\n❌ ERROR: No positive examples found!")
        print("   Check your spike metadata and pose data.")
        return
    
    # Extract negative examples (non-spikes)
    negative_examples = extract_negative_examples(
        pose_by_frame,
        spike_sequences,
        len(positive_examples)
    )
    
    if len(negative_examples) == 0:
        print("\n❌ ERROR: No negative examples found!")
        print("   Check if there are gaps between spikes in your video.")
        return
    
    # Combine examples
    all_examples = positive_examples + negative_examples
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total examples: {len(all_examples)}")
    print(f"   Positive (spikes): {len(positive_examples)} ({len(positive_examples)/len(all_examples)*100:.1f}%)")
    print(f"   Negative (non-spikes): {len(negative_examples)} ({len(negative_examples)/len(all_examples)*100:.1f}%)")
    print(f"   Window size: {WINDOW_SIZE} frames")
    print(f"   Features per frame: 33")
    
    # Split into train/val/test
    print(f"\n✂️  Splitting dataset...")
    train, val, test = split_dataset(all_examples)
    
    print(f"   Train: {len(train)} examples ({len(train)/len(all_examples)*100:.1f}%)")
    print(f"   Val:   {len(val)} examples ({len(val)/len(all_examples)*100:.1f}%)")
    print(f"   Test:  {len(test)} examples ({len(test)/len(all_examples)*100:.1f}%)")
    
    # Count labels in each split
    for split_name, split_data in [('Train', train), ('Val', val), ('Test', test)]:
        pos = sum(1 for ex in split_data if ex['label'] == 1)
        neg = sum(1 for ex in split_data if ex['label'] == 0)
        print(f"     {split_name}: {pos} spikes, {neg} non-spikes")
    
    # Save dataset
    output_data = {
        'train': train,
        'val': val,
        'test': test,
        'metadata': {
            'window_size': WINDOW_SIZE,
            'features_per_frame': 33,
            'total_examples': len(all_examples),
            'positive_examples': len(positive_examples),
            'negative_examples': len(negative_examples),
            'negative_ratio': NEGATIVE_RATIO,
            'splits': {
                'train': len(train),
                'val': len(val),
                'test': len(test)
            }
        }
    }
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Dataset saved to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 70)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"\nNext step:")
    print(f"  python scripts/02b_train_spike_detector.py")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()

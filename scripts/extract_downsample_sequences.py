#!/usr/bin/env python3
"""
Extract and Downsample Action Sequences
========================================

Extracts frames from video and downsamples each continuous action sequence
to a fixed number of frames (default: 10).

Keeps first and last frames of each action sequence, then selects evenly-spaced
frames in between. This creates consistent sequence lengths for LSTM training.

Example: If "approach" has 173 frames in one continuous sequence, this will
create one 10-frame sequence with uniform temporal spacing.
"""

import xml.etree.ElementTree as ET
import cv2
import os
import shutil
from typing import Set, Dict, Tuple, List
import numpy as np
from collections import defaultdict

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
VIDEO_PATH = "/Users/jensenhu/Documents/GitHub/volley-vision-vids/hitting-session.mp4"
ANNOTATIONS_PATH = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/annotations/annotations.xml"
OUTPUT_DIR = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/downsampled_frames"
LABELS_TO_EXTRACT = ["all"]              # Options: "all", "approach", "jump", "swing", "land"
FILENAME_PREFIX = "frame"                # Prefix for output files

# Cropping options
CROP_TO_BBOX = True                      # Crop to bounding box
PADDING_PIXELS = 50                      # Extra pixels around bbox

# Downsampling options
FRAMES_PER_SEQUENCE = 10                 # Target frames per action sequence
KEEP_FIRST_LAST = True                   # Always keep first and last frames
SPACING_METHOD = "uniform"               # "uniform" or "random"
MIN_SEQUENCE_LENGTH = 5                  # Minimum frames needed in original sequence

# Display options
VERBOSE = True                           # Show progress
SAVE_METADATA = True                     # Save JSON with sequence info
# ============================================================================


def parse_annotations_by_track(xml_path: str) -> Dict[int, Dict]:
    """
    Parse CVAT annotations and group frames by track (continuous action sequence).
    
    Args:
        xml_path: Path to annotations.xml
    
    Returns:
        Dictionary mapping track_id -> track info with frames
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    tracks = {}
    
    for track in root.findall('track'):
        track_id = int(track.get('id'))
        label = track.get('label')
        
        frames_info = []
        
        for box in track.findall('box'):
            frame_num = int(box.get('frame'))
            outside = int(box.get('outside', 0))
            
            # Skip frames marked as "outside"
            if outside == 1:
                continue
            
            # Get bounding box
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            frames_info.append({
                'frame_number': frame_num,
                'bbox': (xtl, ytl, xbr, ybr)
            })
        
        if frames_info:
            # Sort by frame number
            frames_info.sort(key=lambda x: x['frame_number'])
            
            tracks[track_id] = {
                'label': label,
                'frames': frames_info,
                'start_frame': frames_info[0]['frame_number'],
                'end_frame': frames_info[-1]['frame_number'],
                'total_frames': len(frames_info)
            }
    
    return tracks


def select_representative_frames(frames_info: List[Dict], 
                                 target_count: int,
                                 keep_first_last: bool = True,
                                 method: str = "uniform") -> List[Dict]:
    """
    Select representative frames from a sequence.
    
    Args:
        frames_info: List of frame info dicts (sorted by frame_number)
        target_count: Number of frames to select
        keep_first_last: If True, always keep first and last frames
        method: "uniform" for evenly-spaced, "random" for random selection
    
    Returns:
        List of selected frame info dicts
    """
    if len(frames_info) <= target_count:
        # Already have target or fewer frames, keep all
        return frames_info
    
    selected = []
    
    if keep_first_last:
        # Keep first and last
        selected.append(frames_info[0])
        selected.append(frames_info[-1])
        
        # Select middle frames
        middle_count = target_count - 2
        middle_frames = frames_info[1:-1]
        
        if middle_count > 0 and len(middle_frames) > 0:
            if method == "uniform":
                # Evenly spaced indices
                indices = np.linspace(0, len(middle_frames) - 1, middle_count, dtype=int)
                # Remove duplicates and sort
                indices = sorted(set(indices))
                selected_middle = [middle_frames[i] for i in indices]
            else:  # random
                # Random selection
                indices = np.random.choice(len(middle_frames), 
                                         min(middle_count, len(middle_frames)), 
                                         replace=False)
                indices = sorted(indices)
                selected_middle = [middle_frames[i] for i in indices]
            
            # Insert middle frames between first and last
            selected = [selected[0]] + selected_middle + [selected[-1]]
        
        # Sort by frame number to maintain temporal order
        selected.sort(key=lambda x: x['frame_number'])
        
    else:
        # Don't require first/last
        if method == "uniform":
            indices = np.linspace(0, len(frames_info) - 1, target_count, dtype=int)
            indices = sorted(set(indices))
            selected = [frames_info[i] for i in indices]
        else:  # random
            indices = np.random.choice(len(frames_info), target_count, replace=False)
            indices = sorted(indices)
            selected = [frames_info[i] for i in indices]
    
    return selected


def crop_frame_to_bbox(frame, bbox: Tuple[float, float, float, float], padding: int = 0):
    """Crop frame to bounding box with optional padding."""
    frame_height, frame_width = frame.shape[:2]
    xtl, ytl, xbr, ybr = bbox
    
    x1 = max(0, int(xtl) - padding)
    y1 = max(0, int(ytl) - padding)
    x2 = min(frame_width, int(xbr) + padding)
    y2 = min(frame_height, int(ybr) + padding)
    
    return frame[y1:y2, x1:x2]


def extract_and_downsample_sequences(
    video_path: str,
    tracks: Dict[int, Dict],
    labels_to_extract: List[str],
    output_dir: str,
    frames_per_sequence: int = 10,
    keep_first_last: bool = True,
    spacing_method: str = "uniform",
    min_sequence_length: int = 5,
    crop: bool = True,
    padding: int = 0,
    prefix: str = "frame",
    verbose: bool = True
) -> Dict:
    """
    Extract frames from video and downsample each action sequence.
    
    Args:
        video_path: Path to video file
        tracks: Dictionary of tracks from parse_annotations_by_track()
        labels_to_extract: List of labels to process
        output_dir: Output directory
        frames_per_sequence: Target number of frames per sequence
        keep_first_last: Keep first and last frames
        spacing_method: "uniform" or "random"
        min_sequence_length: Skip sequences shorter than this
        crop: Crop to bounding box
        padding: Padding around bbox
        prefix: Filename prefix
        verbose: Print progress
    
    Returns:
        Dictionary with processing statistics and sequence info
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if verbose:
        print(f"Video info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Cropping: {'ENABLED' if crop else 'DISABLED'}")
        if crop:
            print(f"  Padding: {padding}px")
        print()
    
    # Filter tracks by label
    filtered_tracks = {}
    for track_id, track_info in tracks.items():
        label = track_info['label']
        if 'all' in labels_to_extract or label in labels_to_extract:
            if track_info['total_frames'] >= min_sequence_length:
                filtered_tracks[track_id] = track_info
            elif verbose:
                print(f"⚠️  Skipping track {track_id} ({label}): only {track_info['total_frames']} frames (min: {min_sequence_length})")
    
    if verbose:
        print(f"\n📊 Processing {len(filtered_tracks)} action sequences:")
        for track_id, track_info in filtered_tracks.items():
            print(f"  Track {track_id} ({track_info['label']}): {track_info['total_frames']} frames → {min(frames_per_sequence, track_info['total_frames'])} frames")
        print()
    
    # Process each track
    sequence_metadata = []
    total_extracted = 0
    
    for track_idx, (track_id, track_info) in enumerate(filtered_tracks.items(), 1):
        label = track_info['label']
        frames_info = track_info['frames']
        
        if verbose:
            print(f"[{track_idx}/{len(filtered_tracks)}] Processing track {track_id} ({label})...")
        
        # Select representative frames
        selected_frames = select_representative_frames(
            frames_info,
            frames_per_sequence,
            keep_first_last,
            spacing_method
        )
        
        if verbose:
            print(f"  Selected {len(selected_frames)} frames from {len(frames_info)} total")
            print(f"  Frame range: {selected_frames[0]['frame_number']} → {selected_frames[-1]['frame_number']}")
        
        # Extract selected frames
        extracted_files = []
        
        for frame_info in selected_frames:
            frame_num = frame_info['frame_number']
            bbox = frame_info['bbox']
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                if verbose:
                    print(f"  ⚠️  Warning: Could not read frame {frame_num}")
                continue
            
            # Crop if requested
            if crop:
                frame = crop_frame_to_bbox(frame, bbox, padding)
            
            # Save frame with track info in filename
            filename = f"{prefix}_track{track_id:02d}_{label}_f{frame_num:06d}.png"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)
            
            extracted_files.append({
                'filename': filename,
                'frame_number': frame_num,
                'bbox': bbox
            })
            total_extracted += 1
        
        # Store metadata for this sequence
        sequence_metadata.append({
            'track_id': track_id,
            'label': label,
            'original_frames': len(frames_info),
            'downsampled_frames': len(selected_frames),
            'start_frame': frames_info[0]['frame_number'],
            'end_frame': frames_info[-1]['frame_number'],
            'selected_frame_numbers': [f['frame_number'] for f in selected_frames],
            'files': extracted_files
        })
        
        if verbose:
            print(f"  ✓ Extracted {len(extracted_files)} frames")
    
    cap.release()
    
    # Compile statistics
    stats = {
        'total_sequences': len(filtered_tracks),
        'total_frames_extracted': total_extracted,
        'frames_per_sequence': frames_per_sequence,
        'spacing_method': spacing_method,
        'keep_first_last': keep_first_last,
        'video_info': {
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames
        },
        'sequences': sequence_metadata
    }
    
    if verbose:
        print(f"\n" + "=" * 60)
        print("📊 EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total sequences processed: {len(filtered_tracks)}")
        print(f"Total frames extracted: {total_extracted}")
        print(f"Frames per sequence: {frames_per_sequence}")
        print(f"\nSequences by label:")
        label_counts = defaultdict(int)
        for seq in sequence_metadata:
            label_counts[seq['label']] += 1
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} sequences")
    
    return stats


def main():
    """Main function."""
    
    print("=" * 60)
    print("🎬 EXTRACT AND DOWNSAMPLE ACTION SEQUENCES")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Video: {os.path.basename(VIDEO_PATH)}")
    print(f"  Annotations: {os.path.basename(ANNOTATIONS_PATH)}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Labels: {', '.join(LABELS_TO_EXTRACT)}")
    print(f"  Frames per sequence: {FRAMES_PER_SEQUENCE}")
    print(f"  Spacing method: {SPACING_METHOD}")
    print(f"  Keep first/last: {KEEP_FIRST_LAST}")
    print(f"  Min sequence length: {MIN_SEQUENCE_LENGTH}")
    print(f"  Crop to bbox: {CROP_TO_BBOX}")
    if CROP_TO_BBOX:
        print(f"  Padding: {PADDING_PIXELS}px")
    print()
    
    # Check files exist
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ ERROR: Video file not found: {VIDEO_PATH}")
        return
    
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ ERROR: Annotations file not found: {ANNOTATIONS_PATH}")
        return
    
    # Parse annotations by track
    if VERBOSE:
        print("Step 1: Parsing annotations by track...")
    
    tracks = parse_annotations_by_track(ANNOTATIONS_PATH)
    
    if VERBOSE:
        print(f"✓ Found {len(tracks)} action sequences (tracks)")
        print(f"\nSequence breakdown:")
        label_stats = defaultdict(lambda: {'count': 0, 'total_frames': 0})
        for track_id, track_info in tracks.items():
            label = track_info['label']
            label_stats[label]['count'] += 1
            label_stats[label]['total_frames'] += track_info['total_frames']
        
        for label, stats in sorted(label_stats.items()):
            avg_frames = stats['total_frames'] / stats['count']
            print(f"  {label}: {stats['count']} sequences, avg {avg_frames:.1f} frames/sequence")
        print()
    
    # Extract and downsample
    if VERBOSE:
        print("Step 2: Extracting and downsampling sequences...")
        print()
    
    stats = extract_and_downsample_sequences(
        video_path=VIDEO_PATH,
        tracks=tracks,
        labels_to_extract=LABELS_TO_EXTRACT,
        output_dir=OUTPUT_DIR,
        frames_per_sequence=FRAMES_PER_SEQUENCE,
        keep_first_last=KEEP_FIRST_LAST,
        spacing_method=SPACING_METHOD,
        min_sequence_length=MIN_SEQUENCE_LENGTH,
        crop=CROP_TO_BBOX,
        padding=PADDING_PIXELS,
        prefix=FILENAME_PREFIX,
        verbose=VERBOSE
    )
    
    # Save metadata
    if SAVE_METADATA:
        import json
        metadata_path = os.path.join(OUTPUT_DIR, 'sequence_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        if VERBOSE:
            print(f"\n💾 Metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Total frames extracted: {stats['total_frames_extracted']}")
    print(f"Ready for pose detection and LSTM training!")


if __name__ == "__main__":
    main()

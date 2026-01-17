#!/usr/bin/env python3
"""
Extract and Downsample Complete Spike Sequences
================================================

Recognizes complete volleyball spike sequences (approach→jump→swing→land)
and downsamples each phase to a fixed number of frames while maintaining
the temporal order.

This creates LSTM-ready sequences where each spike is represented by:
- 10 approach frames
- 10 jump frames  
- 10 swing frames
- 10 land frames
= 40 total frames per complete spike sequence

Author: AI-Generated
Date: December 2025
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2
import os
from typing import Set, Dict, Tuple, List
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
VIDEO_PATH = "data/raw/videos/recorded_videos/hitting-session.mp4"
ANNOTATIONS_PATH = "data/raw/annotations/annotations.xml"
OUTPUT_DIR = "data/processed/pose_sequences/frames_downsampled"
FILENAME_PREFIX = "frame"

# Cropping options
CROP_TO_BBOX = True
PADDING_PIXELS = 50

# Downsampling options
FRAMES_PER_PHASE = 10                    # Frames per phase (approach, jump, swing, land)
KEEP_FIRST_LAST = True                   # Always keep first and last frames
SPACING_METHOD = "uniform"               # "uniform" or "random"

# Spike sequence detection
EXPECTED_PHASE_ORDER = ["approach", "jump", "swing", "land"]
MAX_GAP_BETWEEN_PHASES = 100            # Maximum frame gap to consider phases part of same spike

# Display options
VERBOSE = True
SAVE_METADATA = True
CREATE_DISTILLED_ANNOTATIONS = True
# ============================================================================


def parse_annotations_by_track(xml_path: str) -> Dict[int, Dict]:
    """Parse CVAT annotations and group frames by track."""
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
            
            if outside == 1:
                continue
            
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            frames_info.append({
                'frame_number': frame_num,
                'bbox': (xtl, ytl, xbr, ybr)
            })
        
        if frames_info:
            frames_info.sort(key=lambda x: x['frame_number'])
            
            tracks[track_id] = {
                'label': label,
                'frames': frames_info,
                'start_frame': frames_info[0]['frame_number'],
                'end_frame': frames_info[-1]['frame_number'],
                'total_frames': len(frames_info)
            }
    
    return tracks


def group_tracks_into_spikes(tracks: Dict[int, Dict], 
                             expected_order: List[str],
                             max_gap: int,
                             verbose: bool = True) -> List[Dict]:
    """
    Group consecutive tracks into complete spike sequences.
    
    Args:
        tracks: Dictionary of tracks by track_id
        expected_order: Expected phase order (e.g., ["approach", "jump", "swing", "land"])
        max_gap: Maximum frame gap between phases
        verbose: Print grouping info
    
    Returns:
        List of spike sequence dictionaries
    """
    # Sort tracks by start frame
    sorted_tracks = sorted(tracks.items(), key=lambda x: x[1]['start_frame'])
    
    spikes = []
    current_spike = []
    expected_phase_idx = 0
    
    for track_id, track_info in sorted_tracks:
        label = track_info['label']
        
        # Check if this is the next expected phase
        if expected_phase_idx < len(expected_order):
            expected_label = expected_order[expected_phase_idx]
            
            if label == expected_label:
                # Check frame gap if we have previous phases
                if current_spike:
                    prev_track = current_spike[-1]
                    gap = track_info['start_frame'] - prev_track['end_frame']
                    
                    if gap > max_gap:
                        # Gap too large, start new spike
                        if verbose:
                            print(f"  ⚠️  Large gap ({gap} frames) between track {prev_track['track_id']} and {track_id}")
                            print(f"      Starting new spike sequence...")
                        
                        # Save incomplete spike if it has at least 2 phases
                        if len(current_spike) >= 2:
                            spikes.append({
                                'spike_id': len(spikes),
                                'phases': current_spike,
                                'complete': False
                            })
                        
                        # Start new spike
                        current_spike = []
                        expected_phase_idx = 0
                        
                        # Re-check if this track starts a new spike
                        if label == expected_order[0]:
                            current_spike.append({
                                'track_id': track_id,
                                **track_info
                            })
                            expected_phase_idx = 1
                    else:
                        # Normal continuation
                        current_spike.append({
                            'track_id': track_id,
                            **track_info
                        })
                        expected_phase_idx += 1
                else:
                    # First phase of new spike
                    current_spike.append({
                        'track_id': track_id,
                        **track_info
                    })
                    expected_phase_idx += 1
                
                # Check if spike is complete
                if expected_phase_idx == len(expected_order):
                    spikes.append({
                        'spike_id': len(spikes),
                        'phases': current_spike,
                        'complete': True
                    })
                    current_spike = []
                    expected_phase_idx = 0
    
    # Add any remaining incomplete spike
    if current_spike and len(current_spike) >= 2:
        spikes.append({
            'spike_id': len(spikes),
            'phases': current_spike,
            'complete': False
        })
    
    if verbose:
        print(f"\n📊 Found {len(spikes)} spike sequences:")
        for spike in spikes:
            phase_labels = [p['label'] for p in spike['phases']]
            status = "✓ Complete" if spike['complete'] else "⚠️  Incomplete"
            print(f"   Spike {spike['spike_id']}: {' → '.join(phase_labels)} {status}")
    
    return spikes


def select_representative_frames(frames_info: List[Dict], 
                                 target_count: int,
                                 keep_first_last: bool = True,
                                 method: str = "uniform") -> List[Dict]:
    """Select representative frames from a sequence."""
    if len(frames_info) <= target_count:
        return frames_info
    
    if keep_first_last:
        first_frame = frames_info[0]
        last_frame = frames_info[-1]
        
        middle_count = target_count - 2
        middle_frames = frames_info[1:-1]
        
        if middle_count > 0 and len(middle_frames) > 0:
            if method == "uniform":
                if len(middle_frames) >= middle_count:
                    indices = np.linspace(0, len(middle_frames) - 1, middle_count)
                    indices = np.unique(np.round(indices).astype(int))
                    
                    while len(indices) < middle_count and len(indices) < len(middle_frames):
                        gaps = np.diff(indices)
                        if len(gaps) > 0:
                            max_gap_idx = np.argmax(gaps)
                            new_idx = indices[max_gap_idx] + gaps[max_gap_idx] // 2
                            if new_idx not in indices and new_idx < len(middle_frames):
                                indices = np.sort(np.append(indices, new_idx))
                            else:
                                break
                        else:
                            break
                    
                    selected_middle = [middle_frames[i] for i in indices[:middle_count]]
                else:
                    selected_middle = middle_frames
            else:  # random
                actual_count = min(middle_count, len(middle_frames))
                indices = np.random.choice(len(middle_frames), actual_count, replace=False)
                indices = sorted(indices)
                selected_middle = [middle_frames[i] for i in indices]
            
            selected = [first_frame] + selected_middle + [last_frame]
        else:
            selected = [first_frame, last_frame]
        
        selected.sort(key=lambda x: x['frame_number'])
    else:
        if method == "uniform":
            indices = np.linspace(0, len(frames_info) - 1, target_count, dtype=int)
            indices = sorted(set(indices))
            selected = [frames_info[i] for i in indices]
        else:
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


def extract_spike_sequences(
    video_path: str,
    spikes: List[Dict],
    output_dir: str,
    frames_per_phase: int = 10,
    keep_first_last: bool = True,
    spacing_method: str = "uniform",
    crop: bool = True,
    padding: int = 0,
    prefix: str = "frame",
    verbose: bool = True
) -> Dict:
    """Extract and downsample complete spike sequences."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if verbose:
        print(f"\nVideo info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Cropping: {'ENABLED' if crop else 'DISABLED'}")
        if crop:
            print(f"  Padding: {padding}px")
        print()
    
    spike_metadata = []
    total_extracted = 0
    
    for spike in spikes:
        spike_id = spike['spike_id']
        phases = spike['phases']
        
        if verbose:
            print(f"Processing Spike {spike_id}...")
        
        spike_data = {
            'spike_id': spike_id,
            'complete': spike['complete'],
            'phases': []
        }
        
        for phase in phases:
            track_id = phase['track_id']
            label = phase['label']
            frames_info = phase['frames']
            
            if verbose:
                print(f"  Phase: {label} (track {track_id}): {len(frames_info)} frames → {min(frames_per_phase, len(frames_info))} frames")
            
            # Select representative frames
            selected_frames = select_representative_frames(
                frames_info,
                frames_per_phase,
                keep_first_last,
                spacing_method
            )
            
            # Extract frames
            extracted_files = []
            
            for frame_info in selected_frames:
                frame_num = frame_info['frame_number']
                bbox = frame_info['bbox']
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    if verbose:
                        print(f"    ⚠️  Warning: Could not read frame {frame_num}")
                    continue
                
                if crop:
                    frame = crop_frame_to_bbox(frame, bbox, padding)
                
                # Filename: frame_spike00_phase00_approach_f000992.png
                filename = f"{prefix}_spike{spike_id:02d}_phase{len(spike_data['phases']):02d}_{label}_f{frame_num:06d}.png"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, frame)
                
                extracted_files.append({
                    'filename': filename,
                    'frame_number': frame_num,
                    'bbox': bbox
                })
                total_extracted += 1
            
            spike_data['phases'].append({
                'track_id': track_id,
                'label': label,
                'original_frames': len(frames_info),
                'downsampled_frames': len(selected_frames),
                'start_frame': frames_info[0]['frame_number'],
                'end_frame': frames_info[-1]['frame_number'],
                'selected_frame_numbers': [f['frame_number'] for f in selected_frames],
                'files': extracted_files
            })
        
        spike_metadata.append(spike_data)
        
        if verbose:
            total_in_spike = sum(p['downsampled_frames'] for p in spike_data['phases'])
            print(f"  ✓ Spike {spike_id}: {total_in_spike} frames extracted\n")
    
    cap.release()
    
    stats = {
        'total_spikes': len(spikes),
        'complete_spikes': sum(1 for s in spikes if s['complete']),
        'total_frames_extracted': total_extracted,
        'frames_per_phase': frames_per_phase,
        'spacing_method': spacing_method,
        'keep_first_last': keep_first_last,
        'video_info': {
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames
        },
        'spike_sequences': spike_metadata
    }
    
    return stats


def main():
    """Main function."""
    
    print("=" * 60)
    print("🎬 EXTRACT AND DOWNSAMPLE SPIKE SEQUENCES")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Video: {os.path.basename(VIDEO_PATH)}")
    print(f"  Annotations: {os.path.basename(ANNOTATIONS_PATH)}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Frames per phase: {FRAMES_PER_PHASE}")
    print(f"  Expected sequence: {' → '.join(EXPECTED_PHASE_ORDER)}")
    print()
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ ERROR: Video file not found: {VIDEO_PATH}")
        return
    
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ ERROR: Annotations file not found: {ANNOTATIONS_PATH}")
        return
    
    # Parse annotations
    if VERBOSE:
        print("Step 1: Parsing annotations by track...")
    
    tracks = parse_annotations_by_track(ANNOTATIONS_PATH)
    
    if VERBOSE:
        print(f"✓ Found {len(tracks)} tracks\n")
    
    # Group into spike sequences
    if VERBOSE:
        print("Step 2: Grouping tracks into spike sequences...")
    
    spikes = group_tracks_into_spikes(
        tracks,
        EXPECTED_PHASE_ORDER,
        MAX_GAP_BETWEEN_PHASES,
        VERBOSE
    )
    
    if not spikes:
        print("❌ ERROR: No spike sequences found!")
        return
    
    # Extract and downsample
    if VERBOSE:
        print("\nStep 3: Extracting and downsampling spike sequences...")
    
    stats = extract_spike_sequences(
        video_path=VIDEO_PATH,
        spikes=spikes,
        output_dir=OUTPUT_DIR,
        frames_per_phase=FRAMES_PER_PHASE,
        keep_first_last=KEEP_FIRST_LAST,
        spacing_method=SPACING_METHOD,
        crop=CROP_TO_BBOX,
        padding=PADDING_PIXELS,
        prefix=FILENAME_PREFIX,
        verbose=VERBOSE
    )
    
    # Save metadata
    if SAVE_METADATA:
        metadata_path = os.path.join(OUTPUT_DIR, 'spike_sequences_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        if VERBOSE:
            print(f"\n💾 Metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Total spike sequences: {stats['total_spikes']}")
    print(f"Complete spikes: {stats['complete_spikes']}")
    print(f"Total frames extracted: {stats['total_frames_extracted']}")
    print(f"\nEach complete spike has {FRAMES_PER_PHASE * len(EXPECTED_PHASE_ORDER)} frames")
    print(f"({FRAMES_PER_PHASE} frames × {len(EXPECTED_PHASE_ORDER)} phases)")


if __name__ == "__main__":
    main()
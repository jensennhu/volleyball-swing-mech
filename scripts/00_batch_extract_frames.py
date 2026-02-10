#!/usr/bin/env python3
"""
Batch Extract and Downsample Spike Sequences from Multiple Videos
==================================================================

Processes multiple video+annotation pairs and consolidates them into
a unified dataset for LSTM training.

Author: AI-Generated
Date: January 2025
"""

import xml.etree.ElementTree as ET
import cv2
import os
import sys
from typing import Set, Dict, Tuple, List
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
from pathlib import Path

# Note: The helper functions are duplicated here instead of imported
# to keep the script self-contained. If you prefer to avoid duplication,
# you can refactor 00_extract_frames.py to create importable functions.


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

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================

# List of video+annotation pairs
# Each entry: {"video": "path/to/video.mp4", "annotations": "path/to/annotations.xml", "name": "session1"}
VIDEO_CONFIGS = [
    {
        "video": "data/raw/videos/recorded_videos/hitting-session.mp4",
        "annotations": "data/raw/annotations/annotations.xml",
        "name": "hitting-session"
    },
    # Add more video configs here:
    {
        "video": "data/raw/videos/recorded_videos/usa-hitting-lines.mp4",
        "annotations": "data/raw/annotations/usa-hitting.xml",
        "name": "usa-hitting"
    },
]

# Global output directory (all videos will be consolidated here)
OUTPUT_DIR = "data/processed/pose_sequences/frames_downsampled_multi"
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
# ============================================================================


class MultiVideoProcessor:
    """Process multiple videos and consolidate into unified dataset."""
    
    def __init__(self, output_dir: str, verbose: bool = True):
        self.output_dir = output_dir
        self.verbose = verbose
        self.global_spike_id = 0  # Track spikes across all videos
        self.all_metadata = {
            'videos': [],
            'total_spikes': 0,
            'total_complete_spikes': 0,
            'total_frames_extracted': 0,
            'frames_per_phase': FRAMES_PER_PHASE,
            'spacing_method': SPACING_METHOD,
            'keep_first_last': KEEP_FIRST_LAST,
            'expected_phase_order': EXPECTED_PHASE_ORDER,
            'spike_sequences': []
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def process_single_video(
        self,
        video_path: str,
        annotations_path: str,
        video_name: str
    ) -> Dict:
        """Process a single video and return metadata."""
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"🎬 PROCESSING: {video_name}")
            print("=" * 70)
            print(f"Video: {video_path}")
            print(f"Annotations: {annotations_path}")
        
        # Validate files exist
        if not os.path.exists(video_path):
            print(f"❌ ERROR: Video file not found: {video_path}")
            return None
        
        if not os.path.exists(annotations_path):
            print(f"❌ ERROR: Annotations file not found: {annotations_path}")
            return None
        
        # Parse annotations
        if self.verbose:
            print("\n📋 Parsing annotations by track...")
        
        tracks = parse_annotations_by_track(annotations_path)
        
        if self.verbose:
            print(f"✓ Found {len(tracks)} tracks")
        
        # Group into spike sequences
        if self.verbose:
            print("\n🔄 Grouping tracks into spike sequences...")
        
        spikes = group_tracks_into_spikes(
            tracks,
            EXPECTED_PHASE_ORDER,
            MAX_GAP_BETWEEN_PHASES,
            self.verbose
        )
        
        if not spikes:
            print(f"⚠️  No spike sequences found in {video_name}")
            return None
        
        # Extract and downsample
        if self.verbose:
            print("\n🎞️  Extracting and downsampling frames...")
        
        video_metadata = self.extract_spike_sequences(
            video_path=video_path,
            spikes=spikes,
            video_name=video_name
        )
        
        return video_metadata
    
    def extract_spike_sequences(
        self,
        video_path: str,
        spikes: List[Dict],
        video_name: str
    ) -> Dict:
        """Extract and downsample spike sequences from one video."""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.verbose:
            print(f"\nVideo info:")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps:.2f}")
            print(f"  Total frames: {total_frames}")
        
        video_metadata = {
            'video_name': video_name,
            'video_path': video_path,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames,
            'spike_count': len(spikes),
            'complete_spike_count': sum(1 for s in spikes if s['complete']),
            'frames_extracted': 0,
            'spikes': []
        }
        
        for spike in spikes:
            # Use global spike ID (unique across all videos)
            global_spike_id = self.global_spike_id
            self.global_spike_id += 1
            
            phases = spike['phases']
            
            if self.verbose:
                status = "✓ Complete" if spike['complete'] else "⚠️  Incomplete"
                phase_labels = [p['label'] for p in phases]
                print(f"\nSpike {global_spike_id} ({video_name}): {' → '.join(phase_labels)} {status}")
            
            spike_data = {
                'spike_id': global_spike_id,
                'video_name': video_name,
                'complete': spike['complete'],
                'phases': []
            }
            
            for phase in phases:
                track_id = phase['track_id']
                label = phase['label']
                frames_info = phase['frames']
                
                # Select representative frames
                selected_frames = select_representative_frames(
                    frames_info,
                    FRAMES_PER_PHASE,
                    KEEP_FIRST_LAST,
                    SPACING_METHOD
                )
                
                if self.verbose:
                    print(f"  Phase: {label} (track {track_id}): {len(frames_info)} → {len(selected_frames)} frames")
                
                # Extract frames
                extracted_files = []
                
                for frame_info in selected_frames:
                    frame_num = frame_info['frame_number']
                    bbox = frame_info['bbox']
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if not ret:
                        if self.verbose:
                            print(f"    ⚠️  Warning: Could not read frame {frame_num}")
                        continue
                    
                    if CROP_TO_BBOX:
                        frame = crop_frame_to_bbox(frame, bbox, PADDING_PIXELS)
                    
                    # Filename includes video name for traceability
                    # Format: frame_video-hitting-session_spike00_phase00_approach_f000992.png
                    filename = (f"{FILENAME_PREFIX}_video-{video_name}_"
                              f"spike{global_spike_id:03d}_"
                              f"phase{len(spike_data['phases']):02d}_{label}_"
                              f"f{frame_num:06d}.png")
                    
                    output_path = os.path.join(self.output_dir, filename)
                    cv2.imwrite(output_path, frame)
                    
                    extracted_files.append({
                        'filename': filename,
                        'frame_number': frame_num,
                        'bbox': bbox,
                        'video_name': video_name
                    })
                    
                    video_metadata['frames_extracted'] += 1
                    self.all_metadata['total_frames_extracted'] += 1
                
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
            
            video_metadata['spikes'].append(spike_data)
            self.all_metadata['spike_sequences'].append(spike_data)
            
            if spike['complete']:
                self.all_metadata['total_complete_spikes'] += 1
            self.all_metadata['total_spikes'] += 1
        
        cap.release()
        
        return video_metadata
    
    def process_all_videos(self, video_configs: List[Dict]):
        """Process all videos in the configuration."""
        
        if self.verbose:
            print("=" * 70)
            print("🎬 BATCH PROCESSING MULTIPLE VIDEOS")
            print("=" * 70)
            print(f"\nTotal videos to process: {len(video_configs)}")
            print(f"Output directory: {self.output_dir}")
            print()
        
        for i, config in enumerate(video_configs, 1):
            video_path = config['video']
            annotations_path = config['annotations']
            video_name = config['name']
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Video {i}/{len(video_configs)}")
                print(f"{'='*70}")
            
            video_metadata = self.process_single_video(
                video_path,
                annotations_path,
                video_name
            )
            
            if video_metadata:
                self.all_metadata['videos'].append(video_metadata)
            else:
                print(f"⚠️  Skipped {video_name} due to errors")
        
        # Save consolidated metadata
        if SAVE_METADATA:
            self.save_metadata()
        
        # Print final summary
        self.print_summary()
    
    def save_metadata(self):
        """Save consolidated metadata to JSON."""
        metadata_path = os.path.join(self.output_dir, 'spike_sequences_metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(self.all_metadata, f, indent=2)
        
        if self.verbose:
            print(f"\n💾 Consolidated metadata saved to: {metadata_path}")
    
    def print_summary(self):
        """Print final processing summary."""
        print("\n" + "=" * 70)
        print("✅ BATCH PROCESSING COMPLETE!")
        print("=" * 70)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Videos processed: {len(self.all_metadata['videos'])}")
        print(f"  Total spike sequences: {self.all_metadata['total_spikes']}")
        print(f"  Complete spikes: {self.all_metadata['total_complete_spikes']}")
        print(f"  Total frames extracted: {self.all_metadata['total_frames_extracted']}")
        
        print(f"\n📁 Output directory: {self.output_dir}")
        
        print(f"\n🎥 Per-Video Breakdown:")
        for video in self.all_metadata['videos']:
            print(f"\n  {video['video_name']}:")
            print(f"    Spikes: {video['spike_count']} ({video['complete_spike_count']} complete)")
            print(f"    Frames: {video['frames_extracted']}")
        
        # Phase distribution
        phase_counts = defaultdict(int)
        for spike in self.all_metadata['spike_sequences']:
            for phase in spike['phases']:
                phase_counts[phase['label']] += 1
        
        print(f"\n📈 Phase Distribution (across all videos):")
        for label, count in sorted(phase_counts.items()):
            print(f"  {label}: {count} sequences")
        
        print(f"\n💡 Next Steps:")
        print(f"  1. Run pose detection: python scripts/01_pose_detection.py")
        print(f"     (Update FRAMES_DIR to: {self.output_dir})")
        print(f"  2. Train LSTM model: python scripts/02_model_training.py")
        print(f"     (Your model will now be trained on {len(self.all_metadata['videos'])} videos!)")


def validate_configs(configs: List[Dict]) -> bool:
    """Validate all video configurations."""
    valid = True
    
    for i, config in enumerate(configs):
        if 'video' not in config:
            print(f"❌ Config {i}: Missing 'video' key")
            valid = False
        
        if 'annotations' not in config:
            print(f"❌ Config {i}: Missing 'annotations' key")
            valid = False
        
        if 'name' not in config:
            print(f"❌ Config {i}: Missing 'name' key")
            valid = False
        else:
            # Check for duplicate names
            names = [c['name'] for c in configs]
            if names.count(config['name']) > 1:
                print(f"❌ Config {i}: Duplicate name '{config['name']}'")
                valid = False
    
    return valid


def main():
    """Main batch processing function."""
    
    # Validate configurations
    if not validate_configs(VIDEO_CONFIGS):
        print("\n❌ Configuration validation failed!")
        print("Please fix the issues above and try again.")
        return
    
    # Create processor
    processor = MultiVideoProcessor(OUTPUT_DIR, VERBOSE)
    
    # Process all videos
    processor.process_all_videos(VIDEO_CONFIGS)


if __name__ == "__main__":
    main()

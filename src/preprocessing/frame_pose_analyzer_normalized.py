#!/usr/bin/env python3
"""
Frame Pose Analyzer with Normalized Features
=============================================

Enhanced version that extracts NORMALIZED pose features:
- Root normalization: Centered at pelvis
- Scale normalization: Scaled by torso length
- Frame-size independent

This fixes the issue where models trained on one video don't work on others!

Usage:
    python frame_pose_analyzer_normalized.py <frames_dir> <annotations.xml>
"""

import cv2
import mediapipe as mp
import numpy as np
import xml.etree.ElementTree as ET
import os
import glob
import re
from typing import Dict, List, Tuple, Optional
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalized_pose_features import NormalizedPoseExtractor

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def parse_frame_filename(filename: str) -> Optional[int]:
    """Extract frame number from filename."""
    # Try new format first: frame_track00_approach_f000992.png
    match = re.search(r'_f(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Fall back to old format: frame_000992.png
    match = re.search(r'_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None


def load_annotations(xml_path: str) -> Dict[int, List[str]]:
    """Parse CVAT XML annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    frame_labels = {}
    
    for track in root.findall('track'):
        label = track.get('label')
        
        for box in track.findall('box'):
            frame_num = int(box.get('frame'))
            
            if frame_num not in frame_labels:
                frame_labels[frame_num] = []
            
            if label not in frame_labels[frame_num]:
                frame_labels[frame_num].append(label)
    
    return frame_labels


def process_frame_with_normalized_pose(
    frame: np.ndarray,
    frame_number: int,
    labels: List[str],
    pose_detector,
    feature_extractor: NormalizedPoseExtractor
) -> Tuple[np.ndarray, Dict]:
    """
    Process frame with NORMALIZED pose features.
    
    Key difference from old version:
    - Landmarks are normalized (pelvis-centered, torso-scaled)
    - Features are resolution-independent
    - Works across different videos/cameras
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)
    
    pose_data = {
        'frame_number': frame_number,
        'labels': labels,
        'pose_detected': False,
        'confidence': 0.0,
        'normalized_features': None,  # ← NEW: Normalized features
        'landmarks': {},  # Still store for visualization
        'angles': {}
    }
    
    if not results.pose_landmarks:
        # No pose detected
        y_pos = 30
        for label in labels:
            cv2.putText(frame, f"Label: {label}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30
        
        cv2.putText(frame, "NO POSE DETECTED", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, pose_data
    
    # Pose detected - extract NORMALIZED features
    landmarks = results.pose_landmarks
    pose_data['pose_detected'] = True
    
    # Extract normalized features (pelvis-centered, scale-normalized)
    features, metadata = feature_extractor.extract_features_with_metadata(landmarks)
    
    if features is not None:
        pose_data['normalized_features'] = features.tolist()
        pose_data['angles'] = metadata['angles']
        pose_data['confidence'] = metadata['confidence']
        
        # Also store normalized landmarks for reference
        pose_data['normalized_landmarks'] = {
            k: list(v) for k, v in metadata['normalized_landmarks'].items()
        }
    
    # Draw skeleton
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )
    
    # Add text annotations
    y_pos = 30
    for label in labels:
        cv2.putText(frame, f"Label: {label}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
    
    cv2.putText(frame, f"Frame: {frame_number}", (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30
    
    cv2.putText(frame, f"Confidence: {pose_data['confidence']:.2f}", (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30
    
    # Show some key angles
    if pose_data['angles']:
        cv2.putText(frame, "Joint Angles:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += 25
        
        for key in ['right_shoulder', 'right_elbow', 'right_knee']:
            angle = pose_data['angles'].get(key, 0)
            cv2.putText(frame, f"{key}: {angle:.1f}°", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_pos += 20
    
    # Add normalization indicator
    cv2.putText(frame, "✓ NORMALIZED", (frame_width - 200, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame, pose_data


def analyze_extracted_frames(
    frames_dir: str,
    annotations_path: str,
    output_dir: str = 'frames_with_pose',
    save_data: bool = True,
    verbose: bool = True
) -> Dict:
    """Process all extracted frames with NORMALIZED pose features."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("🏐 VOLLEYBALL FRAME POSE ANALYZER (NORMALIZED)")
        print("=" * 60)
        print(f"\n📂 Frames directory: {frames_dir}")
        print(f"📄 Annotations: {annotations_path}")
        print(f"💾 Output directory: {output_dir}")
        print(f"\n✨ Using NORMALIZED features:")
        print(f"   - Root: Centered at pelvis")
        print(f"   - Scale: Normalized by torso length")
        print(f"   - Resolution: Independent of frame size\n")
    
    # Load annotations
    if verbose:
        print("📋 Loading annotations...")
    frame_labels = load_annotations(annotations_path)
    if verbose:
        print(f"   Found {len(frame_labels)} annotated frames\n")
    
    # Find all frame files
    if verbose:
        print("🔍 Finding frame files...")
    frame_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        frame_files.extend(glob.glob(os.path.join(frames_dir, ext)))
    
    if not frame_files:
        raise ValueError(f"No frame files found in {frames_dir}")
    
    if verbose:
        print(f"   Found {len(frame_files)} frame files\n")
    
    # Parse frame numbers
    frame_info = []
    for filepath in frame_files:
        filename = os.path.basename(filepath)
        frame_num = parse_frame_filename(filename)
        if frame_num is not None:
            frame_info.append({
                'path': filepath,
                'filename': filename,
                'frame_number': frame_num,
                'labels': frame_labels.get(frame_num, ['unknown'])
            })
    
    frame_info.sort(key=lambda x: x['frame_number'])
    
    if verbose:
        print(f"📊 Processing {len(frame_info)} frames with NORMALIZED pose detection...\n")
    
    # Initialize feature extractor
    feature_extractor = NormalizedPoseExtractor()
    
    # Process frames
    results = {
        'processed_frames': [],
        'pose_data': [],
        'normalization_method': 'pelvis_centered_torso_scaled',
        'summary': {
            'total_frames': len(frame_info),
            'pose_detected': 0,
            'no_pose': 0,
            'avg_confidence': 0.0,
            'labels_distribution': {}
        }
    }
    
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        for i, info in enumerate(frame_info):
            if verbose and (i + 1) % 10 == 0:
                print(f"   Processing frame {i + 1}/{len(frame_info)}...")
            
            # Read frame
            frame = cv2.imread(info['path'])
            if frame is None:
                print(f"Warning: Could not read {info['filename']}")
                continue
            
            # Process with NORMALIZED pose detection
            annotated_frame, pose_data = process_frame_with_normalized_pose(
                frame,
                info['frame_number'],
                info['labels'],
                pose,
                feature_extractor
            )
            
            # Save annotated frame
            output_filename = f"pose_{info['filename']}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, annotated_frame)
            
            # Store results
            results['processed_frames'].append(info['frame_number'])
            results['pose_data'].append(pose_data)
            
            # Update summary
            if pose_data['pose_detected']:
                results['summary']['pose_detected'] += 1
            else:
                results['summary']['no_pose'] += 1
            
            for label in info['labels']:
                results['summary']['labels_distribution'][label] = \
                    results['summary']['labels_distribution'].get(label, 0) + 1
    
    # Calculate average confidence
    confidences = [pd['confidence'] for pd in results['pose_data'] if pd['pose_detected']]
    if confidences:
        results['summary']['avg_confidence'] = np.mean(confidences)
    
    # Save pose data as JSON
    if save_data:
        json_path = os.path.join(output_dir, 'pose_data_normalized.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert all results
        serializable_results = convert_to_serializable(results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        if verbose:
            print(f"\n💾 Normalized pose data saved to: {json_path}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total frames processed: {results['summary']['total_frames']}")
        print(f"Pose detected: {results['summary']['pose_detected']}")
        print(f"No pose: {results['summary']['no_pose']}")
        print(f"Average confidence: {results['summary']['avg_confidence']:.2%}")
        print(f"\n✨ Features are NORMALIZED:")
        print(f"   - Position-invariant (centered at pelvis)")
        print(f"   - Scale-invariant (normalized by torso)")
        print(f"   - Resolution-invariant (works on any video size)")
        print(f"\nLabel distribution:")
        for label, count in results['summary']['labels_distribution'].items():
            print(f"   {label}: {count} frames")
        print("=" * 60)
        print(f"\n✅ Annotated frames saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add NORMALIZED pose detection to volleyball frames"
    )
    parser.add_argument('frames_dir', help='Directory containing extracted frames')
    parser.add_argument('annotations', help='Path to CVAT annotations XML file')
    parser.add_argument('-o', '--output', default='frames_with_pose_normalized',
                       help='Output directory (default: frames_with_pose_normalized)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    results = analyze_extracted_frames(
        frames_dir=args.frames_dir,
        annotations_path=args.annotations,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    print(f"\n🎉 Complete! Processed {len(results['processed_frames'])} frames with NORMALIZED features")
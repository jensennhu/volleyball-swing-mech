#!/usr/bin/env python3
"""
Volleyball Frame Pose Analyzer - YOLO Version
==============================================

Processes extracted frames from CVAT annotations and adds skeletal pose detection using YOLO.
Uses YOLOv8-pose model for fast and accurate pose estimation.

Author: AI-Generated
Date: December 2025
Dependencies: ultralytics, opencv-python, numpy

Installation:
    pip install ultralytics opencv-python numpy --break-system-packages

Usage:
    from frame_pose_analyzer_yolo import analyze_extracted_frames_yolo
    
    results = analyze_extracted_frames_yolo(
        frames_dir='extracted_frames',
        annotations_path='annotations.xml',
        output_dir='frames_with_pose_yolo'
    )
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import glob
import re
from typing import Dict, List, Tuple, Optional
import json
from ultralytics import YOLO

# YOLO Pose keypoint indices (COCO format)
YOLO_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}


def parse_frame_filename(filename: str) -> Optional[int]:
    """
    Extract frame number from filename.
    
    Args:
        filename: Name of the frame file
    
    Returns:
        int: Frame number, or None if not found
    """
    match = re.search(r'_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def load_annotations(xml_path: str) -> Dict[int, List[str]]:
    """
    Parse CVAT XML annotations and map frame numbers to labels.
    
    Args:
        xml_path: Path to annotations.xml file
    
    Returns:
        Dict mapping frame_number -> list of labels for that frame
    """
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


def calculate_angle(a: Tuple[float, float], 
                    b: Tuple[float, float], 
                    c: Tuple[float, float]) -> float:
    """
    Calculate the angle at point b formed by three points a, b, c.
    
    Args:
        a: First point (x, y)
        b: Middle point (vertex of angle) (x, y)
        c: Third point (x, y)
    
    Returns:
        float: Angle in degrees (0-180)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def extract_keypoints(results, confidence_threshold: float = 0.5) -> Optional[Dict]:
    """
    Extract keypoints from YOLO pose detection results.
    
    Args:
        results: YOLO detection results
        confidence_threshold: Minimum confidence to accept keypoint
    
    Returns:
        Dict with keypoints or None if no person detected
    """
    if len(results) == 0 or results[0].keypoints is None:
        return None
    
    # Get the first person detected (highest confidence)
    keypoints_data = results[0].keypoints.data
    
    if len(keypoints_data) == 0:
        return None
    
    # Take the first detection (most confident person)
    kpts = keypoints_data[0].cpu().numpy()  # Shape: (17, 3) - [x, y, confidence]
    
    keypoints = {}
    for name, idx in YOLO_KEYPOINTS.items():
        x, y, conf = kpts[idx]
        if conf > confidence_threshold:
            keypoints[name] = {
                'x': float(x),
                'y': float(y),
                'confidence': float(conf)
            }
        else:
            keypoints[name] = None
    
    return keypoints


def draw_skeleton_yolo(frame: np.ndarray, keypoints: Dict) -> np.ndarray:
    """
    Draw skeleton connections on frame using YOLO keypoints.
    
    Args:
        frame: Input frame
        keypoints: Dictionary of keypoints from extract_keypoints()
    
    Returns:
        Annotated frame
    """
    # Define skeleton connections (COCO format)
    skeleton = [
        # Face
        ('nose', 'left_eye'),
        ('nose', 'right_eye'),
        ('left_eye', 'left_ear'),
        ('right_eye', 'right_ear'),
        # Arms
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        # Torso
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        # Legs
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle')
    ]
    
    # Draw connections
    for part1, part2 in skeleton:
        if keypoints.get(part1) and keypoints.get(part2):
            pt1 = (int(keypoints[part1]['x']), int(keypoints[part1]['y']))
            pt2 = (int(keypoints[part2]['x']), int(keypoints[part2]['y']))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for name, kpt in keypoints.items():
        if kpt is not None:
            center = (int(kpt['x']), int(kpt['y']))
            # Color based on confidence
            conf = kpt['confidence']
            color = (0, int(255 * conf), int(255 * (1 - conf)))  # Green to red based on confidence
            cv2.circle(frame, center, 4, color, -1)
            cv2.circle(frame, center, 4, (255, 255, 255), 1)
    
    return frame


def process_frame_with_yolo_pose(frame: np.ndarray, 
                                  frame_number: int,
                                  labels: List[str],
                                  model: YOLO) -> Tuple[np.ndarray, Dict]:
    """
    Process a single frame: detect pose using YOLO and annotate.
    
    Args:
        frame: Input frame (BGR format from OpenCV)
        frame_number: Frame number from video
        labels: List of CVAT labels for this frame
        model: YOLO pose model
    
    Returns:
        Tuple of (annotated_frame, pose_data_dict)
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Run YOLO pose detection
    results = model(frame, verbose=False)
    
    pose_data = {
        'frame_number': frame_number,
        'labels': labels,
        'pose_detected': False,
        'confidence': 0.0,
        'keypoints': {},
        'angles': {},
        'num_persons': 0
    }
    
    # Extract keypoints
    keypoints = extract_keypoints(results)
    
    if keypoints is None or not any(kpt is not None for kpt in keypoints.values()):
        # No pose detected
        y_pos = 30
        for label in labels:
            cv2.putText(frame, f"Label: {label}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30
        
        cv2.putText(frame, "NO POSE DETECTED", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, pose_data
    
    # Pose detected!
    pose_data['pose_detected'] = True
    pose_data['keypoints'] = keypoints
    
    # Calculate average confidence
    confidences = [kpt['confidence'] for kpt in keypoints.values() if kpt is not None]
    pose_data['confidence'] = np.mean(confidences) if confidences else 0.0
    
    # Count number of persons detected
    if len(results) > 0 and results[0].boxes is not None:
        pose_data['num_persons'] = len(results[0].boxes)
    
    # Calculate joint angles
    angles = {}
    
    # Right arm angles
    if all(keypoints.get(k) for k in ['right_hip', 'right_shoulder', 'right_elbow']):
        r_shoulder_angle = calculate_angle(
            (keypoints['right_hip']['x'], keypoints['right_hip']['y']),
            (keypoints['right_shoulder']['x'], keypoints['right_shoulder']['y']),
            (keypoints['right_elbow']['x'], keypoints['right_elbow']['y'])
        )
        angles['right_shoulder'] = r_shoulder_angle
    
    if all(keypoints.get(k) for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        r_elbow_angle = calculate_angle(
            (keypoints['right_shoulder']['x'], keypoints['right_shoulder']['y']),
            (keypoints['right_elbow']['x'], keypoints['right_elbow']['y']),
            (keypoints['right_wrist']['x'], keypoints['right_wrist']['y'])
        )
        angles['right_elbow'] = r_elbow_angle
    
    # Left arm angles
    if all(keypoints.get(k) for k in ['left_hip', 'left_shoulder', 'left_elbow']):
        l_shoulder_angle = calculate_angle(
            (keypoints['left_hip']['x'], keypoints['left_hip']['y']),
            (keypoints['left_shoulder']['x'], keypoints['left_shoulder']['y']),
            (keypoints['left_elbow']['x'], keypoints['left_elbow']['y'])
        )
        angles['left_shoulder'] = l_shoulder_angle
    
    if all(keypoints.get(k) for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
        l_elbow_angle = calculate_angle(
            (keypoints['left_shoulder']['x'], keypoints['left_shoulder']['y']),
            (keypoints['left_elbow']['x'], keypoints['left_elbow']['y']),
            (keypoints['left_wrist']['x'], keypoints['left_wrist']['y'])
        )
        angles['left_elbow'] = l_elbow_angle
    
    # Leg angles
    if all(keypoints.get(k) for k in ['right_hip', 'right_knee', 'right_ankle']):
        r_knee_angle = calculate_angle(
            (keypoints['right_hip']['x'], keypoints['right_hip']['y']),
            (keypoints['right_knee']['x'], keypoints['right_knee']['y']),
            (keypoints['right_ankle']['x'], keypoints['right_ankle']['y'])
        )
        angles['right_knee'] = r_knee_angle
    
    if all(keypoints.get(k) for k in ['left_hip', 'left_knee', 'left_ankle']):
        l_knee_angle = calculate_angle(
            (keypoints['left_hip']['x'], keypoints['left_hip']['y']),
            (keypoints['left_knee']['x'], keypoints['left_knee']['y']),
            (keypoints['left_ankle']['x'], keypoints['left_ankle']['y'])
        )
        angles['left_knee'] = l_knee_angle
    
    # Hip angles
    if all(keypoints.get(k) for k in ['right_shoulder', 'right_hip', 'right_knee']):
        r_hip_angle = calculate_angle(
            (keypoints['right_shoulder']['x'], keypoints['right_shoulder']['y']),
            (keypoints['right_hip']['x'], keypoints['right_hip']['y']),
            (keypoints['right_knee']['x'], keypoints['right_knee']['y'])
        )
        angles['right_hip'] = r_hip_angle
    
    if all(keypoints.get(k) for k in ['left_shoulder', 'left_hip', 'left_knee']):
        l_hip_angle = calculate_angle(
            (keypoints['left_shoulder']['x'], keypoints['left_shoulder']['y']),
            (keypoints['left_hip']['x'], keypoints['left_hip']['y']),
            (keypoints['left_knee']['x'], keypoints['left_knee']['y'])
        )
        angles['left_hip'] = l_hip_angle
    
    pose_data['angles'] = angles
    
    # Draw skeleton on frame
    frame = draw_skeleton_yolo(frame, keypoints)
    
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
    
    cv2.putText(frame, f"Model: YOLOv8-Pose", (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    y_pos += 30
    
    # Determine spike hand
    if keypoints.get('right_wrist') and keypoints.get('left_wrist'):
        r_wrist_y = keypoints['right_wrist']['y']
        l_wrist_y = keypoints['left_wrist']['y']
        spike_side = 'RIGHT' if r_wrist_y < l_wrist_y else 'LEFT'
        
        cv2.putText(frame, f"Spike Hand: {spike_side}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        
        # Display relevant angles
        if spike_side == 'RIGHT' and 'right_shoulder' in angles:
            cv2.putText(frame, f"Shoulder: {angles['right_shoulder']:.1f}deg", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 25
            if 'right_elbow' in angles:
                cv2.putText(frame, f"Elbow: {angles['right_elbow']:.1f}deg", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 25
            if 'right_knee' in angles:
                cv2.putText(frame, f"Knee: {angles['right_knee']:.1f}deg", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif spike_side == 'LEFT' and 'left_shoulder' in angles:
            cv2.putText(frame, f"Shoulder: {angles['left_shoulder']:.1f}deg", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 25
            if 'left_elbow' in angles:
                cv2.putText(frame, f"Elbow: {angles['left_elbow']:.1f}deg", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 25
            if 'left_knee' in angles:
                cv2.putText(frame, f"Knee: {angles['left_knee']:.1f}deg", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame, pose_data


def analyze_extracted_frames_yolo(
    frames_dir: str,
    annotations_path: str,
    output_dir: str = 'frames_with_pose_yolo',
    model_name: str = 'yolov8n-pose.pt',
    save_data: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Process all extracted frames with YOLO pose detection and save results.
    
    Args:
        frames_dir: Directory containing extracted frame images
        annotations_path: Path to CVAT annotations XML file
        output_dir: Directory to save annotated frames
        model_name: YOLO model to use ('yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', etc.)
        save_data: If True, save pose data as JSON
        verbose: Print progress information
    
    Returns:
        Dict containing processed frames, pose data, and summary statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("🏐 VOLLEYBALL FRAME POSE ANALYZER - YOLO")
        print("=" * 60)
        print(f"\n📂 Frames directory: {frames_dir}")
        print(f"📄 Annotations: {annotations_path}")
        print(f"🤖 Model: {model_name}")
        print(f"💾 Output directory: {output_dir}\n")
    
    # Load YOLO model
    if verbose:
        print("🔄 Loading YOLO model...")
    model = YOLO(model_name)
    if verbose:
        print("✅ Model loaded successfully\n")
    
    # Load annotations
    if verbose:
        print("📋 Loading annotations...")
    frame_labels = load_annotations(annotations_path)
    if verbose:
        print(f"   Found {len(frame_labels)} annotated frames")
        print(f"   Labels: {set(label for labels in frame_labels.values() for label in labels)}\n")
    
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
    
    # Parse frame numbers and sort
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
        print(f"📊 Processing {len(frame_info)} frames with YOLO pose detection...\n")
    
    # Process frames
    results = {
        'processed_frames': [],
        'pose_data': [],
        'summary': {
            'total_frames': len(frame_info),
            'pose_detected': 0,
            'no_pose': 0,
            'avg_confidence': 0.0,
            'model': model_name,
            'labels_distribution': {}
        }
    }
    
    for i, info in enumerate(frame_info):
        if verbose and (i + 1) % 10 == 0:
            print(f"   Processing frame {i + 1}/{len(frame_info)}...")
        
        # Read frame
        frame = cv2.imread(info['path'])
        if frame is None:
            print(f"Warning: Could not read {info['filename']}")
            continue
        
        # Process with YOLO pose detection
        annotated_frame, pose_data = process_frame_with_yolo_pose(
            frame,
            info['frame_number'],
            info['labels'],
            model
        )
        
        # Save annotated frame
        output_filename = f"yolo_{info['filename']}"
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
        results['summary']['avg_confidence'] = float(np.mean(confidences))
    
    # Save pose data as JSON
    if save_data:
        json_path = os.path.join(output_dir, 'pose_data_yolo.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\n💾 Pose data saved to: {json_path}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Total frames processed: {results['summary']['total_frames']}")
        print(f"Pose detected: {results['summary']['pose_detected']}")
        print(f"No pose: {results['summary']['no_pose']}")
        print(f"Average confidence: {results['summary']['avg_confidence']:.2%}")
        print(f"\nLabel distribution:")
        for label, count in results['summary']['labels_distribution'].items():
            print(f"   {label}: {count} frames")
        print("=" * 60)
        print(f"\n✅ Annotated frames saved to: {output_dir}")
    
    return results


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add YOLO pose detection to extracted volleyball frames"
    )
    parser.add_argument('frames_dir', help='Directory containing extracted frames')
    parser.add_argument('annotations', help='Path to CVAT annotations XML file')
    parser.add_argument('-o', '--output', default='frames_with_pose_yolo',
                       help='Output directory (default: frames_with_pose_yolo)')
    parser.add_argument('-m', '--model', default='yolov8n-pose.pt',
                       help='YOLO model (default: yolov8n-pose.pt). Options: n, s, m, l, x')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    results = analyze_extracted_frames_yolo(
        frames_dir=args.frames_dir,
        annotations_path=args.annotations,
        output_dir=args.output,
        model_name=args.model,
        verbose=not args.quiet
    )
    
    print(f"\n🎉 Complete! Processed {len(results['processed_frames'])} frames with YOLO")

#!/usr/bin/env python3
"""
Volleyball Frame Pose Analyzer
==============================

Processes extracted frames from CVAT annotations and adds skeletal pose detection.
Reads frames by their frame number and associated label, then applies MediaPipe pose estimation.

Author: AI-Generated
Date: December 2025
Dependencies: opencv-python, mediapipe, numpy, matplotlib

Installation:
    pip install mediapipe opencv-python matplotlib numpy --break-system-packages

Usage:
    from frame_pose_analyzer import analyze_extracted_frames
    
    results = analyze_extracted_frames(
        frames_dir='extracted_frames',
        annotations_path='annotations.xml',
        output_dir='frames_with_pose'
    )
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

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def parse_frame_filename(filename: str) -> Optional[int]:
    """
    Extract frame number from filename.
    
    Supports formats like:
    - frame_000992.png -> 992
    - frame_001234.png -> 1234
    - my_frame_000500.png -> 500
    
    Args:
        filename: Name of the frame file
    
    Returns:
        int: Frame number, or None if not found
    """
    # Match pattern: prefix_NNNNNN.extension
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
        Example: {992: ['approach'], 1082: ['approach', 'jump'], ...}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    frame_labels = {}
    
    # Parse all tracks
    for track in root.findall('track'):
        label = track.get('label')
        
        # Get all box elements (each represents a frame)
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


def get_landmark_coords(landmarks, landmark_id: int, 
                        frame_width: int, frame_height: int) -> Tuple[int, int]:
    """
    Extract pixel coordinates for a specific body landmark.
    
    Args:
        landmarks: MediaPipe pose landmarks object
        landmark_id: ID of the landmark
        frame_width: Width of frame in pixels
        frame_height: Height of frame in pixels
    
    Returns:
        Tuple[int, int]: (x, y) pixel coordinates
    """
    landmark = landmarks.landmark[landmark_id]
    return (int(landmark.x * frame_width), int(landmark.y * frame_height))


def process_frame_with_pose(frame: np.ndarray, 
                            frame_number: int,
                            labels: List[str],
                            pose_detector) -> Tuple[np.ndarray, Dict]:
    """
    Process a single frame: detect pose and annotate with skeletal points.
    
    Args:
        frame: Input frame (BGR format from OpenCV)
        frame_number: Frame number from video
        labels: List of CVAT labels for this frame
        pose_detector: MediaPipe Pose object
    
    Returns:
        Tuple of (annotated_frame, pose_data_dict)
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
        'landmarks': {},
        'angles': {}
    }
    
    if not results.pose_landmarks:
        # No pose detected - just add label info
        y_pos = 30
        for label in labels:
            cv2.putText(frame, f"Label: {label}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30
        
        cv2.putText(frame, "NO POSE DETECTED", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, pose_data
    
    # Pose detected!
    landmarks = results.pose_landmarks
    pose_data['pose_detected'] = True
    
    # Calculate confidence
    confidence = np.mean([
        landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility,
        landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility,
        landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
        landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
        landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
        landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
    ])
    pose_data['confidence'] = confidence
    
    # Extract key landmarks
    try:
        # Right side
        r_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, frame_width, frame_height)
        r_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, frame_width, frame_height)
        r_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, frame_width, frame_height)
        r_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, frame_width, frame_height)
        r_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, frame_width, frame_height)
        r_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, frame_width, frame_height)
        
        # Left side
        l_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, frame_width, frame_height)
        l_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, frame_width, frame_height)
        l_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, frame_width, frame_height)
        l_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP, frame_width, frame_height)
        l_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, frame_width, frame_height)
        l_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, frame_width, frame_height)
        
        # Store landmark coordinates
        pose_data['landmarks'] = {
            'right_shoulder': r_shoulder,
            'right_elbow': r_elbow,
            'right_wrist': r_wrist,
            'right_hip': r_hip,
            'right_knee': r_knee,
            'right_ankle': r_ankle,
            'left_shoulder': l_shoulder,
            'left_elbow': l_elbow,
            'left_wrist': l_wrist,
            'left_hip': l_hip,
            'left_knee': l_knee,
            'left_ankle': l_ankle
        }
        
        # Calculate angles
        r_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        
        l_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        
        pose_data['angles'] = {
            'right_shoulder': r_shoulder_angle,
            'right_elbow': r_elbow_angle,
            'right_hip': r_hip_angle,
            'right_knee': r_knee_angle,
            'left_shoulder': l_shoulder_angle,
            'left_elbow': l_elbow_angle,
            'left_hip': l_hip_angle,
            'left_knee': l_knee_angle
        }
        
    except Exception as e:
        print(f"Warning: Error extracting landmarks for frame {frame_number}: {e}")
    
    # Draw pose landmarks on frame
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
    
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30
    
    # Add angle information if available
    if pose_data['angles']:
        # Determine which side is higher (likely the spiking arm)
        if r_wrist[1] < l_wrist[1]:  # Right is higher (lower y-coord)
            spike_side = 'RIGHT'
            shoulder_angle = pose_data['angles']['right_shoulder']
            elbow_angle = pose_data['angles']['right_elbow']
            knee_angle = pose_data['angles']['right_knee']
        else:
            spike_side = 'LEFT'
            shoulder_angle = pose_data['angles']['left_shoulder']
            elbow_angle = pose_data['angles']['left_elbow']
            knee_angle = pose_data['angles']['left_knee']
        
        cv2.putText(frame, f"Spike Hand: {spike_side}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        
        cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f}deg", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += 25
        
        cv2.putText(frame, f"Elbow: {elbow_angle:.1f}deg", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += 25
        
        cv2.putText(frame, f"Knee: {knee_angle:.1f}deg", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame, pose_data


def analyze_extracted_frames(
    frames_dir: str,
    annotations_path: str,
    output_dir: str = 'frames_with_pose',
    save_data: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Process all extracted frames, add pose detection, and save results.
    
    Args:
        frames_dir: Directory containing extracted frame images
        annotations_path: Path to CVAT annotations XML file
        output_dir: Directory to save annotated frames
        save_data: If True, save pose data as JSON
        verbose: Print progress information
    
    Returns:
        Dict containing:
            - 'processed_frames': List of frame numbers processed
            - 'pose_data': List of pose data dictionaries
            - 'summary': Summary statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("🏐 VOLLEYBALL FRAME POSE ANALYZER")
        print("=" * 60)
        print(f"\n📂 Frames directory: {frames_dir}")
        print(f"📄 Annotations: {annotations_path}")
        print(f"💾 Output directory: {output_dir}\n")
    
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
        print(f"📊 Processing {len(frame_info)} frames with pose detection...\n")
    
    # Process frames
    results = {
        'processed_frames': [],
        'pose_data': [],
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
            
            # Process with pose detection
            annotated_frame, pose_data = process_frame_with_pose(
                frame,
                info['frame_number'],
                info['labels'],
                pose
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
        json_path = os.path.join(output_dir, 'pose_data.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\n💾 Pose data saved to: {json_path}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
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
        description="Add pose detection to extracted volleyball frames"
    )
    parser.add_argument('frames_dir', help='Directory containing extracted frames')
    parser.add_argument('annotations', help='Path to CVAT annotations XML file')
    parser.add_argument('-o', '--output', default='frames_with_pose',
                       help='Output directory (default: frames_with_pose)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    results = analyze_extracted_frames(
        frames_dir=args.frames_dir,
        annotations_path=args.annotations,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    print(f"\n🎉 Complete! Processed {len(results['processed_frames'])} frames")

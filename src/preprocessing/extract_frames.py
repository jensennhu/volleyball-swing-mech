#!/usr/bin/env python3
"""
Extract frames from video based on CVAT XML annotations.
This script reads the annotations.xml file and extracts frames as PNG images.
"""

import xml.etree.ElementTree as ET
import cv2
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set


def parse_annotations(xml_path: str) -> Dict[str, Set[int]]:
    """
    Parse the CVAT XML annotations and extract frame numbers by label.
    
    Args:
        xml_path: Path to the annotations.xml file
        
    Returns:
        Dictionary mapping label names to sets of frame numbers
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Dictionary to store frame numbers by label
    frames_by_label = {}
    all_frames = set()
    
    # Parse all tracks
    for track in root.findall('track'):
        label = track.get('label')
        
        if label not in frames_by_label:
            frames_by_label[label] = set()
        
        # Get all box elements (each represents a frame)
        for box in track.findall('box'):
            frame_num = int(box.get('frame'))
            frames_by_label[label].add(frame_num)
            all_frames.add(frame_num)
    
    # Add 'all' category with all unique frames
    frames_by_label['all'] = all_frames
    
    return frames_by_label


def extract_frames(video_path: str, frame_numbers: Set[int], output_dir: str, 
                   prefix: str = "frame", verbose: bool = True) -> None:
    """
    Extract specific frames from a video and save them as PNG files.
    
    Args:
        video_path: Path to the input video file
        frame_numbers: Set of frame numbers to extract
        output_dir: Directory to save the extracted frames
        prefix: Prefix for the output filenames
        verbose: Whether to print progress information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if verbose:
        print(f"Video info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Frames to extract: {len(frame_numbers)}")
        print()
    
    # Sort frame numbers for sequential reading
    sorted_frames = sorted(frame_numbers)
    
    # Extract frames
    current_frame = 0
    extracted_count = 0
    
    for target_frame in sorted_frames:
        # Skip to the target frame
        if target_frame < current_frame:
            # Frame already passed, need to reset
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            current_frame = target_frame
        else:
            # Skip frames until we reach the target
            while current_frame < target_frame:
                ret = cap.read()
                current_frame += 1
                if not ret[0]:
                    print(f"Warning: Could not read frame {current_frame}")
                    break
        
        # Read the target frame
        ret, frame = cap.read()
        
        if ret:
            # Save the frame
            output_path = os.path.join(output_dir, f"{prefix}_{target_frame:06d}.png")
            cv2.imwrite(output_path, frame)
            extracted_count += 1
            
            if verbose and extracted_count % 100 == 0:
                print(f"Extracted {extracted_count}/{len(frame_numbers)} frames...")
        else:
            print(f"Warning: Could not read frame {target_frame}")
        
        current_frame += 1
    
    cap.release()
    
    if verbose:
        print(f"\nExtraction complete! Extracted {extracted_count} frames to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video based on CVAT annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all annotated frames
  python extract_frames.py video.mp4 annotations.xml -o output_frames

  # Extract only 'jump' labeled frames
  python extract_frames.py video.mp4 annotations.xml -o output_frames -l jump

  # Extract frames from multiple labels
  python extract_frames.py video.mp4 annotations.xml -o output_frames -l jump swing land
        """
    )
    
    parser.add_argument('video', help='Path to the input video file')
    parser.add_argument('annotations', help='Path to the CVAT annotations XML file')
    parser.add_argument('-o', '--output', default='extracted_frames', 
                       help='Output directory for extracted frames (default: extracted_frames)')
    parser.add_argument('-l', '--labels', nargs='+', default=['all'],
                       help='Labels to extract frames for (default: all). Options: approach, jump, swing, land, all')
    parser.add_argument('-p', '--prefix', default='frame',
                       help='Prefix for output filenames (default: frame)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Parse annotations
    if not args.quiet:
        print(f"Parsing annotations from: {args.annotations}")
    
    frames_by_label = parse_annotations(args.annotations)
    
    # Display available labels
    if not args.quiet:
        print(f"\nAvailable labels and frame counts:")
        for label, frames in sorted(frames_by_label.items()):
            if label != 'all':
                print(f"  {label}: {len(frames)} frames")
        print(f"  all: {len(frames_by_label['all'])} unique frames")
        print()
    
    # Collect frames to extract based on requested labels
    frames_to_extract = set()
    for label in args.labels:
        if label not in frames_by_label:
            print(f"Warning: Label '{label}' not found in annotations. Available labels: {', '.join(sorted(frames_by_label.keys()))}")
            continue
        frames_to_extract.update(frames_by_label[label])
    
    if not frames_to_extract:
        print("Error: No frames to extract. Check your label selection.")
        return
    
    # Extract frames
    extract_frames(
        args.video,
        frames_to_extract,
        args.output,
        prefix=args.prefix,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

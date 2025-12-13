#!/usr/bin/env python3
"""
Standalone script to analyze annotations and extract frames.
Configure the parameters below and run this script directly.
"""

# Import the functions from the modules
from extract_frames import parse_annotations, extract_frames
from analyze_annotations import analyze_annotations

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
VIDEO_PATH = "/Users/jensenhu/Documents/GitHub/volley-vision-vids/hitting-session.mp4"           # Path to your video file
ANNOTATIONS_PATH = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/annotations/annotations.xml"     # Path to your annotations XML file
OUTPUT_DIR = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/extracted_frames"          # Where to save extracted frames
LABELS_TO_EXTRACT = ["all"]              # Options: "all", "approach", "jump", "swing", "land"
                                         # or combine multiple: ["jump", "swing", "land"]
FILENAME_PREFIX = "frame"                # Prefix for output files (e.g., "frame_000992.png")
VERBOSE = True                           # Set to False to suppress progress output

CROP_TO_BBOX = True              # Enable cropping!
PADDING_PIXELS = 50     
# Analyzer options
RUN_ANALYZER = True                      # Set to True to analyze annotations first
                                         # This shows detailed statistics before extraction
                                         
# ============================================================================

import xml.etree.ElementTree as ET
import cv2
import os
from typing import Set, Dict, Tuple

def parse_annotations_with_boxes(xml_path: str) -> Tuple[Dict[str, Set[int]], Dict[int, Dict]]:
    """Parse annotations and extract frame numbers with bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    frames_by_label = {}
    all_frames = set()
    frame_boxes = {}
    
    for track in root.findall('track'):
        label = track.get('label')
        
        if label not in frames_by_label:
            frames_by_label[label] = set()
        
        for box in track.findall('box'):
            frame_num = int(box.get('frame'))
            frames_by_label[label].add(frame_num)
            all_frames.add(frame_num)
            
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            if frame_num in frame_boxes:
                # Merge bounding boxes
                existing_bbox = frame_boxes[frame_num]['bbox']
                merged_bbox = (
                    min(xtl, existing_bbox[0]),
                    min(ytl, existing_bbox[1]),
                    max(xbr, existing_bbox[2]),
                    max(ybr, existing_bbox[3])
                )
                frame_boxes[frame_num]['bbox'] = merged_bbox
                frame_boxes[frame_num]['labels'].append(label)
            else:
                frame_boxes[frame_num] = {
                    'labels': [label],
                    'bbox': (xtl, ytl, xbr, ybr)
                }
    
    frames_by_label['all'] = all_frames
    return frames_by_label, frame_boxes


def crop_frame_to_bbox(frame, bbox: Tuple[float, float, float, float], padding: int = 0):
    """Crop frame to bounding box with optional padding."""
    frame_height, frame_width = frame.shape[:2]
    xtl, ytl, xbr, ybr = bbox
    
    x1 = max(0, int(xtl) - padding)
    y1 = max(0, int(ytl) - padding)
    x2 = min(frame_width, int(xbr) + padding)
    y2 = min(frame_height, int(ybr) + padding)
    
    return frame[y1:y2, x1:x2]


def extract_frames_with_crop(video_path: str, frame_numbers: Set[int], 
                             frame_boxes: Dict[int, Dict],
                             output_dir: str, prefix: str = "frame",
                             crop: bool = True, padding: int = 0,
                             verbose: bool = True) -> None:
    """Extract and optionally crop frames from video."""
    os.makedirs(output_dir, exist_ok=True)
    
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
        print(f"  Frames to extract: {len(frame_numbers)}")
        print(f"  Cropping: {'ENABLED' if crop else 'DISABLED'}")
        if crop:
            print(f"  Padding: {padding}px")
        print()
    
    sorted_frames = sorted(frame_numbers)
    current_frame = 0
    extracted_count = 0
    cropped_count = 0
    
    for target_frame in sorted_frames:
        if target_frame < current_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            current_frame = target_frame
        else:
            while current_frame < target_frame:
                ret = cap.read()
                current_frame += 1
                if not ret[0]:
                    break
        
        ret, frame = cap.read()
        
        if ret:
            if crop and target_frame in frame_boxes:
                bbox = frame_boxes[target_frame]['bbox']
                frame_to_save = crop_frame_to_bbox(frame, bbox, padding)
                cropped_count += 1
            else:
                frame_to_save = frame
            
            output_path = os.path.join(output_dir, f"{prefix}_{target_frame:06d}.png")
            cv2.imwrite(output_path, frame_to_save)
            extracted_count += 1
            
            if verbose and extracted_count % 100 == 0:
                print(f"Extracted {extracted_count}/{len(frame_numbers)} frames...")
        
        current_frame += 1
    
    cap.release()
    
    if verbose:
        print(f"\nExtraction complete!")
        print(f"  Total extracted: {extracted_count} frames")
        if crop:
            print(f"  Cropped: {cropped_count} frames")
        print(f"  Output directory: {output_dir}")


def main():
    """Main function to run the frame extraction with cropping."""
    
    print("=" * 60)
    print("Frame Extraction with Bounding Box Cropping")
    print("=" * 60)
    print(f"\nVideo: {VIDEO_PATH}")
    print(f"Annotations: {ANNOTATIONS_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Labels to extract: {', '.join(LABELS_TO_EXTRACT)}")
    print(f"Crop to bbox: {CROP_TO_BBOX}")
    if CROP_TO_BBOX:
        print(f"Padding: {PADDING_PIXELS}px")
    print()
    
    # Parse annotations
    if VERBOSE:
        print("Step 1: Parsing annotations...")
    
    frames_by_label, frame_boxes = parse_annotations_with_boxes(ANNOTATIONS_PATH)
    
    if VERBOSE:
        print(f"\nAvailable labels and frame counts:")
        for label, frames in sorted(frames_by_label.items()):
            if label != 'all':
                print(f"  {label}: {len(frames)} frames")
        print(f"  all: {len(frames_by_label['all'])} unique frames")
        print()
    
    # Collect frames to extract
    if VERBOSE:
        print("Step 2: Collecting frames to extract...")
    
    frames_to_extract = set()
    for label in LABELS_TO_EXTRACT:
        if label not in frames_by_label:
            print(f"Warning: Label '{label}' not found in annotations.")
            continue
        frames_to_extract.update(frames_by_label[label])
    
    if not frames_to_extract:
        print("ERROR: No frames to extract. Check your label selection.")
        return
    
    if VERBOSE:
        print(f"Total frames to extract: {len(frames_to_extract)}")
        print()
        print("Step 3: Extracting frames...")
        print()
    
    # Extract frames
    extract_frames_with_crop(
        video_path=VIDEO_PATH,
        frame_numbers=frames_to_extract,
        frame_boxes=frame_boxes,
        output_dir=OUTPUT_DIR,
        prefix=FILENAME_PREFIX,
        crop=CROP_TO_BBOX,
        padding=PADDING_PIXELS,
        verbose=VERBOSE
    )
    
    print("\n" + "=" * 60)
    print("✓ EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"\nFrames saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
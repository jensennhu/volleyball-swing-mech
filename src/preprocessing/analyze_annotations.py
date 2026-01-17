#!/usr/bin/env python3
"""
Analyze CVAT annotations and show summary statistics.
"""

import xml.etree.ElementTree as ET
import argparse
from collections import defaultdict


def analyze_annotations(xml_path: str):
    """Parse and analyze the CVAT annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get video metadata
    meta = root.find('meta')
    job = meta.find('job')
    original_size = meta.find('original_size')
    
    print("=" * 60)
    print("CVAT Annotation Analysis")
    print("=" * 60)
    
    # Job info
    print(f"\nJob Information:")
    print(f"  Job ID: {job.find('id').text}")
    print(f"  Total frames: {job.find('size').text}")
    print(f"  Start frame: {job.find('start_frame').text}")
    print(f"  Stop frame: {job.find('stop_frame').text}")
    print(f"  Mode: {job.find('mode').text}")
    
    # Video info
    width = original_size.find('width').text
    height = original_size.find('height').text
    print(f"\nVideo Information:")
    print(f"  Resolution: {width}x{height}")
    
    # Labels
    print(f"\nAvailable Labels:")
    labels_elem = job.find('labels')
    if labels_elem is not None:
        for label in labels_elem.findall('label'):
            name_elem = label.find('name')
            if name_elem is None:
                name_elem = label.find('n')
            color_elem = label.find('color')
            
            name = name_elem.text if name_elem is not None else "Unknown"
            color = color_elem.text if color_elem is not None else "Unknown"
            print(f"  - {name} (color: {color})")
    
    # Analyze tracks
    tracks_by_label = defaultdict(list)
    frames_by_label = defaultdict(set)
    
    for track in root.findall('track'):
        label = track.get('label')
        track_id = track.get('id')
        
        boxes = track.findall('box')
        if boxes:
            frames = [int(box.get('frame')) for box in boxes]
            tracks_by_label[label].append({
                'id': track_id,
                'frames': frames,
                'start': min(frames),
                'end': max(frames),
                'count': len(frames)
            })
            frames_by_label[label].update(frames)
    
    # Print track statistics
    print(f"\n" + "=" * 60)
    print("Track Statistics by Label")
    print("=" * 60)
    
    all_frames = set()
    for label in sorted(tracks_by_label.keys()):
        tracks = tracks_by_label[label]
        frames = frames_by_label[label]
        all_frames.update(frames)
        
        print(f"\n{label.upper()}:")
        print(f"  Number of tracks: {len(tracks)}")
        print(f"  Total annotated frames: {len(frames)}")
        print(f"  Frame range: {min(frames)} - {max(frames)}")
        
        # Show each track
        for track in tracks:
            print(f"\n  Track {track['id']}:")
            print(f"    Frames: {track['count']}")
            print(f"    Range: {track['start']} - {track['end']}")
            print(f"    First 10 frames: {track['frames'][:10]}")
            if len(track['frames']) > 10:
                print(f"    Last 10 frames: {track['frames'][-10:]}")
    
    # Overall summary
    print(f"\n" + "=" * 60)
    print("Overall Summary")
    print("=" * 60)
    print(f"Total unique annotated frames: {len(all_frames)}")
    print(f"First annotated frame: {min(all_frames)}")
    print(f"Last annotated frame: {max(all_frames)}")
    
    # Frame overlap analysis
    print(f"\nFrame Overlap Analysis:")
    frame_label_count = defaultdict(list)
    for label, frames in frames_by_label.items():
        for frame in frames:
            frame_label_count[frame].append(label)
    
    overlapping_frames = {f: labels for f, labels in frame_label_count.items() if len(labels) > 1}
    if overlapping_frames:
        print(f"  Frames with multiple labels: {len(overlapping_frames)}")
        print(f"  Examples:")
        for frame in sorted(overlapping_frames.keys())[:5]:
            print(f"    Frame {frame}: {', '.join(overlapping_frames[frame])}")
    else:
        print(f"  No overlapping frames (each frame has only one label)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CVAT annotations and show statistics"
    )
    parser.add_argument('annotations', help='Path to the CVAT annotations XML file')
    
    args = parser.parse_args()
    analyze_annotations(args.annotations)


if __name__ == "__main__":
    main()

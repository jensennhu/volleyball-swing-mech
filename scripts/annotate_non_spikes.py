#!/usr/bin/env python3
"""
Simple Non-Spike Region Annotation Tool
========================================

Interactive tool to mark non-spike regions in your video.
These will be used as negative examples for training the spike detector.

Controls:
  SPACE - Mark start of non-spike region
  ENTER - Mark end of non-spike region  
  → (right arrow) - Next frame
  ← (left arrow) - Previous frame
  j - Jump forward 1 second
  k - Jump backward 1 second
  s - Save current regions
  q - Save and quit

Usage: python annotate_non_spikes.py
"""

import cv2
import json
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
VIDEO_PATH = "data/raw/videos/recorded_videos/hitting-session.mp4"
OUTPUT_PATH = "data/raw/annotations/non_spike_regions.json"
# ============================================================================


def format_time(frame_num, fps):
    """Convert frame number to MM:SS format."""
    seconds = int(frame_num / fps)
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"


def annotate_non_spikes(video_path, output_path):
    """Interactive annotation tool."""
    
    print("=" * 70)
    print("NON-SPIKE REGION ANNOTATION TOOL")
    print("=" * 70)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return
    
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo Info:")
    print(f"  Path: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {format_time(total_frames, fps)}")
    
    print(f"\nControls:")
    print(f"  SPACE       - Mark START of non-spike region")
    print(f"  ENTER       - Mark END of non-spike region")
    print(f"  → (right)   - Next frame")
    print(f"  ← (left)    - Previous frame")
    print(f"  j           - Jump forward 1 second ({int(fps)} frames)")
    print(f"  k           - Jump backward 1 second ({int(fps)} frames)")
    print(f"  s           - Save current regions")
    print(f"  q           - Save and quit")
    print()
    
    # Load existing annotations if they exist
    regions = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
                regions = data.get('regions', [])
            print(f"✅ Loaded {len(regions)} existing regions from {output_path}\n")
        except:
            print(f"⚠️  Could not load existing annotations\n")
    
    # State
    current_frame = 0
    marking_region = False
    region_start = None
    
    while True:
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            if current_frame >= total_frames:
                print("Reached end of video")
            else:
                print(f"Error reading frame {current_frame}")
            break
        
        # Create display
        display = frame.copy()
        
        # Add overlay info
        overlay_height = 150
        overlay = display[:overlay_height].copy()
        overlay[:] = (0, 0, 0)
        alpha = 0.7
        display[:overlay_height] = cv2.addWeighted(overlay, alpha, display[:overlay_height], 1 - alpha, 0)
        
        # Frame info
        time_str = format_time(current_frame, fps)
        cv2.putText(display, f"Frame: {current_frame}/{total_frames} ({time_str})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Marking status
        if marking_region:
            start_time = format_time(region_start, fps)
            cv2.putText(display, f"MARKING REGION - Start: {region_start} ({start_time})", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Press ENTER to finish", 
                       (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(display, "Press SPACE to start marking a non-spike region", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Regions count
        cv2.putText(display, f"Regions marked: {len(regions)}", 
                   (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show existing regions
        for i, region in enumerate(regions[-3:]):  # Show last 3
            start_time = format_time(region['start_frame'], fps)
            end_time = format_time(region['end_frame'], fps)
            duration = region['end_frame'] - region['start_frame']
            info = f"  Region {len(regions) - 2 + i}: {region['start_frame']}-{region['end_frame']} ({duration} frames)"
            cv2.putText(display, info, 
                       (width - 550, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow('Non-Spike Annotation', display)
        
        # Handle keys
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
        
        elif key == 83:  # Right arrow
            current_frame = min(current_frame + 1, total_frames - 1)
        
        elif key == 81:  # Left arrow
            current_frame = max(current_frame - 1, 0)
        
        elif key == ord('j'):
            # Jump forward 1 second
            current_frame = min(current_frame + int(fps), total_frames - 1)
        
        elif key == ord('k'):
            # Jump backward 1 second
            current_frame = max(current_frame - int(fps), 0)
        
        elif key == ord(' '):  # SPACE
            if not marking_region:
                region_start = current_frame
                marking_region = True
                print(f"✓ Region start: frame {region_start} ({format_time(region_start, fps)})")
            else:
                print("⚠️  Already marking a region. Press ENTER to finish it first.")
        
        elif key == 13:  # ENTER
            if marking_region:
                region_end = current_frame
                duration_frames = region_end - region_start
                duration_seconds = duration_frames / fps
                
                # Validate
                if region_end <= region_start:
                    print(f"❌ Error: End frame must be after start frame")
                elif duration_frames < 30:
                    print(f"❌ Error: Region too short ({duration_frames} frames). Need at least 30 frames (1 second).")
                else:
                    regions.append({
                        'start_frame': region_start,
                        'end_frame': region_end,
                        'duration_frames': duration_frames,
                        'duration_seconds': round(duration_seconds, 2),
                        'start_time': format_time(region_start, fps),
                        'end_time': format_time(region_end, fps)
                    })
                    print(f"✅ Region saved: frames {region_start}-{region_end} ({duration_frames} frames, {duration_seconds:.1f}s)")
                    marking_region = False
                    region_start = None
            else:
                print("⚠️  Not marking a region. Press SPACE first.")
        
        elif key == ord('s'):
            # Save current progress
            save_regions(video_path, regions, output_path)
            print(f"💾 Saved {len(regions)} regions to {output_path}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final save
    if regions:
        save_regions(video_path, regions, output_path)
        print(f"\n✅ Final save: {len(regions)} regions saved to {output_path}")
        
        # Summary
        total_duration = sum(r['duration_seconds'] for r in regions)
        print(f"\nSummary:")
        print(f"  Total regions: {len(regions)}")
        print(f"  Total duration: {total_duration:.1f} seconds")
        print(f"  Average duration: {total_duration / len(regions):.1f} seconds")
    else:
        print("\n⚠️  No regions marked")


def save_regions(video_path, regions, output_path):
    """Save regions to JSON file."""
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        'video_path': video_path,
        'annotation_date': datetime.now().isoformat(),
        'num_regions': len(regions),
        'regions': regions,
        'metadata': {
            'purpose': 'Non-spike regions for training binary spike detector',
            'tool': 'annotate_non_spikes.py',
            'version': '1.0'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    annotate_non_spikes(VIDEO_PATH, OUTPUT_PATH)

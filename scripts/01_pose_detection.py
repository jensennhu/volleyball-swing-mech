#!/usr/bin/env python3
"""
Standalone script to add NORMALIZED pose detection to extracted frames.
Configure the parameters below and run directly.

This version extracts normalized features that are:
- Position-invariant (centered at pelvis)
- Scale-invariant (normalized by torso length)
- Resolution-independent (works on any video size)
"""

import sys
import os

# Add current directory to path to import normalized_pose_features
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.frame_pose_analyzer_normalized import analyze_extracted_frames

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
FRAMES_DIR = "data/processed/pose_sequences/frames_downsampled_multi"       # Directory with your extracted PNG frames
ANNOTATIONS_PATH = "data/raw/annotations/annotations.xml"          # Path to your annotations XML file
OUTPUT_DIR =  "data/processed/pose_sequences/frames_with_pose"      # Where to save frames with pose overlays
SAVE_JSON = True                               # Save pose data as JSON file
VERBOSE = True                                 # Show progress output
# ============================================================================


def main():
    """Run the normalized frame pose analyzer."""
    
    print("=" * 70)
    print("🏐 VOLLEYBALL FRAME POSE ANALYZER (NORMALIZED)")
    print("=" * 70)
    print("\n✨ Using NORMALIZED pose features:")
    print("   - Root: Centered at pelvis (hip midpoint)")
    print("   - Scale: Normalized by torso length")
    print("   - Works across different videos, cameras, and resolutions!")
    print()
    print(f"Configuration:")
    print(f"  Frames directory: {FRAMES_DIR}")
    print(f"  Annotations: {ANNOTATIONS_PATH}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Save JSON data: {SAVE_JSON}")
    print()
    
    try:
        results = analyze_extracted_frames(
            frames_dir=FRAMES_DIR,
            annotations_path=ANNOTATIONS_PATH,
            output_dir=OUTPUT_DIR,
            save_data=SAVE_JSON,
            verbose=VERBOSE
        )
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"\nProcessed {len(results['processed_frames'])} frames")
        print(f"Frames with pose detected: {results['summary']['pose_detected']}")
        print(f"Average detection confidence: {results['summary']['avg_confidence']:.1%}")
        
        print(f"\n✨ Features are NORMALIZED:")
        print(f"   ✓ Position-invariant")
        print(f"   ✓ Scale-invariant")
        print(f"   ✓ Resolution-invariant")
        print(f"\nThis means your model will work on:")
        print(f"   • Different video resolutions")
        print(f"   • Different camera distances")
        print(f"   • Different player sizes")
        print(f"   • Different positions in frame")
        
        print(f"\nOutput location: {OUTPUT_DIR}/")
        
        if SAVE_JSON:
            print(f"Normalized pose data JSON: {OUTPUT_DIR}/pose_data_normalized.json")
        
        print("\n💡 Next steps:")
        print("   1. Use pose_data_normalized.json for LSTM training")
        print("   2. Model will generalize much better to new videos!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found")
        print(f"   {e}")
        print(f"\nPlease check:")
        print(f"  1. Does '{FRAMES_DIR}' directory exist?")
        print(f"  2. Does '{ANNOTATIONS_PATH}' file exist?")
        print(f"  3. Are there PNG/JPG files in '{FRAMES_DIR}'?")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

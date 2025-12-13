#!/usr/bin/env python3
"""
Standalone script to add pose detection to extracted frames.
Configure the parameters below and run directly.
"""

from frame_pose_analyzer import analyze_extracted_frames

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
FRAMES_DIR = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/extracted_frames"          # Directory with your extracted PNG frames
ANNOTATIONS_PATH = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/annotations/annotations.xml"     # Path to your annotations XML file
OUTPUT_DIR = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/frames_with_pose"          # Where to save frames with pose overlays
SAVE_JSON = True                         # Save pose data as JSON file
VERBOSE = True                           # Show progress output
# ============================================================================


def main():
    """Run the frame pose analyzer."""
    
    print("=" * 60)
    print("🏐 VOLLEYBALL FRAME POSE ANALYZER")
    print("=" * 60)
    print(f"\nConfiguration:")
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
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"\nProcessed {len(results['processed_frames'])} frames")
        print(f"Frames with pose detected: {results['summary']['pose_detected']}")
        print(f"Average detection confidence: {results['summary']['avg_confidence']:.1%}")
        print(f"\nOutput location: {OUTPUT_DIR}/")
        
        if SAVE_JSON:
            print(f"Pose data JSON: {OUTPUT_DIR}/pose_data.json")
        
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

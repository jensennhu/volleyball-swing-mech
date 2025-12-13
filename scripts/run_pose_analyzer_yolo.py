#!/usr/bin/env python3
"""
Standalone script to add YOLO pose detection to extracted frames.
Configure the parameters below and run directly.
"""

from frame_pose_analyzer_yolo import analyze_extracted_frames_yolo

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
FRAMES_DIR = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/extracted_frames"          # Directory with your extracted PNG frames
ANNOTATIONS_PATH = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/annotations/annotations.xml"     # Path to your annotations XML file
OUTPUT_DIR = "/Users/jensenhu/Documents/GitHub/volleyball-swing-mech/frames_with_pose_yol_l"          # Where to save frames with pose overlays
YOLO_MODEL = "yolov8l-pose.pt"           # YOLO model: n (nano), s (small), m (medium), l (large), x (xlarge)
SAVE_JSON = True                         # Save pose data as JSON file
VERBOSE = True                           # Show progress output
# ============================================================================

"""
YOLO Model Options:
- yolov8n-pose.pt: Fastest, good for real-time (recommended for testing)
- yolov8s-pose.pt: Small, balanced speed/accuracy
- yolov8m-pose.pt: Medium, better accuracy
- yolov8l-pose.pt: Large, high accuracy
- yolov8x-pose.pt: Extra large, best accuracy but slower
"""


def main():
    """Run the YOLO pose analyzer."""
    
    print("=" * 60)
    print("🏐 VOLLEYBALL FRAME POSE ANALYZER - YOLO")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Frames directory: {FRAMES_DIR}")
    print(f"  Annotations: {ANNOTATIONS_PATH}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  YOLO model: {YOLO_MODEL}")
    print(f"  Save JSON data: {SAVE_JSON}")
    print()
    
    try:
        results = analyze_extracted_frames_yolo(
            frames_dir=FRAMES_DIR,
            annotations_path=ANNOTATIONS_PATH,
            output_dir=OUTPUT_DIR,
            model_name=YOLO_MODEL,
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
            print(f"Pose data JSON: {OUTPUT_DIR}/pose_data_yolo.json")
        
        print("\n💡 Tip: Compare with MediaPipe results in 'frames_with_pose/'")
        
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

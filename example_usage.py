#!/usr/bin/env python3
"""
Simple Example: Volleyball Spike Analysis
==========================================

This script demonstrates basic usage of the volleyball spike analyzer.
Modify the parameters below to analyze your own spike videos.
"""

from volleyball_spike_analyzer import (
    analyze_spike_biomechanics,
    print_performance_summary,
    detect_spike_phases,
    extract_spike_features
)
import matplotlib.pyplot as plt
import pickle

# =============================================================================
# CONFIGURATION - Modify these parameters
# =============================================================================

VIDEO_PATH = 'my_spike_video.mp4'  # Path to your spike video
PLAYER_HEIGHT_CM = 180              # Your height in centimeters
NUM_FRAMES = 15                     # Number of frames to analyze (10-20 recommended)
OUTPUT_PLOT = 'spike_analysis.png'  # Where to save the visualization
SAVE_DATA = True                    # Whether to save data for later use

# =============================================================================
# ANALYSIS
# =============================================================================

def main():
    print("🏐 Volleyball Spike Biomechanics Analysis")
    print("=" * 60)
    
    # Step 1: Analyze the video
    print(f"\n📹 Analyzing video: {VIDEO_PATH}")
    print(f"📏 Player height: {PLAYER_HEIGHT_CM} cm")
    print(f"🎬 Frames to analyze: {NUM_FRAMES}")
    
    fig, data = analyze_spike_biomechanics(
        video_path=VIDEO_PATH,
        num_frames=NUM_FRAMES,
        player_height_cm=PLAYER_HEIGHT_CM,
        auto_detect_hand=True,
        output_path=OUTPUT_PLOT
    )
    
    # Step 2: Print performance summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print_performance_summary(data)
    
    # Step 3: Detect movement phases
    print("\n" + "=" * 60)
    print("🎯 MOVEMENT ANALYSIS")
    print("=" * 60)
    phases = detect_spike_phases(data)
    unique_phases = list(dict.fromkeys(phases))  # Preserve order
    print(f"Detected phases: {' → '.join(unique_phases)}")
    
    # Count frames in each phase
    print("\nPhase distribution:")
    from collections import Counter
    phase_counts = Counter(phases)
    for phase, count in phase_counts.items():
        print(f"  • {phase}: {count} frames")
    
    # Step 4: Extract features for machine learning
    print("\n" + "=" * 60)
    print("🤖 MACHINE LEARNING FEATURES")
    print("=" * 60)
    features = extract_spike_features(data)
    print(f"Feature matrix shape: {features.shape}")
    print(f"  • Frames: {features.shape[0]}")
    print(f"  • Features per frame: {features.shape[1]}")
    
    # Step 5: Key insights
    print("\n" + "=" * 60)
    print("💡 KEY INSIGHTS")
    print("=" * 60)
    
    # Jump quality
    jump_height = data['jump_height_cm']
    if jump_height > 70:
        jump_quality = "Excellent! 🌟"
    elif jump_height > 60:
        jump_quality = "Very good! 👍"
    elif jump_height > 50:
        jump_quality = "Good 👌"
    else:
        jump_quality = "Room for improvement 💪"
    print(f"Jump height: {jump_height:.1f} cm - {jump_quality}")
    
    # Arm speed
    arm_speed = data['max_arm_speed']
    if arm_speed > 15:
        speed_quality = "Exceptional! ⚡"
    elif arm_speed > 12:
        speed_quality = "Very fast! 🚀"
    elif arm_speed > 10:
        speed_quality = "Good speed 💨"
    else:
        speed_quality = "Focus on arm swing velocity 📈"
    print(f"Max arm speed: {arm_speed:.2f} m/s - {speed_quality}")
    
    # Arm extension
    min_elbow = min(data['elbow_angles'])
    if min_elbow < 150:
        extension_quality = "Excellent extension! 💪"
    elif min_elbow < 160:
        extension_quality = "Good extension 👍"
    else:
        extension_quality = "Work on full arm extension 📐"
    print(f"Minimum elbow angle: {min_elbow:.1f}° - {extension_quality}")
    
    # Step 6: Save data
    if SAVE_DATA:
        print("\n" + "=" * 60)
        print("💾 SAVING DATA")
        print("=" * 60)
        
        data_package = {
            'raw_data': data,
            'phases': phases,
            'features': features,
            'metadata': {
                'video_path': VIDEO_PATH,
                'player_height_cm': PLAYER_HEIGHT_CM,
                'num_frames': NUM_FRAMES,
                'spike_hand': data['spike_hand'],
                'jump_height_cm': data['jump_height_cm'],
                'max_arm_speed': data['max_arm_speed']
            }
        }
        
        with open('spike_analysis_data.pkl', 'wb') as f:
            pickle.dump(data_package, f)
        
        print("✅ Data saved to: spike_analysis_data.pkl")
        print("✅ Visualization saved to:", OUTPUT_PLOT)
    
    # Step 7: Display plot
    print("\n" + "=" * 60)
    print("📈 Displaying visualization...")
    print("   Close the plot window to exit")
    print("=" * 60)
    plt.show()
    
    print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print(f"\n❌ Error: Video file '{VIDEO_PATH}' not found!")
        print("   Please update VIDEO_PATH in this script to point to your video file.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("   Check the README.md troubleshooting section for help.")

# ============================================================================
# VIDEO BATCH PROCESSING CONFIGURATION
# ============================================================================
# 
# This is a template configuration file for batch processing multiple videos.
# 
# INSTRUCTIONS:
# 1. Copy the VIDEO_CONFIGS list below
# 2. Paste it into scripts/00_batch_extract_frames.py (replace existing VIDEO_CONFIGS)
# 3. Update the paths to match your videos and annotations
# 4. Run: python scripts/00_batch_extract_frames.py
#
# ============================================================================

VIDEO_CONFIGS = [
    # ─────────────────────────────────────────────────────────────────────
    # Video 1: Your first video
    # ─────────────────────────────────────────────────────────────────────
    {
        "name": "hitting-session",              # Unique identifier (no spaces or special chars)
        "video": "data/raw/videos/recorded_videos/hitting-session.mp4",
        "annotations": "data/raw/annotations/annotations.xml"
    },
    
    # ─────────────────────────────────────────────────────────────────────
    # Video 2: Add your second video here
    # ─────────────────────────────────────────────────────────────────────
    # {
    #     "name": "practice-day2",              # MUST be unique!
    #     "video": "data/raw/videos/practice-day2.mp4",
    #     "annotations": "data/raw/annotations/annotations-day2.xml"
    # },
    
    # ─────────────────────────────────────────────────────────────────────
    # Video 3: Add your third video here
    # ─────────────────────────────────────────────────────────────────────
    # {
    #     "name": "player-john",
    #     "video": "data/raw/videos/player-john.mp4",
    #     "annotations": "data/raw/annotations/annotations-john.xml"
    # },
    
    # ─────────────────────────────────────────────────────────────────────
    # Add more videos as needed...
    # ─────────────────────────────────────────────────────────────────────
    # {
    #     "name": "session-morning",
    #     "video": "data/raw/videos/morning-practice.mp4",
    #     "annotations": "data/raw/annotations/morning-annotations.xml"
    # },
    
]

# ============================================================================
# ADDITIONAL CONFIGURATION OPTIONS
# ============================================================================

# Global output directory (all videos will be consolidated here)
OUTPUT_DIR = "data/processed/pose_sequences/frames_downsampled_multi"

# File naming prefix
FILENAME_PREFIX = "frame"

# ────────────────────────────────────────────────────────────────────────────
# Cropping options
# ────────────────────────────────────────────────────────────────────────────
CROP_TO_BBOX = True              # Crop frames to bounding box
PADDING_PIXELS = 50              # Add padding around bounding box

# ────────────────────────────────────────────────────────────────────────────
# Downsampling options
# ────────────────────────────────────────────────────────────────────────────
FRAMES_PER_PHASE = 10            # Number of frames per phase (approach, jump, swing, land)
KEEP_FIRST_LAST = True           # Always keep first and last frames of each phase
SPACING_METHOD = "uniform"       # "uniform" or "random" frame selection

# ────────────────────────────────────────────────────────────────────────────
# Spike sequence detection
# ────────────────────────────────────────────────────────────────────────────
EXPECTED_PHASE_ORDER = ["approach", "jump", "swing", "land"]
MAX_GAP_BETWEEN_PHASES = 100     # Maximum frame gap to consider phases part of same spike

# ────────────────────────────────────────────────────────────────────────────
# Display options
# ────────────────────────────────────────────────────────────────────────────
VERBOSE = True                   # Show detailed progress output
SAVE_METADATA = True             # Save metadata JSON file

# ============================================================================
# VALIDATION CHECKLIST
# ============================================================================
# 
# Before running 00_batch_extract_frames.py, verify:
# 
# □ Each video has a unique "name"
# □ All video paths are correct and files exist
# □ All annotation paths are correct and files exist
# □ Annotations were exported as "CVAT for video 1.1" format
# □ OUTPUT_DIR path is where you want combined output
# 
# ============================================================================

# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────
# Example 1: Simple 2-video setup
# ─────────────────────────────────────────────────────────────────────────
# VIDEO_CONFIGS = [
#     {
#         "name": "session1",
#         "video": "data/raw/videos/session1.mp4",
#         "annotations": "data/raw/annotations/session1.xml"
#     },
#     {
#         "name": "session2",
#         "video": "data/raw/videos/session2.mp4",
#         "annotations": "data/raw/annotations/session2.xml"
#     },
# ]

# ─────────────────────────────────────────────────────────────────────────
# Example 2: Multiple players
# ─────────────────────────────────────────────────────────────────────────
# VIDEO_CONFIGS = [
#     {
#         "name": "player-alice",
#         "video": "data/raw/videos/players/alice_spikes.mp4",
#         "annotations": "data/raw/annotations/alice.xml"
#     },
#     {
#         "name": "player-bob",
#         "video": "data/raw/videos/players/bob_spikes.mp4",
#         "annotations": "data/raw/annotations/bob.xml"
#     },
#     {
#         "name": "player-charlie",
#         "video": "data/raw/videos/players/charlie_spikes.mp4",
#         "annotations": "data/raw/annotations/charlie.xml"
#     },
# ]

# ─────────────────────────────────────────────────────────────────────────
# Example 3: Different sessions/days
# ─────────────────────────────────────────────────────────────────────────
# VIDEO_CONFIGS = [
#     {
#         "name": "2024-01-15-morning",
#         "video": "data/raw/videos/2024/jan15_morning.mp4",
#         "annotations": "data/raw/annotations/jan15_morning.xml"
#     },
#     {
#         "name": "2024-01-15-afternoon",
#         "video": "data/raw/videos/2024/jan15_afternoon.mp4",
#         "annotations": "data/raw/annotations/jan15_afternoon.xml"
#     },
#     {
#         "name": "2024-01-22-practice",
#         "video": "data/raw/videos/2024/jan22_practice.mp4",
#         "annotations": "data/raw/annotations/jan22_practice.xml"
#     },
# ]

# ============================================================================
# TIPS FOR NAMING
# ============================================================================
# 
# Good names:
#   ✓ "hitting-session"
#   ✓ "player-sarah-2024-01-15"
#   ✓ "practice_day2"
#   ✓ "tournament-finals"
# 
# Avoid:
#   ✗ "My Video" (spaces)
#   ✗ "session@1" (special characters)
#   ✗ "test" (not descriptive)
#   ✗ Duplicate names
# 
# ============================================================================

# ============================================================================
# NEXT STEPS AFTER CONFIGURATION
# ============================================================================
# 
# 1. Save this file (optional - for your records)
# 
# 2. Copy VIDEO_CONFIGS into scripts/00_batch_extract_frames.py
# 
# 3. Run batch extraction:
#    python scripts/00_batch_extract_frames.py
# 
# 4. Update paths in scripts/01_pose_detection.py:
#    FRAMES_DIR = "data/processed/pose_sequences/frames_downsampled_multi"
#    OUTPUT_DIR = "data/processed/pose_sequences/frames_with_pose_multi"
# 
# 5. Run pose detection:
#    python scripts/01_pose_detection.py
# 
# 6. Update paths in scripts/02_model_training.py:
#    POSE_DATA_PATH = "data/processed/.../frames_with_pose_multi/pose_data_normalized.json"
#    SPIKE_METADATA_PATH = "data/processed/.../frames_downsampled_multi/spike_sequences_metadata.json"
# 
# 7. Train your model:
#    python scripts/02_model_training.py
# 
# 8. Enjoy 85-90% accuracy! 🏐
# 
# ============================================================================

# ============================================================================
# TROUBLESHOOTING
# ============================================================================
# 
# Error: "Duplicate name"
# → Make sure each video has a unique name
# 
# Error: "Video file not found"
# → Check that video paths are correct
# → Try absolute paths instead of relative
# 
# Error: "Annotations file not found"
# → Check that annotation paths are correct
# → Make sure files were exported from CVAT
# 
# Low accuracy after training:
# → Need at least 3 similar videos
# → Check annotation quality/consistency
# → Verify all videos show volleyball spikes
# 
# ============================================================================

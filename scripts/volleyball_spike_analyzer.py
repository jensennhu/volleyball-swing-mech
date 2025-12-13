"""
Volleyball Spike Biomechanics Analyzer
======================================

A comprehensive tool for analyzing volleyball spike biomechanics using computer vision
and pose estimation. This module extracts key performance metrics including jump height,
arm speed, joint angles, and movement phases from video footage.

Author: AI-Generated
Date: December 2025
Dependencies: opencv-python, mediapipe, numpy, scipy, matplotlib

Installation:
    pip install mediapipe opencv-python matplotlib numpy scipy

Usage:
    from volleyball_spike_analyzer import analyze_spike_biomechanics, print_performance_summary
    
    fig, data = analyze_spike_biomechanics(
        video_path='spike_video.mp4',
        num_frames=15,
        player_height_cm=180,
        auto_detect_hand=True
    )
    
    print_performance_summary(data)
    fig.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter
from typing import Tuple, Dict, List, Optional
import warnings

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a: Tuple[float, float], 
                    b: Tuple[float, float], 
                    c: Tuple[float, float]) -> float:
    """
    Calculate the angle at point b formed by three points a, b, c.
    
    This uses the arctangent method to compute the angle between two vectors:
    vector BA and vector BC.
    
    Args:
        a: First point (x, y) coordinates
        b: Middle point (vertex of angle) (x, y) coordinates
        c: Third point (x, y) coordinates
    
    Returns:
        float: Angle in degrees (0-180)
    
    Example:
        >>> shoulder = (100, 200)
        >>> elbow = (150, 250)
        >>> wrist = (200, 300)
        >>> angle = calculate_angle(shoulder, elbow, wrist)
        >>> print(f"Elbow angle: {angle:.1f}°")
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate angle using arctangent of the two vectors
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Normalize to 0-180 range
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def get_landmark_coords(landmarks, 
                        landmark_id: int, 
                        frame_width: int, 
                        frame_height: int) -> Tuple[int, int]:
    """
    Extract pixel coordinates for a specific body landmark.
    
    MediaPipe returns normalized coordinates (0-1), which need to be
    converted to pixel coordinates based on frame dimensions.
    
    Args:
        landmarks: MediaPipe pose landmarks object
        landmark_id: ID of the landmark (e.g., mp_pose.PoseLandmark.RIGHT_WRIST)
        frame_width: Width of video frame in pixels
        frame_height: Height of video frame in pixels
    
    Returns:
        Tuple[int, int]: (x, y) pixel coordinates
    """
    landmark = landmarks.landmark[landmark_id]
    return (int(landmark.x * frame_width), int(landmark.y * frame_height))


def detect_spike_hand(landmarks, 
                     frame_width: int, 
                     frame_height: int) -> str:
    """
    Automatically detect which hand is performing the spike.
    
    Determines spike hand by comparing wrist heights - the higher wrist
    (lower y-coordinate in image space) is assumed to be the spiking hand.
    
    Args:
        landmarks: MediaPipe pose landmarks object
        frame_width: Width of video frame in pixels
        frame_height: Height of video frame in pixels
    
    Returns:
        str: 'left' or 'right' indicating the detected spike hand
    
    Note:
        This heuristic works well for standard spike motions where one arm
        is significantly elevated. May fail if both arms are at similar heights.
    """
    left_wrist = get_landmark_coords(
        landmarks, mp_pose.PoseLandmark.LEFT_WRIST, frame_width, frame_height
    )
    right_wrist = get_landmark_coords(
        landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, frame_width, frame_height
    )
    
    # Lower y-coordinate = higher in frame
    return 'left' if left_wrist[1] < right_wrist[1] else 'right'


def smooth_data(data: List[float], 
                window_length: int = 5, 
                polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth noisy time-series data.
    
    This filter fits successive sub-sets of adjacent data points with a 
    low-degree polynomial using linear least squares, which preserves 
    features like peaks better than moving averages.
    
    Args:
        data: List of numeric values to smooth
        window_length: Length of the filter window (must be odd and >= 3)
        polyorder: Order of the polynomial used to fit the samples
    
    Returns:
        np.ndarray: Smoothed data array
    
    Note:
        If data is too short for filtering, returns original data unchanged.
    """
    if len(data) < window_length:
        return np.array(data)
    
    try:
        return savgol_filter(data, window_length, polyorder)
    except Exception as e:
        warnings.warn(f"Smoothing failed: {e}. Returning original data.")
        return np.array(data)


def analyze_spike_biomechanics(
    video_path: str,
    num_frames: int = 15,
    player_height_cm: float = 180,
    auto_detect_hand: bool = True,
    spike_hand: str = 'right',
    output_path: Optional[str] = None
) -> Tuple[plt.Figure, Dict]:
    """
    Perform comprehensive biomechanical analysis of a volleyball spike from video.
    
    This function:
    1. Extracts evenly-spaced frames from the video
    2. Detects body pose using MediaPipe
    3. Calculates joint angles, heights, and velocities
    4. Identifies key performance metrics
    5. Generates visualization with annotated frames and graphs
    
    Args:
        video_path: Path to input video file (mp4, avi, mov, etc.)
        num_frames: Number of frames to analyze (more = better temporal resolution)
        player_height_cm: Player's actual height in cm for real-world calibration
        auto_detect_hand: If True, automatically detect which hand is spiking
        spike_hand: 'left' or 'right' - used if auto_detect_hand is False
        output_path: Optional path to save visualization image
    
    Returns:
        Tuple[plt.Figure, Dict]: 
            - matplotlib Figure object with complete visualization
            - Dictionary containing all biomechanics data and metrics
    
    Raises:
        IOError: If video file cannot be opened
        ValueError: If no valid pose data is detected
    
    Example:
        >>> fig, data = analyze_spike_biomechanics(
        ...     'my_spike.mp4',
        ...     num_frames=20,
        ...     player_height_cm=185
        ... )
        >>> print(f"Jump height: {data['jump_height_cm']:.1f} cm")
        >>> print(f"Max arm speed: {data['max_arm_speed']:.2f} m/s")
        >>> fig.savefig('analysis.png', dpi=300, bbox_inches='tight')
    
    Data Dictionary Keys:
        - frame_numbers: List of analyzed frame indices
        - timestamps: Time of each frame in seconds
        - shoulder_angles, elbow_angles, hip_angles, knee_angles: Joint angles in degrees
        - hip_heights, shoulder_heights, wrist_heights: Normalized heights (0-1)
        - torso_angles: Torso lean angles
        - wrist_positions: (x, y) pixel coordinates of wrist
        - arm_speeds: Wrist velocity in m/s
        - confidence_scores: MediaPipe detection confidence per frame
        - spike_hand: 'left' or 'right'
        - jump_height_cm: Vertical jump height in centimeters
        - max_arm_speed: Peak arm velocity in m/s
        - avg_confidence: Average pose detection confidence
    """
    # ==================== VIDEO INITIALIZATION ====================
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Extract video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video Info: {total_frames} frames | {frame_width}x{frame_height} | {fps:.1f} FPS")
    
    # Calculate which frames to analyze (evenly spaced)
    frames_to_analyze = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    # ==================== DATA STRUCTURES ====================
    annotated_frames = []
    biomechanics_data = {
        'frame_numbers': [],
        'timestamps': [],
        'shoulder_angles': [],
        'elbow_angles': [],
        'hip_angles': [],
        'knee_angles': [],
        'hip_heights': [],
        'shoulder_heights': [],
        'wrist_heights': [],
        'torso_angles': [],
        'wrist_positions': [],
        'confidence_scores': [],
        'spike_hand': None
    }
    
    detected_hand = None
    
    # ==================== POSE DETECTION ====================
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        for frame_idx in frames_to_analyze:
            # Read specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                warnings.warn(f"Could not read frame {frame_idx}")
                continue
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if not results.pose_landmarks:
                warnings.warn(f"No pose detected in frame {frame_idx}")
                annotated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                continue
            
            landmarks = results.pose_landmarks
            
            # ==================== HAND DETECTION ====================
            if auto_detect_hand and detected_hand is None:
                detected_hand = detect_spike_hand(landmarks, frame_width, frame_height)
                biomechanics_data['spike_hand'] = detected_hand
                print(f"🎯 Detected spike hand: {detected_hand.upper()}")
            elif not auto_detect_hand and detected_hand is None:
                detected_hand = spike_hand.lower()
                biomechanics_data['spike_hand'] = detected_hand
            
            # ==================== EXTRACT LANDMARKS ====================
            if detected_hand == 'right':
                shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, frame_width, frame_height)
                elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, frame_width, frame_height)
                wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, frame_width, frame_height)
                hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, frame_width, frame_height)
                knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, frame_width, frame_height)
                ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, frame_width, frame_height)
            else:
                shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, frame_width, frame_height)
                elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, frame_width, frame_height)
                wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, frame_width, frame_height)
                hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP, frame_width, frame_height)
                knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, frame_width, frame_height)
                ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, frame_width, frame_height)
            
            # ==================== CALCULATE BIOMECHANICS ====================
            shoulder_angle = calculate_angle(hip, shoulder, elbow)
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            torso_angle = calculate_angle(knee, hip, shoulder)
            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)
            
            # Calculate pose detection confidence
            confidence = np.mean([
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            ])
            
            # ==================== STORE DATA ====================
            biomechanics_data['frame_numbers'].append(frame_idx)
            biomechanics_data['timestamps'].append(frame_idx / fps)
            biomechanics_data['shoulder_angles'].append(shoulder_angle)
            biomechanics_data['elbow_angles'].append(elbow_angle)
            biomechanics_data['torso_angles'].append(torso_angle)
            biomechanics_data['hip_angles'].append(hip_angle)
            biomechanics_data['knee_angles'].append(knee_angle)
            
            # Store heights as normalized values (0=bottom, 1=top)
            biomechanics_data['hip_heights'].append((frame_height - hip[1]) / frame_height)
            biomechanics_data['shoulder_heights'].append((frame_height - shoulder[1]) / frame_height)
            biomechanics_data['wrist_heights'].append((frame_height - wrist[1]) / frame_height)
            
            biomechanics_data['wrist_positions'].append(wrist)
            biomechanics_data['confidence_scores'].append(confidence)
            
            # ==================== ANNOTATE FRAME ====================
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Add text overlays
            cv2.putText(frame, f"Shoulder: {int(shoulder_angle)}deg",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Elbow: {int(elbow_angle)}deg",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Knee: {int(knee_angle)}deg",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            annotated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    
    # ==================== VALIDATION ====================
    if len(biomechanics_data['frame_numbers']) == 0:
        raise ValueError("No valid pose data detected. Check video quality and ensure person is visible.")
    
    # ==================== DATA SMOOTHING ====================
    biomechanics_data['shoulder_angles'] = smooth_data(biomechanics_data['shoulder_angles'])
    biomechanics_data['elbow_angles'] = smooth_data(biomechanics_data['elbow_angles'])
    biomechanics_data['hip_heights'] = smooth_data(biomechanics_data['hip_heights'])
    biomechanics_data['wrist_heights'] = smooth_data(biomechanics_data['wrist_heights'])
    
    # ==================== CALIBRATION ====================
    # Calculate average shoulder-to-hip distance in pixels
    avg_shoulder_hip_pixels = np.mean([
        np.linalg.norm(
            np.array([0, biomechanics_data['shoulder_heights'][i] * frame_height]) -
            np.array([0, biomechanics_data['hip_heights'][i] * frame_height])
        )
        for i in range(len(biomechanics_data['shoulder_heights']))
    ])
    
    # Typical shoulder-to-hip distance is ~27% of total height
    shoulder_hip_cm = player_height_cm * 0.27
    pixels_per_cm = avg_shoulder_hip_pixels / shoulder_hip_cm if shoulder_hip_cm > 0 else 1
    
    # ==================== ARM SPEED CALCULATION ====================
    arm_speeds = [0]  # First frame has no previous reference
    
    for i in range(1, len(biomechanics_data['wrist_positions'])):
        pos1 = np.array(biomechanics_data['wrist_positions'][i-1])
        pos2 = np.array(biomechanics_data['wrist_positions'][i])
        time_diff = biomechanics_data['timestamps'][i] - biomechanics_data['timestamps'][i-1]
        
        if time_diff > 0:
            pixel_distance = np.linalg.norm(pos2 - pos1)
            cm_distance = pixel_distance / pixels_per_cm
            speed_m_per_sec = (cm_distance / 100) / time_diff
            arm_speeds.append(speed_m_per_sec)
        else:
            arm_speeds.append(0)
    
    biomechanics_data['arm_speeds'] = smooth_data(arm_speeds)
    
    # ==================== JUMP HEIGHT CALCULATION ====================
    min_hip_height = min(biomechanics_data['hip_heights'])
    max_hip_height = max(biomechanics_data['hip_heights'])
    jump_height_pixels = (max_hip_height - min_hip_height) * frame_height
    jump_height_cm = jump_height_pixels / pixels_per_cm
    
    biomechanics_data['jump_height_cm'] = jump_height_cm
    biomechanics_data['max_arm_speed'] = max(biomechanics_data['arm_speeds'])
    biomechanics_data['avg_confidence'] = np.mean(biomechanics_data['confidence_scores'])
    
    # ==================== IDENTIFY KEY FRAMES ====================
    max_shoulder_idx = np.argmax(biomechanics_data['shoulder_angles'])
    max_wrist_height_idx = np.argmax(biomechanics_data['wrist_heights'])
    max_hip_height_idx = np.argmax(biomechanics_data['hip_heights'])
    min_elbow_idx = np.argmin(biomechanics_data['elbow_angles'])
    max_arm_speed_idx = np.argmax(biomechanics_data['arm_speeds'])
    min_knee_idx = np.argmin(biomechanics_data['knee_angles'])
    
    # ==================== CREATE VISUALIZATION ====================
    fig = plt.figure(figsize=(20, 20))
    
    # Display annotated frames (rows 1-3)
    cols = 5
    for i, img in enumerate(annotated_frames):
        ax = plt.subplot(5, cols, i+1)
        ax.imshow(img)
        ax.axis("off")
        
        title = f"Frame {frames_to_analyze[i]}\n{biomechanics_data['timestamps'][i]:.2f}s"
        title_color = 'black'
        
        # Highlight key performance frames
        if i == max_shoulder_idx:
            title += "\n🔥 MAX SHOULDER"
            title_color = '#FF6B6B'
            ax.add_patch(Rectangle((0, 0), img.shape[1]-1, img.shape[0]-1,
                                   fill=False, edgecolor='#FF6B6B', linewidth=5))
        if i == max_wrist_height_idx:
            title += "\n⬆️ PEAK HEIGHT"
            title_color = '#95E1D3'
        if i == min_elbow_idx:
            title += "\n💪 EXTENSION"
            title_color = '#4ECDC4'
        if i == max_arm_speed_idx:
            title += "\n⚡ MAX SPEED"
            title_color = '#FFD93D'
        
        ax.set_title(title, fontsize=8, fontweight='bold', color=title_color)
    
    # ==================== BIOMECHANICS GRAPHS (ROW 4) ====================
    
    # Graph 1: Arm Joint Angles
    ax1 = plt.subplot(5, 3, 13)
    ax1.plot(biomechanics_data['timestamps'], biomechanics_data['shoulder_angles'],
             'o-', label='Shoulder', linewidth=2.5, markersize=7, color='#FF6B6B')
    ax1.plot(biomechanics_data['timestamps'], biomechanics_data['elbow_angles'],
             's-', label='Elbow', linewidth=2.5, markersize=7, color='#4ECDC4')
    ax1.axhline(y=180, color='gray', linestyle='--', alpha=0.5, label='Full Extension')
    ax1.scatter(biomechanics_data['timestamps'][max_shoulder_idx],
               biomechanics_data['shoulder_angles'][max_shoulder_idx],
               s=300, marker='*', color='red', zorder=5, edgecolors='black', linewidths=1.5)
    ax1.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Angle (degrees)', fontsize=10, fontweight='bold')
    ax1.set_title('Arm Joint Angles', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Graph 2: Height Tracking (Jump Analysis)
    ax2 = plt.subplot(5, 3, 14)
    ax2.plot(biomechanics_data['timestamps'], biomechanics_data['wrist_heights'],
             'o-', label='Wrist', linewidth=2.5, markersize=7, color='#95E1D3')
    ax2.plot(biomechanics_data['timestamps'], biomechanics_data['shoulder_heights'],
             's-', label='Shoulder', linewidth=2.5, markersize=7, color='#F38181')
    ax2.plot(biomechanics_data['timestamps'], biomechanics_data['hip_heights'],
             '^-', label='Hip', linewidth=2.5, markersize=7, color='#AA96DA')
    ax2.scatter(biomechanics_data['timestamps'][max_wrist_height_idx],
               biomechanics_data['wrist_heights'][max_wrist_height_idx],
               s=300, marker='*', color='red', zorder=5, edgecolors='black', linewidths=1.5)
    ax2.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Normalized Height', fontsize=10, fontweight='bold')
    ax2.set_title('Body Position Heights (Jump Tracking)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Graph 3: Arm Speed (Velocity Analysis)
    ax3 = plt.subplot(5, 3, 15)
    ax3.plot(biomechanics_data['timestamps'], biomechanics_data['arm_speeds'],
             'o-', linewidth=2.5, markersize=7, color='#FFD93D')
    ax3.scatter(biomechanics_data['timestamps'][max_arm_speed_idx],
               biomechanics_data['arm_speeds'][max_arm_speed_idx],
               s=300, marker='*', color='red', zorder=5, edgecolors='black', linewidths=1.5)
    ax3.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Speed (m/s)', fontsize=10, fontweight='bold')
    ax3.set_title('Arm Speed (Wrist Velocity)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(biomechanics_data['timestamps'], biomechanics_data['arm_speeds'],
                     alpha=0.3, color='#FFD93D')
    
    # ==================== METRICS SUMMARY (ROW 5) ====================
    ax4 = plt.subplot(5, 1, 5)
    ax4.axis('off')
    
    metrics_text = f"""
    ═══════════════════════════════════════════════════════════════════════════════════════════
    📊 VOLLEYBALL SPIKE PERFORMANCE ANALYSIS
    ═══════════════════════════════════════════════════════════════════════════════════════════
    
    🎯 SPIKE HAND: {biomechanics_data['spike_hand'].upper()}  |  🎥 CONFIDENCE: {biomechanics_data['avg_confidence']:.1%}
    
    🚀 JUMP HEIGHT: {jump_height_cm:.1f} cm  |  ⚡ MAX ARM SPEED: {biomechanics_data['max_arm_speed']:.2f} m/s
    
    📐 Peak Shoulder Angle: {max(biomechanics_data['shoulder_angles']):.1f}°  |  💪 Min Elbow Angle: {min(biomechanics_data['elbow_angles']):.1f}°
    
    🦵 Min Knee Angle: {min(biomechanics_data['knee_angles']):.1f}°  |  📏 Torso Angle Range: {min(biomechanics_data['torso_angles']):.1f}° - {max(biomechanics_data['torso_angles']):.1f}°
    
    🎯 Contact Point: {biomechanics_data['timestamps'][max_wrist_height_idx]:.2f}s  |  ⚡ Speed Peak: {biomechanics_data['timestamps'][max_arm_speed_idx]:.2f}s
    
    📏 Calibration: Player height = {player_height_cm} cm  |  Shoulder-Hip pixels = {avg_shoulder_hip_pixels:.1f}px
    ═══════════════════════════════════════════════════════════════════════════════════════════
    """
    
    ax4.text(0.5, 0.5, metrics_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            family='monospace')
    
    plt.suptitle(f"Volleyball Spike Biomechanics Analysis - {detected_hand.upper()} Hand",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to: {output_path}")
    
    return fig, biomechanics_data


def detect_spike_phases(biomechanics_data: Dict) -> List[str]:
    """
    Identify movement phases throughout the spike motion.
    
    Phases are determined by analyzing hip height, wrist height, and arm speed
    trajectories to identify: approach, jump, arm swing, contact, and follow-through.
    
    Args:
        biomechanics_data: Dictionary returned by analyze_spike_biomechanics()
    
    Returns:
        List[str]: Phase label for each frame
            - 'approach': Initial run-up phase
            - 'jump': Takeoff and ascent
            - 'arm_swing': Arm cocking and forward swing
            - 'contact': Ball contact moment
            - 'follow_through': Post-contact deceleration
    
    Example:
        >>> phases = detect_spike_phases(data)
        >>> print(f"Phases: {' → '.join(set(phases))}")
        >>> # Output: Phases: approach → jump → arm_swing → contact → follow_through
    """
    phases = []
    n_frames = len(biomechanics_data['frame_numbers'])
    
    hip_heights = np.array(biomechanics_data['hip_heights'])
    wrist_heights = np.array(biomechanics_data['wrist_heights'])
    arm_speeds = np.array(biomechanics_data['arm_speeds'])
    
    # Identify key transition points
    max_hip_idx = np.argmax(hip_heights)
    max_wrist_idx = np.argmax(wrist_heights)
    max_speed_idx = np.argmax(arm_speeds)
    
    # Assign phases based on key events
    for i in range(n_frames):
        if i < max_hip_idx * 0.7:
            phases.append('approach')
        elif i < max_hip_idx:
            phases.append('jump')
        elif i < max_speed_idx:
            phases.append('arm_swing')
        elif i <= max_wrist_idx + 1:
            phases.append('contact')
        else:
            phases.append('follow_through')
    
    return phases


def extract_spike_features(biomechanics_data: Dict) -> np.ndarray:
    """
    Extract feature matrix suitable for machine learning applications.
    
    Combines raw measurements (angles, heights) with derived features
    (velocities, accelerations) into a standardized feature matrix.
    
    Args:
        biomechanics_data: Dictionary returned by analyze_spike_biomechanics()
    
    Returns:
        np.ndarray: Feature matrix of shape (n_frames, 15)
        
    Features (in order):
        0-4: Joint angles (shoulder, elbow, torso, hip, knee)
        5-7: Body heights (hip, shoulder, wrist) - normalized
        8: Arm speed (wrist velocity)
        9: Timestamp
        10: Detection confidence score
        11: Torso extension (shoulder_height - hip_height)
        12: Arm reach (wrist_height - shoulder_height)
        13: Acceleration (change in arm speed)
        14: Vertical velocity (change in wrist height)
    
    Example:
        >>> features = extract_spike_features(data)
        >>> print(f"Feature matrix shape: {features.shape}")
        >>> # Use for ML: train LSTM, transformer, or other sequence model
        >>> # X_train = features, y_train = performance_score
    """
    features_list = []
    
    for i in range(len(biomechanics_data['frame_numbers'])):
        frame_features = [
            # Joint angles (degrees)
            biomechanics_data['shoulder_angles'][i],
            biomechanics_data['elbow_angles'][i],
            biomechanics_data['torso_angles'][i],
            biomechanics_data['hip_angles'][i],
            biomechanics_data['knee_angles'][i],
            
            # Body positions (normalized 0-1)
            biomechanics_data['hip_heights'][i],
            biomechanics_data['shoulder_heights'][i],
            biomechanics_data['wrist_heights'][i],
            
            # Velocity (m/s)
            biomechanics_data['arm_speeds'][i],
            
            # Temporal
            biomechanics_data['timestamps'][i],
            
            # Confidence
            biomechanics_data['confidence_scores'][i],
            
            # Derived spatial features
            biomechanics_data['shoulder_heights'][i] - biomechanics_data['hip_heights'][i],  # torso extension
            biomechanics_data['wrist_heights'][i] - biomechanics_data['shoulder_heights'][i],  # arm reach
        ]
        
        # Derived temporal features (velocity derivatives = acceleration proxy)
        if i > 0:
            speed_change = biomechanics_data['arm_speeds'][i] - biomechanics_data['arm_speeds'][i-1]
            wrist_height_change = biomechanics_data['wrist_heights'][i] - biomechanics_data['wrist_heights'][i-1]
        else:
            speed_change = 0
            wrist_height_change = 0
        
        frame_features.extend([speed_change, wrist_height_change])
        features_list.append(frame_features)
    
    return np.array(features_list)


def print_performance_summary(biomechanics_data: Dict) -> None:
    """
    Print a comprehensive text summary of spike performance metrics.
    
    Args:
        biomechanics_data: Dictionary returned by analyze_spike_biomechanics()
    
    Output:
        Formatted console output with all key metrics organized by category:
        - Video metadata (hand, frames, confidence, duration)
        - Jump metrics (height, timing, range)
        - Arm mechanics (speed, angles, timing)
        - Contact point analysis
        - Lower body mechanics (knee, hip, torso angles)
    
    Example:
        >>> fig, data = analyze_spike_biomechanics('spike.mp4')
        >>> print_performance_summary(data)
        ═══════════════════════════
        🏐 VOLLEYBALL SPIKE PERFORMANCE SUMMARY
        ═══════════════════════════
        ...
    """
    print("\n" + "="*80)
    print("🏐 VOLLEYBALL SPIKE PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\n📹 VIDEO METRICS:")
    print(f"   • Spike Hand: {biomechanics_data['spike_hand'].upper()}")
    print(f"   • Frames Analyzed: {len(biomechanics_data['frame_numbers'])}")
    print(f"   • Average Confidence: {biomechanics_data['avg_confidence']:.1%}")
    print(f"   • Duration: {biomechanics_data['timestamps'][-1]:.2f}s")
    
    print(f"\n🚀 JUMP METRICS:")
    print(f"   • Jump Height: {biomechanics_data['jump_height_cm']:.1f} cm")
    print(f"   • Peak Hip Height at: {biomechanics_data['timestamps'][np.argmax(biomechanics_data['hip_heights'])]:.2f}s")
    print(f"   • Hip Height Range: {min(biomechanics_data['hip_heights']):.3f} - {max(biomechanics_data['hip_heights']):.3f}")
    
    print(f"\n💪 ARM MECHANICS:")
    print(f"   • Max Arm Speed: {biomechanics_data['max_arm_speed']:.2f} m/s")
    print(f"   • Speed Peak at: {biomechanics_data['timestamps'][np.argmax(biomechanics_data['arm_speeds'])]:.2f}s")
    print(f"   • Max Shoulder Angle: {max(biomechanics_data['shoulder_angles']):.1f}°")
    print(f"   • Min Elbow Angle: {min(biomechanics_data['elbow_angles']):.1f}° (full extension)")
    print(f"   • Elbow Extension at: {biomechanics_data['timestamps'][np.argmin(biomechanics_data['elbow_angles'])]:.2f}s")
    
    print(f"\n🎯 CONTACT POINT:")
    contact_idx = np.argmax(biomechanics_data['wrist_heights'])
    print(f"   • Time: {biomechanics_data['timestamps'][contact_idx]:.2f}s")
    print(f"   • Frame: {biomechanics_data['frame_numbers'][contact_idx]}")
    print(f"   • Wrist Height: {biomechanics_data['wrist_heights'][contact_idx]:.3f} (normalized)")
    print(f"   • Shoulder Angle: {biomechanics_data['shoulder_angles'][contact_idx]:.1f}°")
    print(f"   • Elbow Angle: {biomechanics_data['elbow_angles'][contact_idx]:.1f}°")
    
    print(f"\n🦵 LOWER BODY:")
    print(f"   • Min Knee Angle: {min(biomechanics_data['knee_angles']):.1f}° (max flexion)")
    print(f"   • Hip Angle Range: {min(biomechanics_data['hip_angles']):.1f}° - {max(biomechanics_data['hip_angles']):.1f}°")
    print(f"   • Torso Angle Range: {min(biomechanics_data['torso_angles']):.1f}° - {max(biomechanics_data['torso_angles']):.1f}°")
    
    print("\n" + "="*80)


# ==================== MAIN EXECUTION EXAMPLE ====================
if __name__ == "__main__":
    """
    Example usage demonstrating all features of the analyzer.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python volleyball_spike_analyzer.py <video_path> [player_height_cm]")
        print("Example: python volleyball_spike_analyzer.py spike.mp4 185")
        sys.exit(1)
    
    video_path = sys.argv[1]
    player_height_cm = float(sys.argv[2]) if len(sys.argv) > 2 else 180
    
    print(f"🏐 Analyzing volleyball spike from: {video_path}")
    print(f"📏 Player height: {player_height_cm} cm\n")
    
    # Perform analysis
    fig, data = analyze_spike_biomechanics(
        video_path=video_path,
        num_frames=15,
        player_height_cm=player_height_cm,
        auto_detect_hand=True,
        output_path='spike_analysis.png'
    )
    
    # Display results
    plt.show()
    print_performance_summary(data)
    
    # Detect movement phases
    phases = detect_spike_phases(data)
    data['phases'] = phases
    print(f"\n🎯 Movement Phases: {' → '.join(dict.fromkeys(phases))}")
    
    # Extract ML features
    features = extract_spike_features(data)
    print(f"\n📊 Feature Matrix: {features.shape[0]} frames × {features.shape[1]} features")
    
    # Save data for further analysis
    import pickle
    with open('spike_biomechanics_data.pkl', 'wb') as f:
        pickle.dump({
            'features': features,
            'phases': phases,
            'raw_data': data,
            'metadata': {
                'video': video_path,
                'player_height_cm': player_height_cm,
                'spike_hand': data['spike_hand'],
                'jump_height_cm': data['jump_height_cm'],
                'max_arm_speed_ms': data['max_arm_speed'],
            }
        }, f)
    
    print("💾 Data saved to 'spike_biomechanics_data.pkl'")
    print("✅ Analysis complete!")

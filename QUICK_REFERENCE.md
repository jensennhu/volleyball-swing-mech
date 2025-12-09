# Quick Reference Guide
## Volleyball Spike Biomechanics Analyzer

---

## 🚀 Quick Start

### Python (Video Analysis)
```python
from volleyball_spike_analyzer import analyze_spike_biomechanics

fig, data = analyze_spike_biomechanics(
    'spike.mp4',          # Your video file
    player_height_cm=180  # Your height
)
fig.show()
```

### R (Data Analysis)
```r
source("volleyball_spike_analyzer.R")
data <- load_pose_data("pose_landmarks.csv")
results <- analyze_spike_biomechanics(data, player_height_cm = 180)
plot_spike_analysis(results)
```

---

## 📋 Key Functions

### Python
```python
# Main analysis
fig, data = analyze_spike_biomechanics(video_path, num_frames, player_height_cm)

# Print summary
print_performance_summary(data)

# Detect phases
phases = detect_spike_phases(data)

# Extract ML features
features = extract_spike_features(data)
```

### R
```r
# Load data
data <- load_pose_data("file.csv")

# Main analysis
results <- analyze_spike_biomechanics(data, player_height_cm, spike_hand)

# Print summary
print_performance_summary(results)

# Visualize
plot_spike_analysis(results, output_file = "plot.png")

# Detect phases
phases <- detect_spike_phases(results)

# Extract features
features <- extract_spike_features(results)

# Export
export_results(results, output_dir = "output")
```

---

## 📊 Output Metrics

### Jump Analysis
- **Jump height** (cm) - Vertical displacement
- **Peak hip height** (timestamp) - Highest point timing
- **Hip height range** - Normalized 0-1 values

### Arm Mechanics
- **Max arm speed** (m/s) - Peak wrist velocity
- **Shoulder angle** (degrees) - Maximum rotation
- **Elbow angle** (degrees) - Extension at contact
- **Timing** - When each peak occurs

### Contact Point
- **Time** (seconds) - When ball is contacted
- **Frame** - Video frame number
- **Joint positions** - All body angles at contact

### Lower Body
- **Knee angle** (degrees) - Maximum flexion
- **Hip angle** (degrees) - Range of motion
- **Torso angle** (degrees) - Lean and rotation

---

## 🎯 Performance Benchmarks

### Jump Height
- **>70 cm** - Excellent elite level
- **60-70 cm** - Very good competitive
- **50-60 cm** - Good recreational
- **<50 cm** - Needs improvement

### Arm Speed
- **>15 m/s** - Elite professional
- **12-15 m/s** - High competitive
- **10-12 m/s** - Good club level
- **<10 m/s** - Developing

### Elbow Extension
- **<150°** - Excellent full extension
- **150-160°** - Good extension
- **>160°** - Limited extension, needs work

---

## 🔧 Common Parameters

### analyze_spike_biomechanics()

**Python:**
```python
analyze_spike_biomechanics(
    video_path='spike.mp4',      # Required: video file
    num_frames=15,                # 10-20 recommended
    player_height_cm=180,         # For calibration
    auto_detect_hand=True,        # Auto or manual
    spike_hand='right',           # If auto_detect=False
    output_path='analysis.png'    # Save location
)
```

**R:**
```r
analyze_spike_biomechanics(
  data,                           # Required: loaded CSV
  player_height_cm = 180,         # For calibration
  spike_hand = "auto",            # "auto", "left", "right"
  frame_width = 1920,             # Video dimensions
  frame_height = 1080
)
```

---

## 📁 File Formats

### Input (Python)
- **Video**: .mp4, .avi, .mov (any OpenCV-supported format)
- **Resolution**: 720p or higher recommended
- **Person**: Fully visible in frame

### Input (R)
- **CSV**: Pose landmarks with columns:
  - frame, timestamp
  - {side}_{landmark}_x, {side}_{landmark}_y
  - Required landmarks: shoulder, elbow, wrist, hip, knee, ankle

### Output
- **PNG**: High-resolution visualization
- **PKL** (Python): Complete data package
- **CSV** (R): Multiple analysis files
- **Console**: Formatted text summary

---

## 💡 Tips & Tricks

### For Best Results
1. **Good lighting** - Well-lit, even illumination
2. **High contrast** - Wear contrasting clothing
3. **Full visibility** - Entire body in frame
4. **Side angle** - Perpendicular to camera
5. **Stable camera** - Minimize shake/movement

### Common Issues
- **Low confidence**: Improve lighting, remove background clutter
- **No pose detected**: Check person visibility, video quality
- **Inaccurate heights**: Verify player_height_cm parameter
- **Choppy data**: Increase num_frames, apply more smoothing

### Optimization
- **More frames** (15-20) = Better temporal resolution
- **Shorter clips** = Faster processing
- **720p minimum** = Acceptable accuracy
- **1080p+** = Best results

---

## 🎓 Feature Descriptions

### 15 ML Features per Frame

1. **shoulder_angle** - Arm elevation angle
2. **elbow_angle** - Arm bend/extension
3. **torso_angle** - Body lean
4. **hip_angle** - Hip flexion
5. **knee_angle** - Knee bend
6. **hip_height** - Vertical position (0-1)
7. **shoulder_height** - Vertical position (0-1)
8. **wrist_height** - Vertical position (0-1)
9. **arm_speed** - Wrist velocity (m/s)
10. **timestamp** - Time in video (s)
11. **confidence** - Detection quality (0-1)
12. **torso_extension** - Shoulder-hip separation
13. **arm_reach** - Wrist-shoulder distance
14. **acceleration** - Speed change
15. **vertical_velocity** - Height change rate

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Can't open video | Check file path, try converting with ffmpeg |
| No pose detected | Ensure person visible, improve lighting |
| Low confidence | Better video quality, contrasting clothes |
| Wrong hand detected | Set `auto_detect_hand=False`, specify hand |
| Inaccurate jump height | Verify `player_height_cm` is correct |
| Smoothing errors | Reduce window_length for short sequences |
| Column not found (R) | Check CSV column names match format |

---

## 📞 Need Help?

1. Check **README.md** for detailed documentation
2. Review **example_usage.py** or **example_usage.R**
3. Verify data format with **example_pose_data.csv**
4. Ensure all dependencies installed from **requirements.txt**

---

**Happy analyzing! 🏐**

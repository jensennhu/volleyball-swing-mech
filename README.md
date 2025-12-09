# Volleyball Spike Biomechanics Analyzer

A comprehensive toolkit for analyzing volleyball spike biomechanics using computer vision and pose estimation. Available in both **Python** (with video processing) and **R** (for data analysis).

---

## 🏐 Features

### Core Analysis Capabilities
- **Automatic spike hand detection** - Identifies left or right-handed spikes
- **Joint angle calculation** - Shoulder, elbow, hip, knee, and torso angles
- **Jump height estimation** - Real-world calibrated measurements in centimeters
- **Arm speed tracking** - Velocity calculations in m/s
- **Movement phase detection** - Approach, jump, arm swing, contact, follow-through
- **Comprehensive visualization** - Annotated frames with graphs and metrics
- **Feature extraction** - ML-ready feature matrices for sequence modeling
- **Data smoothing** - Savitzky-Golay filtering for noise reduction

### Performance Metrics
- Jump height (cm)
- Maximum arm speed (m/s)
- Peak joint angles
- Contact point timing and position
- Movement phase timing
- Confidence scores for pose detection

---

## 📦 Installation

### Python Version

```bash
# Install dependencies
pip install mediapipe opencv-python matplotlib numpy scipy

# Or using requirements.txt
pip install -r requirements.txt
```

**Requirements:**
- Python 3.7+
- opencv-python (cv2)
- mediapipe
- matplotlib
- numpy
- scipy

### R Version

```r
# Install required packages
install.packages(c("tidyverse", "ggplot2", "gridExtra", "signal", "pracma"))
```

**Requirements:**
- R 4.0+
- tidyverse
- ggplot2
- gridExtra
- signal
- pracma

---

## 🚀 Quick Start

### Python (Video Processing)

```python
from volleyball_spike_analyzer import analyze_spike_biomechanics, print_performance_summary

# Analyze a spike video
fig, data = analyze_spike_biomechanics(
    video_path='my_spike.mp4',
    num_frames=15,              # Number of frames to analyze
    player_height_cm=180,       # Your height for calibration
    auto_detect_hand=True,      # Auto-detect spike hand
    output_path='analysis.png'  # Save visualization
)

# Display results
print_performance_summary(data)
fig.show()

# Detect movement phases
from volleyball_spike_analyzer import detect_spike_phases
phases = detect_spike_phases(data)
print(f"Phases: {' → '.join(dict.fromkeys(phases))}")

# Extract features for ML
from volleyball_spike_analyzer import extract_spike_features
features = extract_spike_features(data)
print(f"Feature matrix shape: {features.shape}")
```

### R (Data Analysis)

```r
source("volleyball_spike_analyzer.R")

# Load pre-extracted pose data
data <- load_pose_data("pose_landmarks.csv")

# Analyze biomechanics
results <- analyze_spike_biomechanics(
  data,
  player_height_cm = 180,
  spike_hand = "auto"
)

# Print summary
print_performance_summary(results)

# Create visualizations
plot_spike_analysis(results, output_file = "spike_analysis.png")

# Detect phases
phases <- detect_spike_phases(results)

# Extract features
features <- extract_spike_features(results)

# Export results
export_results(results, output_dir = "analysis_output")
```

---

## 📊 Data Format for R Version

The R version requires pre-extracted pose landmark data. You can generate this using:
1. The Python version's MediaPipe processing
2. Other pose estimation tools (OpenPose, AlphaPose, etc.)
3. Manual annotation tools

### Expected CSV Format (Wide Format)

```csv
frame,timestamp,right_shoulder_x,right_shoulder_y,right_elbow_x,right_elbow_y,right_wrist_x,right_wrist_y,right_hip_x,right_hip_y,right_knee_x,right_knee_y,right_ankle_x,right_ankle_y,left_shoulder_x,left_shoulder_y,left_elbow_x,left_elbow_y,left_wrist_x,left_wrist_y,left_hip_x,left_hip_y,left_knee_x,left_knee_y,left_ankle_x,left_ankle_y
0,0.000,456.2,234.5,523.1,345.6,598.4,456.7,445.3,456.8,447.2,678.9,449.1,890.2,344.5,235.6,277.8,346.7,211.2,457.8,355.4,457.9,357.3,679.0,359.2,891.3
1,0.033,457.1,233.2,524.5,344.1,600.2,454.3,446.1,455.2,448.0,677.5,449.9,889.0,345.2,234.8,278.5,345.9,212.0,456.5,356.1,456.8,358.0,678.2,359.9,890.5
...
```

**Column Specifications:**
- `frame`: Frame number (integer)
- `timestamp`: Time in seconds (float)
- `{side}_{landmark}_x`: X-coordinate in pixels (float)
- `{side}_{landmark}_y`: Y-coordinate in pixels (float)

**Required Landmarks:**
- shoulder (left/right)
- elbow (left/right)
- wrist (left/right)
- hip (left/right)
- knee (left/right)
- ankle (left/right)

---

## 🔧 Advanced Usage

### Python: Export Data for R Analysis

```python
import pandas as pd
from volleyball_spike_analyzer import analyze_spike_biomechanics

# Analyze video
fig, data = analyze_spike_biomechanics('spike.mp4')

# Convert to R-compatible format
df_export = pd.DataFrame({
    'frame': data['frame_numbers'],
    'timestamp': data['timestamps'],
    'right_wrist_x': [pos[0] for pos in data['wrist_positions']],
    'right_wrist_y': [pos[1] for pos in data['wrist_positions']],
    # Add other landmarks as needed
})

df_export.to_csv('pose_data_for_r.csv', index=False)
```

### Custom Spike Hand Detection

```python
# Force specific hand
fig, data = analyze_spike_biomechanics(
    'spike.mp4',
    auto_detect_hand=False,
    spike_hand='left'  # Force left-handed analysis
)
```

### Batch Processing Multiple Videos

```python
import glob
import pickle

results_all = {}

for video_path in glob.glob('videos/*.mp4'):
    print(f"\nProcessing: {video_path}")
    
    try:
        fig, data = analyze_spike_biomechanics(
            video_path,
            num_frames=20,
            player_height_cm=185
        )
        
        # Save individual results
        video_name = video_path.split('/')[-1].replace('.mp4', '')
        fig.savefig(f'results/{video_name}_analysis.png', dpi=300)
        results_all[video_name] = data
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

# Save all results
with open('all_spike_analyses.pkl', 'wb') as f:
    pickle.dump(results_all, f)
```

### R: Comparative Analysis

```r
# Analyze multiple athletes
athletes <- c("player1.csv", "player2.csv", "player3.csv")
all_results <- list()

for (athlete_file in athletes) {
  data <- load_pose_data(athlete_file)
  results <- analyze_spike_biomechanics(data, player_height_cm = 180)
  athlete_name <- gsub(".csv", "", basename(athlete_file))
  all_results[[athlete_name]] <- results
  
  print_performance_summary(results)
}

# Compare jump heights
jump_heights <- sapply(all_results, function(r) r$jump_height_cm)
barplot(jump_heights, 
        main = "Jump Height Comparison",
        ylab = "Height (cm)",
        col = "#4ECDC4")
```

---

## 📈 Output Files

### Python

- **Visualization PNG**: Annotated frames + graphs
- **Pickle file**: Complete analysis data for reuse
- **Console output**: Text summary of metrics

### R

- **Visualization PNG**: Multi-panel ggplot graphs
- **CSV files**: 
  - `biomechanics_data.csv`: Frame-by-frame measurements
  - `summary_metrics.csv`: Key performance indicators
  - `feature_matrix.csv`: ML-ready features

---

## 🧪 Example Outputs

### Jump Metrics
```
🚀 JUMP METRICS:
   • Jump Height: 68.3 cm
   • Peak Hip Height at: 1.45s
   • Hip Height Range: 0.523 - 0.891
```

### Arm Mechanics
```
💪 ARM MECHANICS:
   • Max Arm Speed: 12.34 m/s
   • Speed Peak at: 1.52s
   • Max Shoulder Angle: 167.8°
   • Min Elbow Angle: 145.3° (full extension)
```

### Movement Phases
```
🎯 Movement Phases: approach → jump → arm_swing → contact → follow_through
```

---

## 🎯 Use Cases

### For Athletes
- **Performance tracking** - Monitor improvements over time
- **Technique analysis** - Identify areas for improvement
- **Injury prevention** - Detect biomechanical imbalances

### For Coaches
- **Skill assessment** - Objective performance metrics
- **Comparative analysis** - Compare athletes' techniques
- **Training optimization** - Data-driven coaching decisions

### For Researchers
- **Biomechanics research** - Quantitative movement analysis
- **Machine learning** - Train models on spike technique
- **Longitudinal studies** - Track performance evolution

---

## 🔬 Feature Matrix for Machine Learning

The feature extraction produces a 15-dimensional feature vector per frame:

1. **shoulder_angle** - Shoulder joint angle (degrees)
2. **elbow_angle** - Elbow joint angle (degrees)
3. **torso_angle** - Torso lean angle (degrees)
4. **hip_angle** - Hip joint angle (degrees)
5. **knee_angle** - Knee joint angle (degrees)
6. **hip_height** - Normalized hip height (0-1)
7. **shoulder_height** - Normalized shoulder height (0-1)
8. **wrist_height** - Normalized wrist height (0-1)
9. **arm_speed** - Wrist velocity (m/s)
10. **timestamp** - Time (seconds)
11. **confidence** - Pose detection confidence (0-1)
12. **torso_extension** - Shoulder-hip height difference
13. **arm_reach** - Wrist-shoulder height difference
14. **acceleration** - Change in arm speed
15. **vertical_velocity** - Change in wrist height

### Potential ML Applications
- **Technique classification** - Classify spike types
- **Performance prediction** - Predict spike effectiveness
- **Anomaly detection** - Identify unusual movements
- **Sequence modeling** - LSTM/Transformer for temporal patterns

---

## 🐛 Troubleshooting

### Python Issues

**"Cannot open video file"**
```python
# Check video file exists and is readable
import os
print(os.path.exists('spike.mp4'))  # Should print True

# Try different video codecs
# Convert video: ffmpeg -i input.mp4 -c:v libx264 output.mp4
```

**"No valid pose data detected"**
- Ensure person is fully visible in frame
- Check video quality (minimum 720p recommended)
- Increase `num_frames` for better coverage
- Try adjusting `min_detection_confidence` parameter

**Low confidence scores**
- Improve lighting conditions
- Ensure high-contrast clothing
- Remove background clutter
- Use higher resolution video

### R Issues

**"Column not found"**
```r
# Check your CSV column names
colnames(data)

# Ensure they match expected format:
# right_shoulder_x, right_shoulder_y, etc.
```

**Smoothing errors**
```r
# Reduce window length for short sequences
results <- analyze_spike_biomechanics(
  data,
  window_length = 3  # Smaller window
)
```

---

## 📚 Technical Details

### Angle Calculation Method
Uses arctangent-based method to calculate angles between three points:
```
angle = |atan2(C.y - B.y, C.x - B.x) - atan2(A.y - B.y, A.x - B.x)|
```

### Calibration Method
Uses shoulder-to-hip distance as reference:
- Typical shoulder-hip ratio: 27% of total height
- Calculates pixels per cm for real-world measurements
- Applies to all distance/velocity calculations

### Smoothing Algorithm
Savitzky-Golay filter with:
- Window length: 5 frames (default)
- Polynomial order: 2
- Preserves peak features better than moving average

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Multi-person tracking
- Real-time analysis
- 3D pose estimation
- Additional sports (tennis serve, basketball shot, etc.)
- Web-based interface
- Mobile app version

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review example usage code
3. Verify your data format matches specifications
4. Check that all dependencies are installed

---

## 🎓 Citation

If you use this tool in research, please cite:

```bibtex
@software{volleyball_spike_analyzer,
  title={Volleyball Spike Biomechanics Analyzer},
  author={AI-Generated},
  year={2025},
  url={https://github.com/yourusername/volleyball-spike-analyzer}
}
```

---

## 🔄 Version History

**v1.0.0** (December 2025)
- Initial release
- Python version with MediaPipe integration
- R version with comprehensive data analysis
- Automatic spike hand detection
- Movement phase identification
- ML feature extraction

---

**Made with 🏐 for volleyball athletes, coaches, and researchers**

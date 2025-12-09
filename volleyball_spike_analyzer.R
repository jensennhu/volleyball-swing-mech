# Volleyball Spike Biomechanics Analyzer (R Version)
# =====================================================
#
# This R implementation analyzes volleyball spike biomechanics from pre-extracted
# pose landmark data. It calculates joint angles, velocities, jump height, and
# generates comprehensive visualizations.
#
# Author: AI-Generated
# Date: December 2025
#
# Installation:
#   install.packages(c("tidyverse", "ggplot2", "gridExtra", "signal", "pracma"))
#
# Usage:
#   source("volleyball_spike_analyzer.R")
#   data <- load_pose_data("pose_landmarks.csv")
#   results <- analyze_spike_biomechanics(data, player_height_cm = 180)
#   print_performance_summary(results)
#   plot_spike_analysis(results)
#
# Note: This version requires pre-extracted pose landmarks. Use the Python version
#       with MediaPipe to extract pose data from videos first, or use other pose
#       estimation tools that output similar landmark data.

library(tidyverse)
library(ggplot2)
library(gridExtra)
library(signal)  # For filtering
library(pracma)  # For numeric calculations

# ==================== ANGLE CALCULATION ====================

#' Calculate angle at point B formed by three points A, B, C
#' 
#' Uses arctangent method to compute the angle between vectors BA and BC.
#' 
#' @param a Numeric vector of length 2: (x, y) coordinates of point A
#' @param b Numeric vector of length 2: (x, y) coordinates of point B (vertex)
#' @param c Numeric vector of length 2: (x, y) coordinates of point C
#' @return Numeric: Angle in degrees (0-180)
#' 
#' @examples
#' shoulder <- c(100, 200)
#' elbow <- c(150, 250)
#' wrist <- c(200, 300)
#' angle <- calculate_angle(shoulder, elbow, wrist)
#' cat(sprintf("Elbow angle: %.1f degrees\n", angle))
calculate_angle <- function(a, b, c) {
  # Convert to vectors
  a <- as.numeric(a)
  b <- as.numeric(b)
  c <- as.numeric(c)
  
  # Calculate angle using arctangent
  radians <- atan2(c[2] - b[2], c[1] - b[1]) - atan2(a[2] - b[2], a[1] - b[1])
  angle <- abs(radians * 180.0 / pi)
  
  # Normalize to 0-180 range
  if (angle > 180.0) {
    angle <- 360 - angle
  }
  
  return(angle)
}

# ==================== DATA SMOOTHING ====================

#' Apply Savitzky-Golay filter to smooth noisy time-series data
#' 
#' @param data Numeric vector of values to smooth
#' @param window_length Integer: Length of filter window (must be odd)
#' @param poly_order Integer: Polynomial order for fitting
#' @return Numeric vector: Smoothed data
smooth_data <- function(data, window_length = 5, poly_order = 2) {
  if (length(data) < window_length) {
    return(data)
  }
  
  tryCatch({
    # Ensure window_length is odd
    if (window_length %% 2 == 0) {
      window_length <- window_length + 1
    }
    
    # Apply Savitzky-Golay filter
    smoothed <- sgolayfilt(data, p = poly_order, n = window_length)
    return(smoothed)
  }, error = function(e) {
    warning(sprintf("Smoothing failed: %s. Returning original data.", e$message))
    return(data)
  })
}

# ==================== DATA LOADING ====================

#' Load pose landmark data from CSV file
#' 
#' Expected CSV format with columns:
#' frame, timestamp, landmark_name, x, y, visibility
#' 
#' Or wide format with columns:
#' frame, timestamp, right_shoulder_x, right_shoulder_y, right_elbow_x, ...
#' 
#' @param filepath Character: Path to CSV file
#' @return data.frame: Pose landmark data
#' 
#' @examples
#' data <- load_pose_data("pose_landmarks.csv")
load_pose_data <- function(filepath) {
  cat(sprintf("📂 Loading pose data from: %s\n", filepath))
  
  data <- read_csv(filepath, show_col_types = FALSE)
  
  cat(sprintf("✅ Loaded %d rows, %d columns\n", nrow(data), ncol(data)))
  return(data)
}

# ==================== DETECT SPIKE HAND ====================

#' Automatically detect which hand is performing the spike
#' 
#' @param data data.frame: Pose landmark data
#' @return Character: "left" or "right"
detect_spike_hand <- function(data) {
  # Find frame with highest wrist position
  if ("right_wrist_y" %in% colnames(data) && "left_wrist_y" %in% colnames(data)) {
    # Wide format
    avg_right_wrist_y <- mean(data$right_wrist_y, na.rm = TRUE)
    avg_left_wrist_y <- mean(data$left_wrist_y, na.rm = TRUE)
    
    # Lower y = higher in frame
    spike_hand <- ifelse(avg_right_wrist_y < avg_left_wrist_y, "right", "left")
  } else {
    # Long format - default to right
    warning("Could not auto-detect spike hand. Defaulting to 'right'.")
    spike_hand <- "right"
  }
  
  cat(sprintf("🎯 Detected spike hand: %s\n", toupper(spike_hand)))
  return(spike_hand)
}

# ==================== MAIN ANALYSIS FUNCTION ====================

#' Perform comprehensive biomechanical analysis of volleyball spike
#' 
#' @param data data.frame: Pose landmark data (from load_pose_data)
#' @param player_height_cm Numeric: Player's height in centimeters for calibration
#' @param spike_hand Character: "left" or "right", or "auto" for auto-detection
#' @param frame_width Numeric: Video frame width in pixels (for calibration)
#' @param frame_height Numeric: Video frame height in pixels (for calibration)
#' @return list: Comprehensive biomechanics data and metrics
#' 
#' @examples
#' data <- load_pose_data("pose_landmarks.csv")
#' results <- analyze_spike_biomechanics(
#'   data, 
#'   player_height_cm = 180,
#'   spike_hand = "auto"
#' )
analyze_spike_biomechanics <- function(data, 
                                       player_height_cm = 180,
                                       spike_hand = "auto",
                                       frame_width = 1920,
                                       frame_height = 1080) {
  
  cat("\n🏐 Starting volleyball spike biomechanics analysis...\n")
  
  # Auto-detect spike hand if requested
  if (spike_hand == "auto") {
    spike_hand <- detect_spike_hand(data)
  } else {
    spike_hand <- tolower(spike_hand)
  }
  
  # Initialize results list
  results <- list(
    spike_hand = spike_hand,
    player_height_cm = player_height_cm,
    frame_width = frame_width,
    frame_height = frame_height
  )
  
  # Determine column naming convention (assumes wide format)
  shoulder_col <- paste0(spike_hand, "_shoulder")
  elbow_col <- paste0(spike_hand, "_elbow")
  wrist_col <- paste0(spike_hand, "_wrist")
  hip_col <- paste0(spike_hand, "_hip")
  knee_col <- paste0(spike_hand, "_knee")
  ankle_col <- paste0(spike_hand, "_ankle")
  
  # Extract coordinates
  cat("📊 Extracting landmark coordinates...\n")
  
  biomechanics_df <- data %>%
    mutate(
      # Joint angles
      shoulder_angle = pmap_dbl(
        list(get(paste0(hip_col, "_x")), get(paste0(hip_col, "_y")),
             get(paste0(shoulder_col, "_x")), get(paste0(shoulder_col, "_y")),
             get(paste0(elbow_col, "_x")), get(paste0(elbow_col, "_y"))),
        ~ calculate_angle(c(..1, ..2), c(..3, ..4), c(..5, ..6))
      ),
      
      elbow_angle = pmap_dbl(
        list(get(paste0(shoulder_col, "_x")), get(paste0(shoulder_col, "_y")),
             get(paste0(elbow_col, "_x")), get(paste0(elbow_col, "_y")),
             get(paste0(wrist_col, "_x")), get(paste0(wrist_col, "_y"))),
        ~ calculate_angle(c(..1, ..2), c(..3, ..4), c(..5, ..6))
      ),
      
      hip_angle = pmap_dbl(
        list(get(paste0(shoulder_col, "_x")), get(paste0(shoulder_col, "_y")),
             get(paste0(hip_col, "_x")), get(paste0(hip_col, "_y")),
             get(paste0(knee_col, "_x")), get(paste0(knee_col, "_y"))),
        ~ calculate_angle(c(..1, ..2), c(..3, ..4), c(..5, ..6))
      ),
      
      knee_angle = pmap_dbl(
        list(get(paste0(hip_col, "_x")), get(paste0(hip_col, "_y")),
             get(paste0(knee_col, "_x")), get(paste0(knee_col, "_y")),
             get(paste0(ankle_col, "_x")), get(paste0(ankle_col, "_y"))),
        ~ calculate_angle(c(..1, ..2), c(..3, ..4), c(..5, ..6))
      ),
      
      torso_angle = pmap_dbl(
        list(get(paste0(knee_col, "_x")), get(paste0(knee_col, "_y")),
             get(paste0(hip_col, "_x")), get(paste0(hip_col, "_y")),
             get(paste0(shoulder_col, "_x")), get(paste0(shoulder_col, "_y"))),
        ~ calculate_angle(c(..1, ..2), c(..3, ..4), c(..5, ..6))
      ),
      
      # Normalized heights (0 = bottom, 1 = top)
      hip_height = (frame_height - get(paste0(hip_col, "_y"))) / frame_height,
      shoulder_height = (frame_height - get(paste0(shoulder_col, "_y"))) / frame_height,
      wrist_height = (frame_height - get(paste0(wrist_col, "_y"))) / frame_height,
      
      # Wrist positions for velocity calculation
      wrist_x = get(paste0(wrist_col, "_x")),
      wrist_y = get(paste0(wrist_col, "_y"))
    )
  
  cat("🔧 Applying data smoothing...\n")
  
  # Apply smoothing
  biomechanics_df <- biomechanics_df %>%
    mutate(
      shoulder_angle = smooth_data(shoulder_angle),
      elbow_angle = smooth_data(elbow_angle),
      hip_height = smooth_data(hip_height),
      wrist_height = smooth_data(wrist_height)
    )
  
  cat("📏 Calibrating measurements...\n")
  
  # Calibration: Calculate pixels per cm
  avg_shoulder_hip_pixels <- biomechanics_df %>%
    mutate(
      shoulder_hip_dist = sqrt((shoulder_height * frame_height - hip_height * frame_height)^2)
    ) %>%
    pull(shoulder_hip_dist) %>%
    mean(na.rm = TRUE)
  
  shoulder_hip_cm <- player_height_cm * 0.27  # Typical ratio
  pixels_per_cm <- avg_shoulder_hip_pixels / shoulder_hip_cm
  
  cat("⚡ Calculating velocities...\n")
  
  # Calculate arm speeds
  biomechanics_df <- biomechanics_df %>%
    mutate(
      # Calculate wrist displacement between frames
      wrist_dx = c(0, diff(wrist_x)),
      wrist_dy = c(0, diff(wrist_y)),
      pixel_distance = sqrt(wrist_dx^2 + wrist_dy^2),
      
      # Convert to cm and then m/s
      cm_distance = pixel_distance / pixels_per_cm,
      time_diff = c(0, diff(timestamp)),
      arm_speed = ifelse(time_diff > 0, (cm_distance / 100) / time_diff, 0)
    )
  
  # Smooth arm speeds
  biomechanics_df$arm_speed <- smooth_data(biomechanics_df$arm_speed)
  
  cat("🚀 Calculating jump height...\n")
  
  # Calculate jump height
  min_hip_height <- min(biomechanics_df$hip_height, na.rm = TRUE)
  max_hip_height <- max(biomechanics_df$hip_height, na.rm = TRUE)
  jump_height_pixels <- (max_hip_height - min_hip_height) * frame_height
  jump_height_cm <- jump_height_pixels / pixels_per_cm
  
  # Summary metrics
  results$biomechanics_df <- biomechanics_df
  results$jump_height_cm <- jump_height_cm
  results$max_arm_speed <- max(biomechanics_df$arm_speed, na.rm = TRUE)
  results$max_shoulder_angle <- max(biomechanics_df$shoulder_angle, na.rm = TRUE)
  results$min_elbow_angle <- min(biomechanics_df$elbow_angle, na.rm = TRUE)
  results$min_knee_angle <- min(biomechanics_df$knee_angle, na.rm = TRUE)
  
  # Key frame indices
  results$max_shoulder_idx <- which.max(biomechanics_df$shoulder_angle)
  results$max_wrist_height_idx <- which.max(biomechanics_df$wrist_height)
  results$max_hip_height_idx <- which.max(biomechanics_df$hip_height)
  results$max_arm_speed_idx <- which.max(biomechanics_df$arm_speed)
  results$min_elbow_idx <- which.min(biomechanics_df$elbow_angle)
  
  cat("✅ Analysis complete!\n")
  
  return(results)
}

# ==================== PHASE DETECTION ====================

#' Detect movement phases throughout the spike motion
#' 
#' @param results list: Results from analyze_spike_biomechanics()
#' @return Character vector: Phase labels for each frame
detect_spike_phases <- function(results) {
  df <- results$biomechanics_df
  n_frames <- nrow(df)
  
  max_hip_idx <- results$max_hip_height_idx
  max_wrist_idx <- results$max_wrist_height_idx
  max_speed_idx <- results$max_arm_speed_idx
  
  phases <- character(n_frames)
  
  for (i in 1:n_frames) {
    if (i < max_hip_idx * 0.7) {
      phases[i] <- "approach"
    } else if (i < max_hip_idx) {
      phases[i] <- "jump"
    } else if (i < max_speed_idx) {
      phases[i] <- "arm_swing"
    } else if (i <= max_wrist_idx + 1) {
      phases[i] <- "contact"
    } else {
      phases[i] <- "follow_through"
    }
  }
  
  return(phases)
}

# ==================== VISUALIZATION ====================

#' Create comprehensive visualization of spike biomechanics
#' 
#' @param results list: Results from analyze_spike_biomechanics()
#' @param output_file Character: Optional path to save plot
#' @return ggplot object (invisibly)
plot_spike_analysis <- function(results, output_file = NULL) {
  df <- results$biomechanics_df
  
  cat("📊 Creating visualization...\n")
  
  # Plot 1: Joint Angles
  p1 <- ggplot(df, aes(x = timestamp)) +
    geom_line(aes(y = shoulder_angle, color = "Shoulder"), linewidth = 1.2) +
    geom_point(aes(y = shoulder_angle, color = "Shoulder"), size = 2) +
    geom_line(aes(y = elbow_angle, color = "Elbow"), linewidth = 1.2) +
    geom_point(aes(y = elbow_angle, color = "Elbow"), size = 2) +
    geom_hline(yintercept = 180, linetype = "dashed", color = "gray50", alpha = 0.7) +
    annotate("text", x = max(df$timestamp) * 0.9, y = 182, 
             label = "Full Extension", size = 3, color = "gray50") +
    geom_point(data = df[results$max_shoulder_idx, ],
               aes(y = shoulder_angle), 
               color = "red", size = 6, shape = 8) +
    scale_color_manual(values = c("Shoulder" = "#FF6B6B", "Elbow" = "#4ECDC4")) +
    labs(title = "Arm Joint Angles",
         x = "Time (seconds)",
         y = "Angle (degrees)",
         color = "Joint") +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )
  
  # Plot 2: Height Tracking
  p2 <- ggplot(df, aes(x = timestamp)) +
    geom_line(aes(y = wrist_height, color = "Wrist"), linewidth = 1.2) +
    geom_point(aes(y = wrist_height, color = "Wrist"), size = 2) +
    geom_line(aes(y = shoulder_height, color = "Shoulder"), linewidth = 1.2) +
    geom_point(aes(y = shoulder_height, color = "Shoulder"), size = 2) +
    geom_line(aes(y = hip_height, color = "Hip"), linewidth = 1.2) +
    geom_point(aes(y = hip_height, color = "Hip"), size = 2) +
    geom_point(data = df[results$max_wrist_height_idx, ],
               aes(y = wrist_height),
               color = "red", size = 6, shape = 8) +
    scale_color_manual(values = c("Wrist" = "#95E1D3", 
                                   "Shoulder" = "#F38181", 
                                   "Hip" = "#AA96DA")) +
    labs(title = "Body Position Heights (Jump Tracking)",
         x = "Time (seconds)",
         y = "Normalized Height",
         color = "Body Part") +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )
  
  # Plot 3: Arm Speed
  p3 <- ggplot(df, aes(x = timestamp, y = arm_speed)) +
    geom_area(fill = "#FFD93D", alpha = 0.3) +
    geom_line(color = "#FFD93D", linewidth = 1.2) +
    geom_point(color = "#FFD93D", size = 2) +
    geom_point(data = df[results$max_arm_speed_idx, ],
               aes(y = arm_speed),
               color = "red", size = 6, shape = 8) +
    labs(title = "Arm Speed (Wrist Velocity)",
         x = "Time (seconds)",
         y = "Speed (m/s)") +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      panel.grid.minor = element_blank()
    )
  
  # Plot 4: Lower Body Angles
  p4 <- ggplot(df, aes(x = timestamp)) +
    geom_line(aes(y = hip_angle, color = "Hip"), linewidth = 1.2) +
    geom_point(aes(y = hip_angle, color = "Hip"), size = 2) +
    geom_line(aes(y = knee_angle, color = "Knee"), linewidth = 1.2) +
    geom_point(aes(y = knee_angle, color = "Knee"), size = 2) +
    geom_line(aes(y = torso_angle, color = "Torso"), linewidth = 1.2) +
    geom_point(aes(y = torso_angle, color = "Torso"), size = 2) +
    scale_color_manual(values = c("Hip" = "#AA96DA", 
                                   "Knee" = "#4ECDC4",
                                   "Torso" = "#F38181")) +
    labs(title = "Lower Body Angles",
         x = "Time (seconds)",
         y = "Angle (degrees)",
         color = "Joint") +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )
  
  # Combine plots
  combined_plot <- grid.arrange(
    p1, p2, p3, p4,
    ncol = 2,
    top = textGrob(
      sprintf("Volleyball Spike Biomechanics Analysis - %s Hand", 
              toupper(results$spike_hand)),
      gp = gpar(fontface = "bold", fontsize = 16)
    )
  )
  
  # Save if output file specified
  if (!is.null(output_file)) {
    ggsave(output_file, combined_plot, width = 14, height = 10, dpi = 300)
    cat(sprintf("💾 Visualization saved to: %s\n", output_file))
  }
  
  invisible(combined_plot)
}

# ==================== PERFORMANCE SUMMARY ====================

#' Print comprehensive performance summary to console
#' 
#' @param results list: Results from analyze_spike_biomechanics()
print_performance_summary <- function(results) {
  df <- results$biomechanics_df
  
  cat("\n")
  cat(strrep("=", 80), "\n")
  cat("🏐 VOLLEYBALL SPIKE PERFORMANCE SUMMARY\n")
  cat(strrep("=", 80), "\n")
  
  cat("\n📹 VIDEO METRICS:\n")
  cat(sprintf("   • Spike Hand: %s\n", toupper(results$spike_hand)))
  cat(sprintf("   • Frames Analyzed: %d\n", nrow(df)))
  cat(sprintf("   • Duration: %.2fs\n", max(df$timestamp, na.rm = TRUE)))
  
  cat("\n🚀 JUMP METRICS:\n")
  cat(sprintf("   • Jump Height: %.1f cm\n", results$jump_height_cm))
  cat(sprintf("   • Peak Hip Height at: %.2fs\n", 
              df$timestamp[results$max_hip_height_idx]))
  cat(sprintf("   • Hip Height Range: %.3f - %.3f\n",
              min(df$hip_height, na.rm = TRUE),
              max(df$hip_height, na.rm = TRUE)))
  
  cat("\n💪 ARM MECHANICS:\n")
  cat(sprintf("   • Max Arm Speed: %.2f m/s\n", results$max_arm_speed))
  cat(sprintf("   • Speed Peak at: %.2fs\n",
              df$timestamp[results$max_arm_speed_idx]))
  cat(sprintf("   • Max Shoulder Angle: %.1f°\n", results$max_shoulder_angle))
  cat(sprintf("   • Min Elbow Angle: %.1f° (full extension)\n", 
              results$min_elbow_angle))
  cat(sprintf("   • Elbow Extension at: %.2fs\n",
              df$timestamp[results$min_elbow_idx]))
  
  cat("\n🎯 CONTACT POINT:\n")
  contact_idx <- results$max_wrist_height_idx
  cat(sprintf("   • Time: %.2fs\n", df$timestamp[contact_idx]))
  cat(sprintf("   • Frame: %d\n", contact_idx))
  cat(sprintf("   • Wrist Height: %.3f (normalized)\n", 
              df$wrist_height[contact_idx]))
  cat(sprintf("   • Shoulder Angle: %.1f°\n", 
              df$shoulder_angle[contact_idx]))
  cat(sprintf("   • Elbow Angle: %.1f°\n", 
              df$elbow_angle[contact_idx]))
  
  cat("\n🦵 LOWER BODY:\n")
  cat(sprintf("   • Min Knee Angle: %.1f° (max flexion)\n", 
              results$min_knee_angle))
  cat(sprintf("   • Hip Angle Range: %.1f° - %.1f°\n",
              min(df$hip_angle, na.rm = TRUE),
              max(df$hip_angle, na.rm = TRUE)))
  cat(sprintf("   • Torso Angle Range: %.1f° - %.1f°\n",
              min(df$torso_angle, na.rm = TRUE),
              max(df$torso_angle, na.rm = TRUE)))
  
  cat("\n")
  cat(strrep("=", 80), "\n")
}

# ==================== FEATURE EXTRACTION ====================

#' Extract feature matrix for machine learning
#' 
#' @param results list: Results from analyze_spike_biomechanics()
#' @return matrix: Feature matrix (n_frames x n_features)
extract_spike_features <- function(results) {
  df <- results$biomechanics_df
  
  features <- df %>%
    mutate(
      # Derived features
      torso_extension = shoulder_height - hip_height,
      arm_reach = wrist_height - shoulder_height,
      
      # Temporal derivatives (acceleration proxies)
      acceleration = c(0, diff(arm_speed)),
      vertical_velocity = c(0, diff(wrist_height))
    ) %>%
    select(
      shoulder_angle, elbow_angle, torso_angle, hip_angle, knee_angle,
      hip_height, shoulder_height, wrist_height,
      arm_speed, timestamp,
      torso_extension, arm_reach,
      acceleration, vertical_velocity
    ) %>%
    as.matrix()
  
  cat(sprintf("📊 Feature Matrix: %d frames × %d features\n", 
              nrow(features), ncol(features)))
  
  return(features)
}

# ==================== EXPORT FUNCTIONS ====================

#' Export results to CSV files
#' 
#' @param results list: Results from analyze_spike_biomechanics()
#' @param output_dir Character: Directory to save CSV files
export_results <- function(results, output_dir = ".") {
  # Create output directory if needed
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Export biomechanics data
  write_csv(
    results$biomechanics_df,
    file.path(output_dir, "biomechanics_data.csv")
  )
  
  # Export summary metrics
  summary_df <- data.frame(
    metric = c("spike_hand", "player_height_cm", "jump_height_cm",
               "max_arm_speed", "max_shoulder_angle", "min_elbow_angle",
               "min_knee_angle"),
    value = c(results$spike_hand, results$player_height_cm,
              results$jump_height_cm, results$max_arm_speed,
              results$max_shoulder_angle, results$min_elbow_angle,
              results$min_knee_angle)
  )
  
  write_csv(summary_df, file.path(output_dir, "summary_metrics.csv"))
  
  # Export features
  features <- extract_spike_features(results)
  write.csv(features, file.path(output_dir, "feature_matrix.csv"), row.names = FALSE)
  
  cat(sprintf("💾 Results exported to: %s\n", output_dir))
}

# ==================== EXAMPLE USAGE ====================

# Uncomment and modify to run:
#
# # Load pose data
# data <- load_pose_data("pose_landmarks.csv")
#
# # Analyze biomechanics
# results <- analyze_spike_biomechanics(
#   data,
#   player_height_cm = 180,
#   spike_hand = "auto"
# )
#
# # Print summary
# print_performance_summary(results)
#
# # Detect phases
# phases <- detect_spike_phases(results)
# results$biomechanics_df$phase <- phases
# cat(sprintf("\n🎯 Movement Phases: %s\n", 
#             paste(unique(phases), collapse = " → ")))
#
# # Create visualizations
# plot_spike_analysis(results, output_file = "spike_analysis.png")
#
# # Extract features for ML
# features <- extract_spike_features(results)
#
# # Export all results
# export_results(results, output_dir = "spike_analysis_output")
#
# cat("\n✅ Analysis complete!\n")

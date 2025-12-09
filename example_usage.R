#!/usr/bin/env Rscript
# Simple Example: Volleyball Spike Analysis (R Version)
# ======================================================
#
# This script demonstrates basic usage of the R volleyball spike analyzer.
# Modify the parameters below to analyze your own spike data.

source("volleyball_spike_analyzer.R")

# =============================================================================
# CONFIGURATION - Modify these parameters
# =============================================================================

DATA_PATH <- "pose_landmarks.csv"    # Path to your pose data CSV
PLAYER_HEIGHT_CM <- 180              # Player's height in centimeters
SPIKE_HAND <- "auto"                 # "auto", "left", or "right"
OUTPUT_PLOT <- "spike_analysis.png"  # Where to save the visualization
OUTPUT_DIR <- "analysis_output"      # Directory for all output files

# =============================================================================
# ANALYSIS
# =============================================================================

main <- function() {
  cat("\n")
  cat(strrep("=", 70), "\n")
  cat("🏐 Volleyball Spike Biomechanics Analysis (R Version)\n")
  cat(strrep("=", 70), "\n")
  
  # Step 1: Load pose data
  cat("\n📂 LOADING DATA\n")
  cat(strrep("-", 70), "\n")
  
  tryCatch({
    data <- load_pose_data(DATA_PATH)
  }, error = function(e) {
    cat(sprintf("❌ Error loading data: %s\n", e$message))
    cat(sprintf("   Make sure '%s' exists and is properly formatted.\n", DATA_PATH))
    quit(status = 1)
  })
  
  # Step 2: Analyze biomechanics
  cat("\n🔬 ANALYZING BIOMECHANICS\n")
  cat(strrep("-", 70), "\n")
  
  results <- analyze_spike_biomechanics(
    data,
    player_height_cm = PLAYER_HEIGHT_CM,
    spike_hand = SPIKE_HAND
  )
  
  # Step 3: Print performance summary
  print_performance_summary(results)
  
  # Step 4: Detect movement phases
  cat("\n")
  cat(strrep("=", 70), "\n")
  cat("🎯 MOVEMENT PHASE ANALYSIS\n")
  cat(strrep("=", 70), "\n")
  
  phases <- detect_spike_phases(results)
  unique_phases <- unique(phases)
  cat(sprintf("\nDetected phases: %s\n", paste(unique_phases, collapse = " → ")))
  
  # Count frames in each phase
  cat("\nPhase distribution:\n")
  phase_table <- table(phases)
  for (phase_name in names(phase_table)) {
    cat(sprintf("  • %s: %d frames\n", phase_name, phase_table[phase_name]))
  }
  
  # Add phases to results
  results$biomechanics_df$phase <- phases
  
  # Step 5: Extract features
  cat("\n")
  cat(strrep("=", 70), "\n")
  cat("🤖 MACHINE LEARNING FEATURES\n")
  cat(strrep("=", 70), "\n")
  
  features <- extract_spike_features(results)
  cat(sprintf("Feature matrix: %d frames × %d features\n", nrow(features), ncol(features)))
  
  feature_names <- c(
    "shoulder_angle", "elbow_angle", "torso_angle", "hip_angle", "knee_angle",
    "hip_height", "shoulder_height", "wrist_height", "arm_speed", "timestamp",
    "torso_extension", "arm_reach", "acceleration", "vertical_velocity"
  )
  cat("\nFeature list:\n")
  for (i in seq_along(feature_names)) {
    cat(sprintf("  %2d. %s\n", i, feature_names[i]))
  }
  
  # Step 6: Key insights
  cat("\n")
  cat(strrep("=", 70), "\n")
  cat("💡 KEY INSIGHTS\n")
  cat(strrep("=", 70), "\n\n")
  
  # Jump quality assessment
  jump_height <- results$jump_height_cm
  jump_quality <- if (jump_height > 70) {
    "Excellent! 🌟"
  } else if (jump_height > 60) {
    "Very good! 👍"
  } else if (jump_height > 50) {
    "Good 👌"
  } else {
    "Room for improvement 💪"
  }
  cat(sprintf("Jump height: %.1f cm - %s\n", jump_height, jump_quality))
  
  # Arm speed assessment
  arm_speed <- results$max_arm_speed
  speed_quality <- if (arm_speed > 15) {
    "Exceptional! ⚡"
  } else if (arm_speed > 12) {
    "Very fast! 🚀"
  } else if (arm_speed > 10) {
    "Good speed 💨"
  } else {
    "Focus on arm swing velocity 📈"
  }
  cat(sprintf("Max arm speed: %.2f m/s - %s\n", arm_speed, speed_quality))
  
  # Arm extension assessment
  min_elbow <- results$min_elbow_angle
  extension_quality <- if (min_elbow < 150) {
    "Excellent extension! 💪"
  } else if (min_elbow < 160) {
    "Good extension 👍"
  } else {
    "Work on full arm extension 📐"
  }
  cat(sprintf("Minimum elbow angle: %.1f° - %s\n", min_elbow, extension_quality))
  
  # Timing analysis
  cat("\n⏱️  TIMING ANALYSIS\n")
  contact_time <- results$biomechanics_df$timestamp[results$max_wrist_height_idx]
  speed_peak_time <- results$biomechanics_df$timestamp[results$max_arm_speed_idx]
  time_diff <- abs(speed_peak_time - contact_time)
  
  cat(sprintf("Contact point: %.3f seconds\n", contact_time))
  cat(sprintf("Speed peak: %.3f seconds\n", speed_peak_time))
  cat(sprintf("Time difference: %.3f seconds\n", time_diff))
  
  if (time_diff < 0.05) {
    cat("⭐ Excellent timing - speed peaks near contact!\n")
  } else if (time_diff < 0.10) {
    cat("👍 Good timing coordination\n")
  } else {
    cat("💡 Focus on synchronizing speed peak with contact\n")
  }
  
  # Step 7: Create visualizations
  cat("\n")
  cat(strrep("=", 70), "\n")
  cat("📊 CREATING VISUALIZATIONS\n")
  cat(strrep("=", 70), "\n")
  
  plot_spike_analysis(results, output_file = OUTPUT_PLOT)
  
  # Step 8: Export results
  cat("\n")
  cat(strrep("=", 70), "\n")
  cat("💾 EXPORTING RESULTS\n")
  cat(strrep("=", 70), "\n")
  
  export_results(results, output_dir = OUTPUT_DIR)
  
  cat("\n")
  cat(strrep("=", 70), "\n")
  cat("✅ ANALYSIS COMPLETE!\n")
  cat(strrep("=", 70), "\n\n")
  
  cat("Output files:\n")
  cat(sprintf("  • Visualization: %s\n", OUTPUT_PLOT))
  cat(sprintf("  • Data directory: %s/\n", OUTPUT_DIR))
  cat(sprintf("    - biomechanics_data.csv\n"))
  cat(sprintf("    - summary_metrics.csv\n"))
  cat(sprintf("    - feature_matrix.csv\n"))
  cat("\n")
  
  # Return results invisibly for interactive use
  invisible(results)
}

# =============================================================================
# RUN ANALYSIS
# =============================================================================

if (!interactive()) {
  # Running as script
  tryCatch({
    main()
  }, error = function(e) {
    cat("\n")
    cat(strrep("=", 70), "\n")
    cat("❌ ERROR\n")
    cat(strrep("=", 70), "\n")
    cat(sprintf("An error occurred: %s\n", e$message))
    cat("\nTroubleshooting tips:\n")
    cat("  1. Check that your CSV file exists and is properly formatted\n")
    cat("  2. Ensure all required R packages are installed\n")
    cat("  3. Verify column names match the expected format\n")
    cat("  4. See README.md for more troubleshooting help\n")
    cat(strrep("=", 70), "\n\n")
    quit(status = 1)
  })
} else {
  # Interactive mode - just source the functions
  cat("\n📚 Volleyball Spike Analyzer functions loaded!\n")
  cat("\nTo run the example analysis:\n")
  cat("  results <- main()\n")
  cat("\nAvailable functions:\n")
  cat("  • load_pose_data(filepath)\n")
  cat("  • analyze_spike_biomechanics(data, player_height_cm, spike_hand)\n")
  cat("  • print_performance_summary(results)\n")
  cat("  • plot_spike_analysis(results, output_file)\n")
  cat("  • detect_spike_phases(results)\n")
  cat("  • extract_spike_features(results)\n")
  cat("  • export_results(results, output_dir)\n\n")
}

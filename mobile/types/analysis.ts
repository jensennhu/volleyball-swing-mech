/** Types matching the FastAPI /api/mobile/analyze/{id}/result response. */

export interface PhaseRange {
  start: number;
  end: number;
}

export interface KeyFrames {
  approach_end?: number;
  jump_peak?: number;
  contact?: number;
}

export interface JumpHeight {
  peak_frame: number;
  height_px: number;
  standing_y: number;
  peak_y: number;
  height_m?: number;
  height_ft?: number;
}

export interface ReachHeight {
  reach_frame: number;
  reach_height_m: number;
  reach_height_ft: number;
}

export interface Approach {
  start_frame: number;
  end_frame: number;
  steps: number | null;
}

export interface ArmSwing {
  start_frame: number;
  end_frame: number;
  peak_speed_px_per_sec: number;
  peak_frame: number;
  duration_sec: number;
  peak_speed_ms?: number;
  peak_speed_mph?: number;
}

export interface ArmSwingRange {
  start_frame: number;
  end_frame: number;
  range_px: number;
  range_m?: number;
  range_ft?: number;
}

export interface BallContact {
  contact_frame: number;
  ball_center: [number, number];
  wrist?: [number, number];
  distance_px?: number;
}

export interface BboxFrame {
  frame: number;
  bbox: [number, number, number, number];
}

export interface TrackAnalysis {
  track_id: number;
  px_per_meter: number | null;
  phases: Record<string, PhaseRange>;
  key_frames: KeyFrames;
  jump_height: JumpHeight | null;
  ball_contact: BallContact | null;
  reach_height: ReachHeight | null;
  approach: Approach | null;
  arm_swing: ArmSwing | null;
  arm_swing_range: ArmSwingRange | null;
  ball_detections: BboxFrame[];
  track_bboxes: BboxFrame[];
}

export interface AnalysisResult {
  video_id: string;
  fps: number;
  width: number;
  height: number;
  rotation?: number;
  track: TrackAnalysis;
  spike_event_count?: number;
}

export interface SpikeRecord {
  id: string;
  recordedAt: string;
  localUri?: string;
  thumbnailFrame?: number;
  metrics: {
    jumpHeight?: number;
    reachHeight?: number;
    swingSpeed?: number;
    swingRange?: number;
    spikeEvents?: number;
  };
  status: "uploading" | "processing" | "complete" | "error";
  errorMessage?: string;
  progressPct?: number;
  progressMsg?: string;
  serverUrl: string;
}

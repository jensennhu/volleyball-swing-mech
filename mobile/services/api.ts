/** API client for the spike platform backend. */

import Constants from "expo-constants";
import { AnalysisResult } from "../types/analysis";

/**
 * Auto-detect the dev server IP from Expo's hostUri (e.g. "192.168.1.165:8081").
 * Falls back to localhost for simulators / web.
 */
function getBaseUrl(): string {
  const hostUri = Constants.expoConfig?.hostUri; // "192.168.1.165:8081"
  if (hostUri) {
    const host = hostUri.split(":")[0]; // strip Expo's port
    return `http://${host}:8000/api`;
  }
  return "http://localhost:8000/api";
}

export const BASE_URL = getBaseUrl();

interface UploadResponse {
  video_id: string;
  job_id: string;
}

interface StatusResponse {
  status: string;
  progress_pct: number | null;
  message: string | null;
}

export async function uploadVideo(fileUri: string): Promise<UploadResponse> {
  const filename = fileUri.split("/").pop() || "spike.mp4";
  const formData = new FormData();
  formData.append("file", {
    uri: fileUri,
    name: filename,
    type: "video/mp4",
  } as unknown as Blob);

  const res = await fetch(`${BASE_URL}/mobile/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }

  return res.json();
}

export async function pollStatus(videoId: string): Promise<StatusResponse> {
  const res = await fetch(`${BASE_URL}/mobile/analyze/${videoId}/status`);
  if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
  return res.json();
}

export async function getResult(videoId: string): Promise<AnalysisResult> {
  const res = await fetch(`${BASE_URL}/mobile/analyze/${videoId}/result`);

  if (res.status === 202) {
    throw new Error("STILL_PROCESSING");
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Result fetch failed (${res.status}): ${text}`);
  }

  return res.json();
}

export interface ServerSpike {
  video_id: string;
  filename: string;
  recorded_at: string | null;
  thumbnail_frame: number | null;
  metrics: {
    jump_height_m?: number | null;
    reach_height_m?: number | null;
    swing_speed_ms?: number | null;
    swing_range_m?: number | null;
    spike_events?: number | null;
  };
}

export async function fetchSingleEventSpikes(): Promise<ServerSpike[]> {
  const res = await fetch(`${BASE_URL}/mobile/import/single-events`);
  if (!res.ok) throw new Error(`Import fetch failed: ${res.status}`);
  return res.json();
}

export async function reprocessVideo(videoId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/mobile/analyze/${videoId}/reprocess`, {
    method: "POST",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Reprocess failed (${res.status}): ${text}`);
  }
}

export function getClipUrl(videoId: string): string {
  return `${BASE_URL}/videos/${videoId}/clip`;
}

export function getFrameUrl(videoId: string, frameNum: number): string {
  return `${BASE_URL}/videos/${videoId}/frame/${frameNum}`;
}

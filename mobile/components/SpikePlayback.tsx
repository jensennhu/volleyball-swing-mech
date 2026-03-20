/** Video playback component for spike clips with bounding box overlay. */

import { useState, useRef, useMemo, useCallback } from "react";
import { View, Text, StyleSheet, LayoutChangeEvent } from "react-native";
import { Video, ResizeMode, AVPlaybackStatus } from "expo-av";
import { getClipUrl } from "../services/api";
import type { TrackAnalysis } from "../types/analysis";

interface Props {
  videoId: string;
  track: TrackAnalysis;
  meta: { fps: number; width: number; height: number; rotation?: number };
}

/**
 * Transform a raw-frame bbox [x1,y1,x2,y2] into display-space bbox
 * accounting for video rotation metadata.
 *
 * expo-av applies rotation automatically so the displayed video is rotated,
 * but bboxes from the backend are in raw (un-rotated) frame coordinates.
 */
function transformBbox(
  raw: [number, number, number, number],
  rawW: number,
  rawH: number,
  rotation: number,
): [number, number, number, number] {
  const [x1, y1, x2, y2] = raw;
  switch (rotation) {
    case 90:
      // 90° CW: raw(x,y) → display(rawH-y, x); display dims = rawH × rawW
      return [rawH - y2, x1, rawH - y1, x2];
    case 270:
      // 270° CW: raw(x,y) → display(y, rawW-x); display dims = rawH × rawW
      return [y1, rawW - x2, y2, rawW - x1];
    case 180:
      // 180°: raw(x,y) → display(rawW-x, rawH-y); same dims
      return [rawW - x2, rawH - y2, rawW - x1, rawH - y1];
    default:
      return raw;
  }
}

export function SpikePlayback({ videoId, track, meta }: Props) {
  const videoRef = useRef<Video>(null);
  const [currentFrame, setCurrentFrame] = useState(-1);
  const [layoutWidth, setLayoutWidth] = useState(0);

  const rotation = meta.rotation || 0;
  const isRotated = rotation === 90 || rotation === 270;

  // Display dimensions account for rotation (expo-av rotates the video)
  const displayW = isRotated ? meta.height : meta.width;
  const displayH = isRotated ? meta.width : meta.height;
  const aspectRatio = displayW / displayH;
  const displayHeight = layoutWidth > 0 ? layoutWidth / aspectRatio : 0;

  const scaleX = layoutWidth > 0 ? layoutWidth / displayW : 0;
  const scaleY = displayHeight > 0 ? displayHeight / displayH : 0;

  // Compute frame range from phases, falling back to track_bboxes
  const phaseValues = Object.values(track.phases);
  let frameMin = 0;
  let frameMax = 0;
  if (phaseValues.length > 0) {
    frameMin = Math.min(...phaseValues.map((p) => p.start));
    frameMax = Math.max(...phaseValues.map((p) => p.end));
  } else if (track.track_bboxes.length > 0) {
    frameMin = track.track_bboxes[0].frame;
    frameMax = track.track_bboxes[track.track_bboxes.length - 1].frame;
  }
  const startSec = frameMin / meta.fps;
  const endSec = frameMax / meta.fps;
  const durationSec = endSec - startSec;

  // Pre-build lookup maps for O(1) bbox access per frame
  const trackBboxMap = useMemo(() => {
    const m = new Map<number, [number, number, number, number]>();
    for (const b of track.track_bboxes) m.set(b.frame, b.bbox);
    return m;
  }, [track.track_bboxes]);

  const ballBboxMap = useMemo(() => {
    const m = new Map<number, [number, number, number, number]>();
    for (const b of track.ball_detections) m.set(b.frame, b.bbox);
    return m;
  }, [track.ball_detections]);

  const onLayout = (e: LayoutChangeEvent) => {
    setLayoutWidth(e.nativeEvent.layout.width);
  };

  const onPlaybackStatusUpdate = useCallback((status: AVPlaybackStatus) => {
    if (status.isLoaded && status.positionMillis != null) {
      const frame = Math.round((status.positionMillis / 1000) * meta.fps);
      setCurrentFrame(frame);
    }
  }, [meta.fps]);

  // Get raw bboxes and transform for rotation
  const rawTrackBbox = trackBboxMap.get(currentFrame);
  const rawBallBbox = ballBboxMap.get(currentFrame);
  const trackBbox = rawTrackBbox
    ? transformBbox(rawTrackBbox, meta.width, meta.height, rotation)
    : undefined;
  const ballBbox = rawBallBbox
    ? transformBbox(rawBallBbox, meta.width, meta.height, rotation)
    : undefined;

  return (
    <View style={styles.container}>
      <View style={[styles.videoWrapper, { aspectRatio }]} onLayout={onLayout}>
        <Video
          ref={videoRef}
          source={{ uri: getClipUrl(videoId) }}
          style={StyleSheet.absoluteFill}
          resizeMode={ResizeMode.CONTAIN}
          useNativeControls
          positionMillis={startSec * 1000}
          progressUpdateIntervalMillis={100}
          onPlaybackStatusUpdate={onPlaybackStatusUpdate}
        />
        {/* Track bounding box overlay */}
        {scaleX > 0 && trackBbox && (
          <View
            style={[
              styles.bboxTrack,
              {
                left: trackBbox[0] * scaleX,
                top: trackBbox[1] * scaleY,
                width: (trackBbox[2] - trackBbox[0]) * scaleX,
                height: (trackBbox[3] - trackBbox[1]) * scaleY,
              },
            ]}
            pointerEvents="none"
          />
        )}
        {/* Ball bounding box overlay */}
        {scaleX > 0 && ballBbox && (
          <View
            style={[
              styles.bboxBall,
              {
                left: ballBbox[0] * scaleX,
                top: ballBbox[1] * scaleY,
                width: (ballBbox[2] - ballBbox[0]) * scaleX,
                height: (ballBbox[3] - ballBbox[1]) * scaleY,
              },
            ]}
            pointerEvents="none"
          />
        )}
      </View>
      {durationSec > 0 && (
        <Text style={styles.range}>
          Frames {frameMin}–{frameMax} ({durationSec.toFixed(1)}s)
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { marginBottom: 16 },
  videoWrapper: {
    width: "100%",
    borderRadius: 8,
    backgroundColor: "#111",
    overflow: "hidden",
  },
  bboxTrack: {
    position: "absolute",
    borderWidth: 2,
    borderColor: "#22c55e",
    borderRadius: 2,
  },
  bboxBall: {
    position: "absolute",
    borderWidth: 2,
    borderColor: "#facc15",
    borderRadius: 2,
  },
  range: {
    color: "#666",
    fontSize: 11,
    marginTop: 6,
  },
});

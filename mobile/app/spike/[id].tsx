/** Spike detail screen — metrics, key frames, video playback. */

import { useEffect, useState } from "react";
import { View, Text, ScrollView, Image, StyleSheet, ActivityIndicator } from "react-native";
import { useLocalSearchParams } from "expo-router";
import { useSpikeStore } from "../../stores/spikeStore";
import { getResult, getFrameUrl, getClipUrl } from "../../services/api";
import { MetricsCard } from "../../components/MetricsCard";
import { SpikePlayback } from "../../components/SpikePlayback";
import type { TrackAnalysis } from "../../types/analysis";

export default function SpikeDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const record = useSpikeStore((s) => s.getRecord(id));
  const [track, setTrack] = useState<TrackAnalysis | null>(null);
  const [videoMeta, setVideoMeta] = useState<{ fps: number; width: number; height: number; rotation?: number } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    (async () => {
      try {
        const result = await getResult(id);
        setTrack(result.track);
        setVideoMeta({ fps: result.fps, width: result.width, height: result.height, rotation: result.rotation });
      } catch {
        // If still processing or error, track stays null
      } finally {
        setLoading(false);
      }
    })();
  }, [id]);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#6366f1" />
      </View>
    );
  }

  if (!track || !id) {
    return (
      <View style={styles.center}>
        <Text style={styles.emptyText}>No analysis data available</Text>
      </View>
    );
  }

  const dateStr = record?.recordedAt
    ? new Date(record.recordedAt).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {dateStr ? <Text style={styles.date}>{dateStr}</Text> : null}

      <MetricsCard track={track} />

      {/* Key frames — only show section when there are any */}
      {(track.key_frames.approach_end != null ||
        track.key_frames.jump_peak != null ||
        track.key_frames.contact != null) && (
        <>
          <Text style={styles.sectionTitle}>Key Frames</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.framesRow}>
            {track.key_frames.approach_end != null && (
              <FrameThumb videoId={id} frame={track.key_frames.approach_end} label="Approach End" />
            )}
            {track.key_frames.jump_peak != null && (
              <FrameThumb videoId={id} frame={track.key_frames.jump_peak} label="Jump Peak" />
            )}
            {track.key_frames.contact != null && (
              <FrameThumb videoId={id} frame={track.key_frames.contact} label="Ball Contact" />
            )}
          </ScrollView>
        </>
      )}

      {/* Video playback */}
      <Text style={styles.sectionTitle}>Spike Playback</Text>
      <SpikePlayback videoId={id} track={track} meta={videoMeta!} />
    </ScrollView>
  );
}

function FrameThumb({ videoId, frame, label }: { videoId: string; frame: number; label: string }) {
  return (
    <View style={styles.frameThumbnail}>
      <Image
        source={{ uri: getFrameUrl(videoId, frame) }}
        style={styles.frameImage}
        resizeMode="cover"
      />
      <Text style={styles.frameLabel}>{label}</Text>
      <Text style={styles.frameNum}>Frame {frame}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0a" },
  content: { padding: 16, paddingBottom: 40 },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#0a0a0a" },
  emptyText: { color: "#888", fontSize: 15 },
  date: { color: "#888", fontSize: 13, marginBottom: 12 },
  sectionTitle: { color: "#fff", fontSize: 15, fontWeight: "600", marginTop: 20, marginBottom: 10 },
  framesRow: { marginBottom: 8 },
  frameThumbnail: { marginRight: 12, width: 180 },
  frameImage: { width: 180, height: 120, borderRadius: 8, backgroundColor: "#1a1a1a" },
  frameLabel: { color: "#ccc", fontSize: 12, fontWeight: "600", marginTop: 4 },
  frameNum: { color: "#666", fontSize: 11 },
});

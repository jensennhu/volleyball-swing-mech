/** History screen — summary stats, list of past spike records with swipe-to-delete. */

import { useRef, useEffect, useState, useCallback } from "react";
import {
  View,
  Text,
  FlatList,
  Image,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Animated,
  PanResponder,
} from "react-native";
import { useRouter } from "expo-router";
import { useSpikeStore } from "../stores/spikeStore";
import { TrendChart } from "../components/TrendChart";
import { getFrameUrl, getResult, reprocessVideo, fetchSingleEventSpikes, BASE_URL } from "../services/api";
import type { SpikeRecord } from "../types/analysis";

const ACTION_WIDTH = 80;           // width of each action button
const REVEAL_WIDTH = ACTION_WIDTH * 2; // both buttons revealed
const REVEAL_THRESHOLD = -REVEAL_WIDTH / 2; // swipe halfway to snap open

export default function HistoryScreen() {
  const records = useSpikeStore((s) => s.records);
  const deleteRecord = useSpikeStore((s) => s.deleteRecord);
  const addRecord = useSpikeStore((s) => s.addRecord);
  const router = useRouter();
  const [importing, setImporting] = useState(false);
  const [importMsg, setImportMsg] = useState<string | null>(null);

  const handleImport = useCallback(async () => {
    setImporting(true);
    setImportMsg(null);
    try {
      const spikes = await fetchSingleEventSpikes();
      const existingIds = new Set(records.map((r) => r.id));
      const newSpikes = spikes.filter((s) => !existingIds.has(s.video_id));

      for (const s of newSpikes) {
        addRecord({
          id: s.video_id,
          recordedAt: s.recorded_at ?? new Date().toISOString(),
          thumbnailFrame: s.thumbnail_frame ?? undefined,
          metrics: {
            jumpHeight: s.metrics.jump_height_m ?? undefined,
            reachHeight: s.metrics.reach_height_m ?? undefined,
            swingSpeed: s.metrics.swing_speed_ms ?? undefined,
            swingRange: s.metrics.swing_range_m ?? undefined,
            spikeEvents: s.metrics.spike_events ?? undefined,
          },
          status: "complete",
          serverUrl: BASE_URL,
        });
      }

      setImportMsg(
        newSpikes.length > 0
          ? `Imported ${newSpikes.length} spike${newSpikes.length !== 1 ? "s" : ""}`
          : "All up to date"
      );
    } catch {
      setImportMsg("Import failed — check server connection");
    } finally {
      setImporting(false);
    }
  }, [records, addRecord]);

  const completed = records.filter((r) => r.status === "complete");

  if (records.length === 0) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>No spikes recorded yet</Text>
        <Text style={styles.emptyHint}>Record a spike to see your history here</Text>
        <TouchableOpacity
          style={[styles.importBtn, styles.importBtnEmpty, importing && styles.importBtnDisabled]}
          onPress={handleImport}
          disabled={importing}
        >
          <Text style={styles.importBtnText}>
            {importing ? "Importing…" : "Import from server"}
          </Text>
        </TouchableOpacity>
        {importMsg ? <Text style={styles.importMsg}>{importMsg}</Text> : null}
      </View>
    );
  }

  // Summary stats
  const jumpHeights = completed
    .map((r) => r.metrics.jumpHeight)
    .filter((v): v is number => v != null);
  const reachHeights = completed
    .map((r) => r.metrics.reachHeight)
    .filter((v): v is number => v != null);

  const avgJump = jumpHeights.length > 0
    ? jumpHeights.reduce((a, b) => a + b, 0) / jumpHeights.length
    : null;
  const avgReach = reachHeights.length > 0
    ? reachHeights.reduce((a, b) => a + b, 0) / reachHeights.length
    : null;

  // Trend data (chronological order — oldest first)
  const jumpData = completed
    .filter((r) => r.metrics.jumpHeight != null)
    .map((r) => ({ date: r.recordedAt, value: r.metrics.jumpHeight! }))
    .reverse();

  const speedData = completed
    .filter((r) => r.metrics.swingSpeed != null)
    .map((r) => ({ date: r.recordedAt, value: r.metrics.swingSpeed! }))
    .reverse();

  const confirmDelete = (record: SpikeRecord) => {
    Alert.alert(
      "Delete Spike",
      "Remove this spike record? This cannot be undone.",
      [
        { text: "Cancel", style: "cancel" },
        { text: "Delete", style: "destructive", onPress: () => deleteRecord(record.id) },
      ]
    );
  };

  return (
    <FlatList
      style={styles.container}
      contentContainerStyle={styles.content}
      ListHeaderComponent={
        <>
          {/* Summary stats */}
          <View style={styles.statsRow}>
            <StatBox label="Total Spikes" value={String(completed.length)} />
            <StatBox
              label="Avg Jump"
              value={avgJump != null ? `${avgJump.toFixed(2)}m` : "--"}
            />
            <StatBox
              label="Avg Reach"
              value={avgReach != null ? `${avgReach.toFixed(2)}m` : "--"}
            />
          </View>

          {jumpData.length >= 2 && (
            <TrendChart title="Jump Height (m)" data={jumpData} color="#22d3ee" />
          )}
          {speedData.length >= 2 && (
            <TrendChart title="Arm Swing Speed (m/s)" data={speedData} color="#6366f1" />
          )}
          <View style={styles.listHeaderRow}>
            <Text style={styles.listHeader}>
              {records.length} spike{records.length !== 1 ? "s" : ""} recorded
            </Text>
            <TouchableOpacity
              style={[styles.importBtn, importing && styles.importBtnDisabled]}
              onPress={handleImport}
              disabled={importing}
            >
              <Text style={styles.importBtnText}>
                {importing ? "Importing…" : "Import from server"}
              </Text>
            </TouchableOpacity>
          </View>
          {importMsg ? <Text style={styles.importMsg}>{importMsg}</Text> : null}
        </>
      }
      data={records}
      keyExtractor={(item) => item.id}
      renderItem={({ item }) => (
        <SwipeableRecordCard
          record={item}
          onPress={() => router.push(`/spike/${item.id}`)}
          onDelete={() => confirmDelete(item)}
        />
      )}
    />
  );
}

function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.statBox}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

function SwipeableRecordCard({
  record,
  onPress,
  onDelete,
}: {
  record: SpikeRecord;
  onPress: () => void;
  onDelete: () => void;
}) {
  const updateRecord = useSpikeStore((s) => s.updateRecord);
  const translateX = useRef(new Animated.Value(0)).current;

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gesture) =>
        Math.abs(gesture.dx) > 10 && Math.abs(gesture.dx) > Math.abs(gesture.dy),
      onPanResponderMove: (_, gesture) => {
        if (gesture.dx < 0) {
          translateX.setValue(Math.max(gesture.dx, -REVEAL_WIDTH - 20));
        }
      },
      onPanResponderRelease: (_, gesture) => {
        if (gesture.dx < REVEAL_THRESHOLD) {
          Animated.spring(translateX, {
            toValue: -REVEAL_WIDTH,
            useNativeDriver: true,
          }).start();
        } else {
          Animated.spring(translateX, {
            toValue: 0,
            useNativeDriver: true,
          }).start();
        }
      },
    })
  ).current;

  const close = () => {
    Animated.spring(translateX, { toValue: 0, useNativeDriver: true }).start();
  };

  const handleReprocess = async () => {
    close();
    try {
      await reprocessVideo(record.id);
      updateRecord(record.id, {
        status: "processing",
        progressPct: undefined,
        progressMsg: undefined,
        thumbnailFrame: undefined,
      });
    } catch {
      Alert.alert("Error", "Could not reprocess — video may no longer be on the server.");
    }
  };

  return (
    <View style={styles.swipeContainer}>
      {/* Action buttons sit behind the card */}
      <View style={styles.actionRow}>
        <TouchableOpacity style={styles.reprocessBtn} onPress={handleReprocess}>
          <Text style={styles.actionText}>Reprocess</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.deleteBtn}
          onPress={() => { close(); onDelete(); }}
        >
          <Text style={styles.actionText}>Delete</Text>
        </TouchableOpacity>
      </View>

      {/* Card slides left on swipe */}
      <Animated.View
        style={{ transform: [{ translateX }] }}
        {...panResponder.panHandlers}
      >
        <RecordCard record={record} onPress={onPress} />
      </Animated.View>
    </View>
  );
}

function RecordCard({ record, onPress }: { record: SpikeRecord; onPress: () => void }) {
  const updateRecord = useSpikeStore((s) => s.updateRecord);

  // Backfill thumbnailFrame for records that completed before this field was added
  useEffect(() => {
    if (record.status === "complete" && record.thumbnailFrame == null) {
      getResult(record.id)
        .then((result) => {
          const frame =
            result.track.key_frames.contact ?? result.track.key_frames.jump_peak;
          if (frame != null) {
            updateRecord(record.id, { thumbnailFrame: frame });
          }
        })
        .catch(() => {});
    }
  }, [record.id, record.status, record.thumbnailFrame, updateRecord]);

  const date = new Date(record.recordedAt).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  const statusColor =
    record.status === "complete" ? "#22c55e" :
    record.status === "error" ? "#ef4444" :
    "#f59e0b";

  const thumbUri =
    record.status === "complete" && record.thumbnailFrame != null
      ? getFrameUrl(record.id, record.thumbnailFrame)
      : null;

  return (
    <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.cardInner}>
        <View style={styles.cardLeft}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardDate}>{date}</Text>
            <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
          </View>

          {record.status === "complete" && (
            <View style={styles.cardMetrics}>
              {record.metrics.jumpHeight != null && (
                <MetricChip label="Jump" value={`${record.metrics.jumpHeight.toFixed(2)}m`} />
              )}
              {record.metrics.swingSpeed != null && (
                <MetricChip label="Speed" value={`${record.metrics.swingSpeed.toFixed(1)}m/s`} />
              )}
              {record.metrics.spikeEvents != null && (
                <MetricChip label="Spikes" value={String(record.metrics.spikeEvents)} />
              )}
            </View>
          )}

          {record.status === "processing" && (
            <View style={styles.progressContainer}>
              <View style={styles.progressBarOuter}>
                <View
                  style={[styles.progressBarFill, { width: `${record.progressPct ?? 5}%` }]}
                />
              </View>
              <Text style={styles.processingText}>{record.progressMsg || "Processing..."}</Text>
            </View>
          )}
          {record.status === "error" && <Text style={styles.errorText}>{record.errorMessage || "Failed"}</Text>}
        </View>

        {thumbUri && (
          <Image
            source={{ uri: thumbUri }}
            style={styles.thumbnail}
            resizeMode="cover"
          />
        )}
      </View>
    </TouchableOpacity>
  );
}

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.chip}>
      <Text style={styles.chipLabel}>{label}</Text>
      <Text style={styles.chipValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0a" },
  content: { padding: 16, paddingBottom: 40 },
  empty: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#0a0a0a" },
  emptyText: { color: "#888", fontSize: 16 },
  emptyHint: { color: "#555", fontSize: 13, marginTop: 6 },

  // Summary stats
  statsRow: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 16,
  },
  statBox: {
    flex: 1,
    backgroundColor: "#141414",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#222",
    padding: 12,
    alignItems: "center",
  },
  statValue: { color: "#fff", fontSize: 20, fontWeight: "700" },
  statLabel: { color: "#666", fontSize: 11, fontWeight: "600", marginTop: 4 },

  listHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 16,
    marginBottom: 8,
  },
  listHeader: { color: "#888", fontSize: 13 },
  importBtn: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#6366f1",
  },
  importBtnEmpty: { marginTop: 20 },
  importBtnDisabled: { opacity: 0.5 },
  importBtnText: { color: "#6366f1", fontSize: 12, fontWeight: "600" },
  importMsg: { color: "#888", fontSize: 12, marginBottom: 4, marginTop: 2 },

  // Swipe container
  swipeContainer: {
    marginBottom: 10,
    overflow: "hidden",
    borderRadius: 12,
  },

  // Cards
  card: {
    backgroundColor: "#141414",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#222",
    padding: 14,
  },
  cardInner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  cardLeft: { flex: 1 },
  thumbnail: {
    width: 68,
    height: 90,
    borderRadius: 8,
    backgroundColor: "#1a1a1a",
    flexShrink: 0,
  },
  cardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  cardDate: { color: "#ccc", fontSize: 14, fontWeight: "500" },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  cardMetrics: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 10 },
  chip: {
    backgroundColor: "#1a1a1a",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
  },
  chipLabel: { color: "#666", fontSize: 10, fontWeight: "600" },
  chipValue: { color: "#fff", fontSize: 14, fontWeight: "600", marginTop: 2 },
  progressContainer: { marginTop: 10, gap: 6 },
  progressBarOuter: {
    height: 6,
    borderRadius: 3,
    backgroundColor: "#222",
    overflow: "hidden",
  },
  progressBarFill: {
    height: "100%",
    borderRadius: 3,
    backgroundColor: "#6366f1",
  },
  processingText: { color: "#f59e0b", fontSize: 12 },
  errorText: { color: "#ef4444", fontSize: 13, marginTop: 8 },

  // Swipe actions
  actionRow: {
    position: "absolute",
    right: 0,
    top: 0,
    bottom: 0,
    flexDirection: "row",
    width: REVEAL_WIDTH,
  },
  reprocessBtn: {
    backgroundColor: "#6366f1",
    justifyContent: "center",
    alignItems: "center",
    width: ACTION_WIDTH,
    height: "100%",
  },
  deleteBtn: {
    backgroundColor: "#ef4444",
    justifyContent: "center",
    alignItems: "center",
    width: ACTION_WIDTH,
    height: "100%",
    borderTopRightRadius: 12,
    borderBottomRightRadius: 12,
  },
  actionText: { color: "#fff", fontSize: 13, fontWeight: "600" },
});

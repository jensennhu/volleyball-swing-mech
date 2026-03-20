/** Collection screen — Instagram-style grid of completed spike key frames. */

import { useEffect } from "react";
import {
  View,
  Text,
  FlatList,
  Image,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
} from "react-native";
import { useRouter } from "expo-router";
import { useSpikeStore } from "../stores/spikeStore";
import { getFrameUrl, getResult } from "../services/api";
import type { SpikeRecord } from "../types/analysis";

const COLUMNS = 3;
const GAP = 2;
const CELL_SIZE =
  (Dimensions.get("window").width - GAP * (COLUMNS - 1)) / COLUMNS;

export default function CollectionScreen() {
  const records = useSpikeStore((s) => s.records);
  const router = useRouter();

  const completed = records.filter((r) => r.status === "complete");

  if (completed.length === 0) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>No spikes yet</Text>
        <Text style={styles.emptyHint}>Completed spikes will appear here</Text>
      </View>
    );
  }

  return (
    <FlatList
      data={completed}
      keyExtractor={(item) => item.id}
      numColumns={COLUMNS}
      style={styles.container}
      columnWrapperStyle={{ gap: GAP }}
      ItemSeparatorComponent={() => <View style={{ height: GAP }} />}
      renderItem={({ item }) => (
        <GridCell
          record={item}
          onPress={() => router.push(`/spike/${item.id}`)}
        />
      )}
    />
  );
}

function GridCell({
  record,
  onPress,
}: {
  record: SpikeRecord;
  onPress: () => void;
}) {
  const updateRecord = useSpikeStore((s) => s.updateRecord);

  // Lazily backfill thumbnailFrame for older records
  useEffect(() => {
    if (record.thumbnailFrame == null) {
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
  }, [record.id, record.thumbnailFrame, updateRecord]);

  const uri =
    record.thumbnailFrame != null
      ? getFrameUrl(record.id, record.thumbnailFrame)
      : null;

  return (
    <TouchableOpacity style={styles.cell} onPress={onPress} activeOpacity={0.85}>
      {uri ? (
        <Image source={{ uri }} style={styles.cellImage} resizeMode="cover" />
      ) : (
        <View style={styles.cellPlaceholder} />
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0a" },
  empty: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#0a0a0a",
  },
  emptyText: { color: "#888", fontSize: 16 },
  emptyHint: { color: "#555", fontSize: 13, marginTop: 6 },
  cell: { width: CELL_SIZE, height: CELL_SIZE },
  cellImage: { width: CELL_SIZE, height: CELL_SIZE },
  cellPlaceholder: {
    width: CELL_SIZE,
    height: CELL_SIZE,
    backgroundColor: "#141414",
  },
});

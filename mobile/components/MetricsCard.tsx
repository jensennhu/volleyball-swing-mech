/** Metrics card — displays the 5 biomechanical metrics for a spike. */

import { View, Text, StyleSheet } from "react-native";
import type { TrackAnalysis } from "../types/analysis";

interface Props {
  track: TrackAnalysis;
}

interface MetricRow {
  label: string;
  value: string;
  secondary?: string;
}

export function MetricsCard({ track }: Props) {
  const rows: MetricRow[] = [
    // Jump Height
    track.jump_height?.height_m != null
      ? {
          label: "Jump Height",
          value: `${track.jump_height.height_m.toFixed(2)} m`,
          secondary: track.jump_height.height_ft != null ? `${track.jump_height.height_ft.toFixed(1)} ft` : undefined,
        }
      : track.jump_height?.height_px != null
        ? { label: "Jump Height", value: `${track.jump_height.height_px.toFixed(0)} px` }
        : { label: "Jump Height", value: "--" },

    // Reach Height
    track.reach_height
      ? {
          label: "Reach Height",
          value: `${track.reach_height.reach_height_m.toFixed(2)} m`,
          secondary: `${track.reach_height.reach_height_ft.toFixed(1)} ft`,
        }
      : { label: "Reach Height", value: "--" },

    // Arm Swing Speed
    track.arm_swing?.peak_speed_ms != null
      ? {
          label: "Arm Swing Speed",
          value: `${track.arm_swing.peak_speed_ms.toFixed(1)} m/s`,
          secondary: track.arm_swing.peak_speed_mph != null ? `${track.arm_swing.peak_speed_mph.toFixed(0)} mph` : undefined,
        }
      : track.arm_swing?.peak_speed_px_per_sec != null
        ? { label: "Arm Swing Speed", value: `${track.arm_swing.peak_speed_px_per_sec.toFixed(0)} px/s` }
        : { label: "Arm Swing Speed", value: "--" },

    // Arm Swing Range
    track.arm_swing_range?.range_m != null
      ? {
          label: "Arm Swing Range",
          value: `${track.arm_swing_range.range_m.toFixed(2)} m`,
          secondary: track.arm_swing_range.range_ft != null ? `${track.arm_swing_range.range_ft.toFixed(1)} ft` : undefined,
        }
      : track.arm_swing_range?.range_px != null
        ? { label: "Arm Swing Range", value: `${track.arm_swing_range.range_px.toFixed(0)} px` }
        : { label: "Arm Swing Range", value: "--" },
  ];

  return (
    <View style={styles.card}>
      {rows.map((row, i) => (
        <View key={i} style={[styles.row, i < rows.length - 1 && styles.rowBorder]}>
          <Text style={styles.label}>{row.label}</Text>
          <View style={styles.values}>
            <Text style={[styles.value, row.value === "--" && styles.valueMissing]}>
              {row.value}
            </Text>
            {row.secondary ? <Text style={styles.secondary}>{row.secondary}</Text> : null}
          </View>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#141414",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#222",
    padding: 16,
  },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
  },
  rowBorder: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#222",
  },
  label: { color: "#aaa", fontSize: 14 },
  values: { alignItems: "flex-end" },
  value: { color: "#fff", fontSize: 16, fontWeight: "600" },
  valueMissing: { color: "#444" },
  secondary: { color: "#666", fontSize: 12, marginTop: 2 },
});

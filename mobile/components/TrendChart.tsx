/** Simple trend line chart for historical metrics. */

import { View, Text, StyleSheet } from "react-native";
import Svg, { Polyline, Circle, Line } from "react-native-svg";

interface DataPoint {
  date: string;
  value: number;
}

interface Props {
  title: string;
  data: DataPoint[];
  color: string;
}

const CHART_W = 320;
const CHART_H = 120;
const PAD_X = 32;
const PAD_Y = 16;

export function TrendChart({ title, data, color }: Props) {
  if (data.length < 2) return null;

  const values = data.map((d) => d.value);
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const rangeV = maxV - minV || 1;

  const plotW = CHART_W - PAD_X * 2;
  const plotH = CHART_H - PAD_Y * 2;

  const points = data.map((d, i) => {
    const x = PAD_X + (i / (data.length - 1)) * plotW;
    const y = PAD_Y + plotH - ((d.value - minV) / rangeV) * plotH;
    return { x, y };
  });

  const polylinePoints = points.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{title}</Text>
      <Svg width={CHART_W} height={CHART_H} viewBox={`0 0 ${CHART_W} ${CHART_H}`}>
        {/* Grid lines */}
        <Line x1={PAD_X} y1={PAD_Y} x2={PAD_X} y2={CHART_H - PAD_Y} stroke="#333" strokeWidth={1} />
        <Line x1={PAD_X} y1={CHART_H - PAD_Y} x2={CHART_W - PAD_X} y2={CHART_H - PAD_Y} stroke="#333" strokeWidth={1} />

        {/* Trend line */}
        <Polyline points={polylinePoints} fill="none" stroke={color} strokeWidth={2} />

        {/* Data points */}
        {points.map((p, i) => (
          <Circle key={i} cx={p.x} cy={p.y} r={4} fill={color} />
        ))}
      </Svg>

      {/* Y-axis labels */}
      <View style={styles.yLabels}>
        <Text style={styles.yLabel}>{maxV.toFixed(2)}</Text>
        <Text style={styles.yLabel}>{minV.toFixed(2)}</Text>
      </View>

      {/* Latest value callout */}
      <Text style={[styles.latest, { color }]}>
        Latest: {data[data.length - 1].value.toFixed(2)}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: "#141414",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#222",
    padding: 14,
    marginBottom: 12,
  },
  title: { color: "#aaa", fontSize: 13, fontWeight: "600", marginBottom: 8 },
  yLabels: {
    position: "absolute",
    top: 38,
    left: 6,
    height: CHART_H - PAD_Y * 2,
    justifyContent: "space-between",
  },
  yLabel: { color: "#555", fontSize: 9 },
  latest: { fontSize: 12, fontWeight: "600", marginTop: 6 },
});

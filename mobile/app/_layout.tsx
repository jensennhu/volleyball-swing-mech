import { useEffect, useRef, useMemo } from "react";
import { Tabs } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useSpikeStore } from "../stores/spikeStore";
import { pollStatus, getResult } from "../services/api";

const TINT = "#6366f1";
const BG = "#0a0a0a";

/** Polls all in-flight "processing" records in the background. */
function BackgroundPoller() {
  const records = useSpikeStore((s) => s.records);
  const updateRecord = useSpikeStore((s) => s.updateRecord);
  const pollingRef = useRef(new Set<string>());

  // Stable ref so the poll closure always calls the latest updater
  const updateRef = useRef(updateRecord);
  useEffect(() => {
    updateRef.current = updateRecord;
  }, [updateRecord]);

  // Re-run only when the set of processing IDs changes
  const processingKey = useMemo(
    () =>
      records
        .filter((r) => r.status === "processing")
        .map((r) => r.id)
        .join(","),
    [records]
  );

  useEffect(() => {
    const processing = records.filter((r) => r.status === "processing");

    for (const record of processing) {
      if (pollingRef.current.has(record.id)) continue;
      pollingRef.current.add(record.id);

      const poll = async () => {
        try {
          const status = await pollStatus(record.id);

          if (status.status === "error") {
            updateRef.current(record.id, {
              status: "error",
              errorMessage: status.message || "Processing failed",
            });
            pollingRef.current.delete(record.id);
            return;
          }

          if (status.status === "processed" || status.status === "predicted") {
            try {
              const result = await getResult(record.id);
              const track = result.track;
              updateRef.current(record.id, {
                status: "complete",
                thumbnailFrame:
                  track.key_frames.contact ??
                  track.key_frames.jump_peak ??
                  undefined,
                metrics: {
                  jumpHeight: track.jump_height?.height_m,
                  reachHeight: track.reach_height?.reach_height_m,
                  swingSpeed: track.arm_swing?.peak_speed_ms,
                  swingRange: track.arm_swing_range?.range_m,
                  spikeEvents: result.spike_event_count,
                },
              });
            } catch {
              updateRef.current(record.id, {
                status: "error",
                errorMessage: "Failed to fetch results",
              });
            }
            pollingRef.current.delete(record.id);
            return;
          }

          // Still processing — update progress and schedule next poll
          updateRef.current(record.id, {
            progressPct: status.progress_pct ?? undefined,
            progressMsg: status.message ?? undefined,
          });
          setTimeout(poll, 3000);
        } catch {
          // Network error — back off and retry
          setTimeout(poll, 5000);
        }
      };

      poll();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [processingKey]);

  return null;
}

export default function RootLayout() {
  return (
    <>
      <StatusBar style="light" />
      <BackgroundPoller />
      <Tabs
        initialRouteName="history"
        screenOptions={{
          tabBarActiveTintColor: TINT,
          tabBarInactiveTintColor: "#666",
          tabBarStyle: { backgroundColor: BG, borderTopColor: "#222" },
          headerStyle: { backgroundColor: BG },
          headerTintColor: "#fff",
        }}
      >
        <Tabs.Screen
          name="history"
          options={{ title: "Home", tabBarLabel: "Home" }}
        />
        <Tabs.Screen
          name="collection"
          options={{ title: "Collection", tabBarLabel: "Collection" }}
        />
        <Tabs.Screen
          name="index"
          options={{ title: "Record", tabBarLabel: "Record" }}
        />
        <Tabs.Screen
          name="spike/[id]"
          options={{ href: null }}
        />
      </Tabs>
    </>
  );
}

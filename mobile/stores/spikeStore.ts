/** Persistent store for spike records using Zustand + AsyncStorage. */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { SpikeRecord } from "../types/analysis";

interface SpikeStore {
  records: SpikeRecord[];
  addRecord: (record: SpikeRecord) => void;
  updateRecord: (id: string, updates: Partial<SpikeRecord>) => void;
  deleteRecord: (id: string) => void;
  getRecord: (id: string) => SpikeRecord | undefined;
}

export const useSpikeStore = create<SpikeStore>()(
  persist(
    (set, get) => ({
      records: [],

      addRecord: (record) =>
        set((state) => ({
          records: [record, ...state.records],
        })),

      updateRecord: (id, updates) =>
        set((state) => ({
          records: state.records.map((r) =>
            r.id === id ? { ...r, ...updates } : r
          ),
        })),

      deleteRecord: (id) =>
        set((state) => ({
          records: state.records.filter((r) => r.id !== id),
        })),

      getRecord: (id) => get().records.find((r) => r.id === id),
    }),
    {
      name: "spike-records",
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);

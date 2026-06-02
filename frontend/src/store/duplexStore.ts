/**
 * duplexStore — results for a two-system (A + B) run. Kept separate from the
 * single-system pipelineStore/resultsStore so the existing one-system path is
 * untouched. A duplex run executes two pipeline passes (A then B); each pass's
 * final metrics land here, shown side by side.
 */

import { create } from "zustand";

import type { SystemId } from "@/store/configStore";

export interface SysMetrics {
  engine?: string;
  ber: number | null;
  snr_est_dB: number | null;
  P_rx_avg_uW: number | null;
}

interface DuplexState {
  running: boolean;
  current: SystemId | null; // which pass is in flight
  results: Record<SystemId, SysMetrics | null>;
  error: string | null;

  start: () => void;
  setCurrent: (s: SystemId | null) => void;
  setResult: (s: SystemId, m: SysMetrics) => void;
  finish: () => void;
  fail: (e: string) => void;
  clear: () => void;
}

const empty = { A: null, B: null } as Record<SystemId, SysMetrics | null>;

export const useDuplexStore = create<DuplexState>((set) => ({
  running: false,
  current: null,
  results: { ...empty },
  error: null,

  start: () => set({ running: true, current: null, results: { ...empty }, error: null }),
  setCurrent: (s) => set({ current: s }),
  setResult: (s, m) => set((st) => ({ results: { ...st.results, [s]: m } })),
  finish: () => set({ running: false, current: null }),
  fail: (e) => set({ running: false, current: null, error: e }),
  clear: () => set({ running: false, current: null, results: { ...empty }, error: null }),
}));

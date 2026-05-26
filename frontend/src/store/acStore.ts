/**
 * acStore — small-signal AC analysis result. One result at a time;
 * subsequent runs replace.
 */

import { create } from "zustand";

export interface AcResult {
  frequency_hz: number[];
  magnitude_db: number[];
  phase_deg: number[];
  gain_margin_db: number | null;
  phase_margin_deg: number | null;
  unity_gain_freq_hz: number | null;
  crossover_180_freq_hz: number | null;
  f_low_3dB_hz: number | null;
  f_high_3dB_hz: number | null;
  midband_gain_db: number;
}

interface AcState {
  running: boolean;
  result: AcResult | null;
  error: string | null;
  start: () => void;
  setResult: (r: AcResult) => void;
  fail: (message: string) => void;
  reset: () => void;
}

export const useAcStore = create<AcState>((set) => ({
  running: false,
  result: null,
  error: null,
  start: () => set({ running: true, error: null }),
  setResult: (r) => set({ result: r, running: false, error: null }),
  fail: (message) => set({ running: false, error: message }),
  reset: () => set({ result: null, error: null, running: false }),
}));

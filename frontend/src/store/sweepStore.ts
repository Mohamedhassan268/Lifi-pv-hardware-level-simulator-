/**
 * sweepStore — points stream in live from /ws/sweep. The hook appends to the
 * array; Plotly charts subscribe and re-render only when the array identity
 * changes.
 *
 * The plan calls for a module-scope ring buffer with only the cursor in the
 * store to absorb 60 Hz sweep frames. For thermal + MC at <2 runs/sec, an
 * append-to-array is fine; we keep the design ready to swap if perf bites.
 */

import { create } from "zustand";

export type SweepKind = "thermal" | "montecarlo" | "ber_snr" | "ber_distance";

export interface ThermalPoint {
  temperature_C: number;
  temperature_K: number;
  dark_current_A: number;
  responsivity_A_per_W: number;
  I_ph_avg_uA: number;
  snr_dB: number;
  ber: number;
}

export interface MonteCarloSample {
  i: number;
  ber: number;
  snr_dB: number;
  P_rx_avg_uW: number;
  overrides: Record<string, number>;
}

export interface BerSweepPoint {
  x_label: string;        // "ambient_lux" or "distance_m"
  x_value: number;
  ber: number;
  n_errors: number;
  n_bits: number;
  snr_est_dB: number;
  ci_low: number;
  ci_high: number;
  modulation?: string;
}

export interface ToleranceRow {
  field: string;
  fractional: number;
  source: string;     // "datasheet" | "process" | "estimate"
  via: string;        // which component supplied the spec
}

export type SweepPoint = ThermalPoint | MonteCarloSample | BerSweepPoint;

interface SweepState {
  kind: SweepKind | null;
  running: boolean;
  total: number;
  points: SweepPoint[];
  error: string | null;
  toleranceRows: ToleranceRow[] | null;

  start: (kind: SweepKind, total: number) => void;
  pushPoint: (p: SweepPoint) => void;
  finish: () => void;
  fail: (message: string) => void;
  reset: () => void;
  setToleranceRows: (rows: ToleranceRow[]) => void;
}

export const useSweepStore = create<SweepState>((set) => ({
  kind: null,
  running: false,
  total: 0,
  points: [],
  error: null,
  toleranceRows: null,

  start: (kind, total) =>
    set({ kind, running: true, total, points: [], error: null, toleranceRows: null }),
  pushPoint: (p) => set((s) => ({ points: [...s.points, p] })),
  finish: () => set({ running: false }),
  fail: (message) => set({ running: false, error: message }),
  reset: () =>
    set({ kind: null, running: false, total: 0, points: [], error: null, toleranceRows: null }),
  setToleranceRows: (rows) => set({ toleranceRows: rows }),
}));

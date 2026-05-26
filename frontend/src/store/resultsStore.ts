/**
 * resultsStore — holds the metrics + waveforms received over the WebSocket.
 * Plotly canvases subscribe via narrow selectors so unrelated updates
 * (e.g. step status changes) don't re-render charts.
 */

import { create } from "zustand";

export interface NoiseBreakdown {
  shot_A2: number;
  thermal_A2: number;
  ambient_A2: number;
  amplifier_A2: number;
  adc_A2: number;
  processing_A2: number;
  total_A2: number;
  total_std_A: number;
}

export interface Metrics {
  engine?: string;
  ber?: number | null;
  ber_n_errors?: number | null;
  ber_n_bits?: number | null;
  snr_est_dB?: number | null;
  P_rx_avg_uW?: number | null;
  I_ph_avg_uA?: number | null;
  G_ch?: number | null;
  noise_breakdown?: NoiseBreakdown | null;
}

export interface Waveforms {
  time: number[];
  P_tx: number[];
  P_rx: number[];
  I_ph: number[];
  V_rx: number[];
  bits_tx: number[];
  bits_rx: number[];
  modulation?: string;
}

interface ResultsState {
  metrics: Metrics | null;
  waveforms: Waveforms | null;
  setMetrics: (m: Metrics) => void;
  setWaveforms: (w: Waveforms) => void;
  clear: () => void;
}

export const useResultsStore = create<ResultsState>((set) => ({
  metrics: null,
  waveforms: null,
  setMetrics: (m) => set({ metrics: m }),
  setWaveforms: (w) => set({ waveforms: w }),
  clear: () => set({ metrics: null, waveforms: null }),
}));

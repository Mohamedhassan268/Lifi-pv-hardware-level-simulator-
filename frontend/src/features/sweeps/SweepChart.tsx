/**
 * SweepChart — kind-aware Plotly view.
 *   thermal      : SNR(T) + BER(T) overlaid, dual y-axes
 *   montecarlo   : BER histogram (log x) + SNR scatter vs trial
 *   ber_snr      : BER vs SNR(dB) with Wilson 95% error bars + analytical overlay
 *   ber_distance : BER vs distance(m) with Wilson 95% error bars + analytical curve
 *                  computed from each point's measured SNR
 */

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

import { analyticalBer } from "@/lib/dsp";
import { Card } from "@/primitives/Card";
import { PlotCanvas } from "@/primitives/PlotCanvas";
import {
  useSweepStore,
  type BerSweepPoint,
  type MonteCarloSample,
  type ThermalPoint,
} from "@/store/sweepStore";

function isThermal(p: unknown): p is ThermalPoint {
  return !!p && typeof (p as ThermalPoint).temperature_C === "number";
}

function isMc(p: unknown): p is MonteCarloSample {
  return (
    !!p &&
    typeof (p as MonteCarloSample).i === "number" &&
    typeof (p as MonteCarloSample).overrides === "object"
  );
}

function isBer(p: unknown): p is BerSweepPoint {
  return (
    !!p &&
    typeof (p as BerSweepPoint).x_value === "number" &&
    typeof (p as BerSweepPoint).ci_low === "number"
  );
}

const FLOOR_BER = 1e-8;

export function SweepChart() {
  const kind = useSweepStore((s) => s.kind);
  const points = useSweepStore((s) => s.points);

  const { data, layout, title } = useMemo<{
    data: Data[];
    layout: Partial<Layout>;
    title: string;
  }>(() => {
    if (kind === "thermal") {
      const pts = points.filter(isThermal);
      return buildThermal(pts);
    }
    if (kind === "montecarlo") {
      const pts = points.filter(isMc);
      return buildMonteCarlo(pts);
    }
    if (kind === "ber_snr" || kind === "ber_distance") {
      const pts = points.filter(isBer);
      return buildBerSweep(pts, kind);
    }
    return { data: [], layout: {}, title: "" };
  }, [kind, points]);

  if (!kind || points.length === 0) {
    return (
      <Card>
        <p className="text-sm text-slate-400">
          Run a sweep to see results stream in live.
        </p>
      </Card>
    );
  }

  return (
    <Card>
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">
        {title}
      </h3>
      <PlotCanvas data={data} layout={layout} aspect="16/9" />
    </Card>
  );
}

// ---------- builders ----------

function buildThermal(pts: ThermalPoint[]) {
  const T = pts.map((p) => p.temperature_C);
  const snr = pts.map((p) => p.snr_dB);
  const ber = pts.map((p) => Math.max(p.ber, FLOOR_BER));
  return {
    title: "SNR & BER vs Temperature",
    data: [
      {
        type: "scatter",
        mode: "lines+markers",
        x: T,
        y: snr,
        name: "SNR (dB)",
        line: { color: "#22d3ee", width: 2 },
        marker: { size: 8 },
        yaxis: "y",
      },
      {
        type: "scatter",
        mode: "lines+markers",
        x: T,
        y: ber,
        name: "BER (log)",
        line: { color: "#fbbf24", width: 2 },
        marker: { size: 8 },
        yaxis: "y2",
      },
    ] as Data[],
    layout: {
      xaxis: { title: { text: "temperature (°C)" } },
      yaxis: { title: { text: "SNR (dB)" }, side: "left" as const },
      yaxis2: {
        title: { text: "BER" },
        type: "log" as const,
        overlaying: "y" as const,
        side: "right" as const,
        showgrid: false,
      },
      legend: { orientation: "h" as const, y: 1.15 },
    },
  };
}

function buildMonteCarlo(pts: MonteCarloSample[]) {
  const ber = pts.map((p) => Math.max(p.ber, FLOOR_BER));
  const idx = pts.map((p) => p.i);
  const snr = pts.map((p) => p.snr_dB);
  return {
    title: "Monte Carlo BER + SNR",
    data: [
      {
        type: "histogram",
        x: ber,
        name: "BER",
        marker: { color: "#22d3ee" },
        opacity: 0.85,
        xaxis: "x",
        yaxis: "y",
        xbins: { size: 0.2 } as never,
      },
      {
        type: "scatter",
        mode: "markers",
        x: idx,
        y: snr,
        name: "SNR per trial",
        marker: { color: "#fbbf24", size: 6 },
        xaxis: "x2",
        yaxis: "y2",
      },
    ] as Data[],
    layout: {
      grid: { rows: 1, columns: 2, pattern: "independent" as const },
      xaxis: { title: { text: "BER (log)" }, type: "log" as const, domain: [0, 0.46] },
      yaxis: { title: { text: "count" } },
      xaxis2: { title: { text: "trial #" }, domain: [0.54, 1] },
      yaxis2: { title: { text: "SNR (dB)" }, anchor: "x2" as const },
      legend: { orientation: "h" as const, y: 1.15 },
    },
  };
}

function buildBerSweep(pts: BerSweepPoint[], kind: "ber_snr" | "ber_distance") {
  // Sort by x (or by SNR for ber_snr — we plot vs SNR_dB, not vs ambient lux).
  const useSnrAxis = kind === "ber_snr";
  const sorted = [...pts].sort((a, b) =>
    useSnrAxis ? a.snr_est_dB - b.snr_est_dB : a.x_value - b.x_value,
  );

  const xs = sorted.map((p) => (useSnrAxis ? p.snr_est_dB : p.x_value));
  const ber = sorted.map((p) => Math.max(p.ber, FLOOR_BER));
  const errLo = sorted.map((p, i) => Math.max(ber[i] - Math.max(p.ci_low, FLOOR_BER), 0));
  const errHi = sorted.map((p, i) => Math.max(Math.min(p.ci_high, 1) - ber[i], 0));

  const modulation = sorted[0]?.modulation ?? "OOK";
  // Analytical curve: pair each point's SNR with theory(SNR), so the
  // overlay tracks the same x-axis. For ber_distance we still parameterize
  // by SNR_dB at each measured point (theory is SNR-only).
  const theory = sorted.map((p) => {
    const v = analyticalBer(modulation, Math.pow(10, p.snr_est_dB / 10));
    return Number.isFinite(v) ? Math.max(v, FLOOR_BER) : null;
  });
  const theoryXs = sorted.map((p) => (useSnrAxis ? p.snr_est_dB : p.x_value));

  const xTitle = useSnrAxis ? "estimated SNR (dB)" : "distance (m)";
  const title = useSnrAxis ? "BER vs SNR" : "BER vs distance";

  return {
    title,
    data: [
      {
        type: "scatter",
        mode: "lines+markers",
        x: xs,
        y: ber,
        name: "measured (95% CI)",
        line: { color: "#22d3ee", width: 2 },
        marker: { size: 7 },
        error_y: {
          type: "data" as const,
          symmetric: false,
          array: errHi,
          arrayminus: errLo,
          color: "rgba(34,211,238,0.5)",
          thickness: 1.2,
          width: 4,
        },
      },
      {
        type: "scatter",
        mode: "lines",
        x: theoryXs,
        y: theory,
        name: `theory (${modulation})`,
        line: { color: "#a78bfa", width: 1.5, dash: "dash" },
        hoverinfo: "skip",
      },
    ] as Data[],
    layout: {
      xaxis: useSnrAxis
        ? { title: { text: xTitle } }
        : { title: { text: xTitle }, type: "log" as const },
      yaxis: { title: { text: "BER" }, type: "log" as const },
      legend: { orientation: "h" as const, y: 1.15 },
    },
  };
}

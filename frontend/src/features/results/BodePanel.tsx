/**
 * BodePanel — magnitude (dB) + phase (deg) subplots with crosshairs at
 * the unity-gain and 180° crossover frequencies plus a header strip
 * showing gain/phase margin and 3-dB bandwidth.
 */

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

import { Card } from "@/primitives/Card";
import { PlotCanvas } from "@/primitives/PlotCanvas";
import { useAcStore } from "@/store/acStore";

export function BodePanel() {
  const result = useAcStore((s) => s.result);
  const running = useAcStore((s) => s.running);
  const error = useAcStore((s) => s.error);

  const { data, layout } = useMemo<{ data: Data[]; layout: Partial<Layout> }>(() => {
    if (!result) return { data: [], layout: {} };
    const shapes: NonNullable<Layout["shapes"]> = [];
    if (typeof result.unity_gain_freq_hz === "number") {
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: result.unity_gain_freq_hz,
        x1: result.unity_gain_freq_hz,
        y0: 0,
        y1: 1,
        line: { color: "rgba(34,211,238,0.4)", dash: "dot", width: 1 },
      });
    }
    if (typeof result.crossover_180_freq_hz === "number") {
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: result.crossover_180_freq_hz,
        x1: result.crossover_180_freq_hz,
        y0: 0,
        y1: 1,
        line: { color: "rgba(251,191,36,0.4)", dash: "dot", width: 1 },
      });
    }
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: result.frequency_hz,
          y: result.magnitude_db,
          name: "|H| (dB)",
          line: { color: "#22d3ee", width: 1.6 },
          yaxis: "y",
          xaxis: "x",
        },
        {
          type: "scatter",
          mode: "lines",
          x: result.frequency_hz,
          y: result.phase_deg,
          name: "phase (deg)",
          line: { color: "#fbbf24", width: 1.4 },
          yaxis: "y2",
          xaxis: "x",
        },
      ] as Data[],
      layout: {
        grid: { rows: 2, columns: 1, pattern: "independent" as const },
        xaxis: { title: { text: "frequency (Hz)" }, type: "log" as const },
        yaxis: { title: { text: "magnitude (dB)" }, domain: [0.54, 1] },
        yaxis2: { title: { text: "phase (deg)" }, domain: [0, 0.44] },
        showlegend: false,
        shapes,
      },
    };
  }, [result]);

  if (error) {
    return (
      <Card>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
          Bode plot
        </h3>
        <p className="text-sm text-rose-400">{error}</p>
      </Card>
    );
  }
  if (!result) {
    return (
      <Card>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
          Bode plot
        </h3>
        <p className="text-sm text-slate-400">
          {running ? "Running AC analysis…" : "Run an AC sweep to see the response."}
        </p>
      </Card>
    );
  }

  return (
    <Card>
      <header className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Bode plot
        </h3>
        <div className="flex flex-wrap gap-2 text-[11px]">
          {chip("GM", fmtDb(result.gain_margin_db))}
          {chip("PM", fmtDeg(result.phase_margin_deg))}
          {chip("f_unity", fmtHz(result.unity_gain_freq_hz))}
          {chip("f_180", fmtHz(result.crossover_180_freq_hz))}
          {chip("BW (-3dB)", fmtBw(result.f_low_3dB_hz, result.f_high_3dB_hz))}
          {chip("midband", `${result.midband_gain_db.toFixed(1)} dB`)}
        </div>
      </header>
      <PlotCanvas data={data} layout={layout} aspect="16/11" />
    </Card>
  );
}

function chip(label: string, value: string) {
  return (
    <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-0.5">
      <span className="text-slate-500">{label}</span>{" "}
      <span className="font-mono text-slate-200">{value}</span>
    </span>
  );
}

function fmtDb(v: number | null): string {
  return v === null || !Number.isFinite(v) ? "—" : `${v.toFixed(1)} dB`;
}

function fmtDeg(v: number | null): string {
  return v === null || !Number.isFinite(v) ? "—" : `${v.toFixed(0)}°`;
}

function fmtHz(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "—";
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)} MHz`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(2)} kHz`;
  return `${v.toFixed(1)} Hz`;
}

function fmtBw(lo: number | null, hi: number | null): string {
  if (lo === null || hi === null) return "—";
  return `${fmtHz(lo)} – ${fmtHz(hi)}`;
}

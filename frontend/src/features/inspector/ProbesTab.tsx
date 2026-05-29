/**
 * ProbesTab — pick a signal from the last simulation run and inspect its
 * statistics (min / max / mean / std / RMS) alongside a thumbnail trace.
 *
 * Today's available signals come from resultsStore.waveforms — P_tx, P_rx,
 * I_ph, V_rx. Per-node RX signals (V_sense, V_ina, V_bpf, V_comp) require
 * SimulationResult plumbing through the WebSocket and will be added once
 * the SPICE-extract path lands.
 */

import { useMemo, useState } from "react";

import { PlotCanvas } from "@/primitives/PlotCanvas";
import { Card } from "@/primitives/Card";
import { useResultsStore, type Waveforms } from "@/store/resultsStore";

type ProbeKey = "P_tx" | "P_rx" | "I_ph" | "V_rx";

const PROBES: { key: ProbeKey; label: string; unit: string }[] = [
  { key: "P_tx", label: "TX optical power", unit: "W" },
  { key: "P_rx", label: "RX optical power", unit: "W" },
  { key: "I_ph", label: "Photocurrent", unit: "A" },
  { key: "V_rx", label: "RX voltage", unit: "V" },
];

interface SignalStats {
  n: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  rms: number;
  peakToPeak: number;
}

function computeStats(arr: number[]): SignalStats | null {
  if (!arr || arr.length === 0) return null;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSq = 0;
  for (const v of arr) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sumSq += v * v;
  }
  const n = arr.length;
  const mean = sum / n;
  const variance = sumSq / n - mean * mean;
  const std = Math.sqrt(Math.max(variance, 0));
  const rms = Math.sqrt(sumSq / n);
  return { n, min, max, mean, std, rms, peakToPeak: max - min };
}

function getSeries(w: Waveforms | null, key: ProbeKey): number[] | null {
  if (!w) return null;
  return w[key];
}

export function ProbesTab() {
  const waveforms = useResultsStore((s) => s.waveforms);
  const [active, setActive] = useState<ProbeKey>("P_rx");

  const series = useMemo(() => getSeries(waveforms, active), [waveforms, active]);
  const stats = useMemo(() => computeStats(series ?? []), [series]);
  const unit = PROBES.find((p) => p.key === active)?.unit ?? "";

  if (!waveforms) {
    return (
      <Card>
        <p className="text-sm text-slate-400">
          No waveforms loaded yet — run a simulation from the Builder or
          Engine route, then come back here to inspect the signals.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex flex-wrap gap-2">
          {PROBES.map((p) => (
            <button
              key={p.key}
              type="button"
              onClick={() => setActive(p.key)}
              className={`rounded-lg border px-3 py-1.5 text-xs transition ${
                active === p.key
                  ? "border-sky-500/60 bg-sky-500/15 text-sky-100"
                  : "border-slate-700 bg-slate-800/30 text-slate-300 hover:border-slate-500"
              }`}
            >
              <span className="font-mono">{p.key}</span>
              <span className="ml-2 text-slate-400">{p.label}</span>
            </button>
          ))}
        </div>
      </Card>

      <Card>
        {stats ? (
          <div className="grid grid-cols-2 gap-3 text-xs md:grid-cols-4">
            <StatCell label="samples" value={stats.n.toLocaleString()} />
            <StatCell label="min" value={`${fmt(stats.min)} ${unit}`} />
            <StatCell label="max" value={`${fmt(stats.max)} ${unit}`} />
            <StatCell label="peak-peak" value={`${fmt(stats.peakToPeak)} ${unit}`} />
            <StatCell label="mean" value={`${fmt(stats.mean)} ${unit}`} />
            <StatCell label="std" value={`${fmt(stats.std)} ${unit}`} />
            <StatCell label="RMS" value={`${fmt(stats.rms)} ${unit}`} />
            <StatCell
              label="dynamic range"
              value={
                stats.std > 0
                  ? `${(20 * Math.log10(Math.max(stats.peakToPeak, 1e-30) / stats.std)).toFixed(1)} dB`
                  : "—"
              }
            />
          </div>
        ) : (
          <p className="text-sm text-slate-400">No samples in {active}.</p>
        )}
      </Card>

      <Card>
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
          {active} trace
        </h4>
        <PlotCanvas
          data={[
            {
              x: waveforms.time,
              y: series ?? [],
              // SVG scatter (not scattergl): the packaged WebView2 has no
              // WebGL, and a few-thousand-point trace renders fine in SVG.
              type: "scatter",
              mode: "lines",
              line: { color: "#7dd3fc", width: 1 },
              hoverinfo: "x+y",
            },
          ]}
          layout={{
            xaxis: { title: { text: "time (s)" }, color: "#94a3b8" },
            yaxis: { title: { text: `${active} (${unit})` }, color: "#94a3b8" },
            showlegend: false,
          }}
          aspect="16/5"
        />
      </Card>
    </div>
  );
}

function StatCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className="font-mono text-sm text-slate-200">{value}</div>
    </div>
  );
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs < 1e-3 || abs >= 1e4) return v.toExponential(3);
  return v.toFixed(4);
}

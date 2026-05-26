/**
 * NoiseBreakdownPanel — visualizes the six-source physical noise model
 * (shot, thermal, ambient, amplifier, ADC, processing).
 *
 * Pure CSS stacked bar + dense table with inline mini-bars. No Plotly: a
 * single 99%+ source used to cause the Plotly legend to overflow on top of
 * the table. This layout cannot overlap by construction.
 */

import { useMemo } from "react";

import { useResultsStore, type NoiseBreakdown } from "@/store/resultsStore";

interface Row {
  key: keyof Omit<NoiseBreakdown, "total_A2" | "total_std_A">;
  label: string;
  color: string;
}

const ROWS: Row[] = [
  { key: "shot_A2", label: "Shot", color: "#22d3ee" },
  { key: "thermal_A2", label: "Thermal", color: "#fbbf24" },
  { key: "ambient_A2", label: "Ambient", color: "#fb7185" },
  { key: "amplifier_A2", label: "Amplifier", color: "#a78bfa" },
  { key: "adc_A2", label: "ADC", color: "#34d399" },
  { key: "processing_A2", label: "Processing", color: "#f59e0b" },
];

export function NoiseBreakdownPanel() {
  const nb = useResultsStore((s) => s.metrics?.noise_breakdown ?? null);

  const view = useMemo(() => {
    if (!nb) return null;
    const total = Math.max(nb.total_A2, 1e-30);
    const segments = ROWS.map((r) => ({
      ...r,
      value: nb[r.key],
      pct: (nb[r.key] / total) * 100,
    }));
    const dominant = [...segments].sort((a, b) => b.value - a.value)[0];
    return { segments, total, dominant };
  }, [nb]);

  if (!view) {
    return (
      <section className="border border-hair bg-slate-950">
        <header className="border-b border-hair px-2 py-1.5">
          <span className="label-inst">Noise breakdown</span>
        </header>
        <p className="px-2 py-3 text-[11px] text-slate-500">
          Run a simulation to see per-source noise contributions.
        </p>
      </section>
    );
  }

  const sigma = view.total ** 0.5;

  return (
    <section className="border border-hair bg-slate-950">
      <header className="flex items-center justify-between gap-3 border-b border-hair px-2 py-1.5">
        <span className="label-inst">Noise breakdown</span>
        <div className="readout flex items-center gap-3 text-[10px] text-slate-400">
          <span>
            <span className="text-slate-600">σ</span> {formatA(sigma)}A
          </span>
          <span>
            <span className="text-slate-600">σ²</span> {formatA2(view.total)} A²
          </span>
          <span>
            <span className="text-slate-600">dom</span>{" "}
            <span style={{ color: view.dominant.color }}>
              {view.dominant.label.toLowerCase()} {view.dominant.pct.toFixed(1)}%
            </span>
          </span>
        </div>
      </header>

      {/* Stacked bar — pure CSS, each segment is a flex item with width = %. */}
      <div className="border-b border-hair px-2 py-2">
        <div className="flex h-2 w-full overflow-hidden border border-hair bg-slate-900">
          {view.segments.map((s) => (
            <div
              key={s.key}
              style={{
                width: `${s.pct}%`,
                backgroundColor: s.color,
                minWidth: s.pct > 0 ? "1px" : 0,
              }}
              title={`${s.label} · ${s.pct.toFixed(2)}%`}
            />
          ))}
        </div>
      </div>

      {/* Dense table — one row per source, inline mini-bar shows % visually. */}
      <table className="w-full readout text-[11px]">
        <thead>
          <tr className="label-inst border-b border-hair">
            <th className="px-2 py-1 text-left">Source</th>
            <th className="px-2 py-1 text-right">σ² (A²)</th>
            <th className="px-2 py-1 text-right">% of total</th>
            <th className="px-2 py-1 text-left">share</th>
          </tr>
        </thead>
        <tbody>
          {view.segments.map((s) => (
            <tr
              key={s.key}
              className="border-b border-hair last:border-b-0 hover:bg-white/[0.02]"
            >
              <td className="px-2 py-1">
                <span className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-2 w-2 shrink-0"
                    style={{ backgroundColor: s.color }}
                  />
                  <span className="text-slate-200">{s.label}</span>
                </span>
              </td>
              <td className="px-2 py-1 text-right text-slate-200">
                {formatA2(s.value)}
              </td>
              <td
                className={
                  "px-2 py-1 text-right tab-nums " +
                  (s.pct >= 1 ? "text-slate-200" : "text-slate-500")
                }
              >
                {s.pct >= 0.05 ? s.pct.toFixed(1) : s.pct.toExponential(1)}%
              </td>
              <td className="px-2 py-1">
                <div className="h-1.5 w-full border border-hair bg-slate-900">
                  <div
                    style={{
                      width: `${Math.max(Math.min(s.pct, 100), s.pct > 0 ? 0.5 : 0)}%`,
                      backgroundColor: s.color,
                      height: "100%",
                    }}
                  />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function formatA(v: number): string {
  if (!Number.isFinite(v) || v === 0) return "0";
  const abs = Math.abs(v);
  if (abs < 1e-9) return `${(v * 1e12).toFixed(2)}p`;
  if (abs < 1e-6) return `${(v * 1e9).toFixed(2)}n`;
  if (abs < 1e-3) return `${(v * 1e6).toFixed(2)}µ`;
  if (abs < 1) return `${(v * 1e3).toFixed(2)}m`;
  return v.toExponential(2);
}

function formatA2(v: number): string {
  if (!Number.isFinite(v) || v === 0) return "0";
  return v.toExponential(2);
}

/**
 * WaveformsPanel — P_tx / P_rx / V_rx over time. Memoized so Plotly's
 * Plotly.react only fires when waveform array identity changes.
 */

import { useMemo } from "react";
import type { Data } from "plotly.js";

import { Card } from "@/primitives/Card";
import { PlotCanvas } from "@/primitives/PlotCanvas";
import { useResultsStore } from "@/store/resultsStore";

export function WaveformsPanel() {
  const waveforms = useResultsStore((s) => s.waveforms);

  const data: Data[] = useMemo(() => {
    if (!waveforms) return [];
    const t_us = waveforms.time.map((t) => t * 1e6); // seconds → microseconds
    return [
      {
        type: "scatter",
        mode: "lines",
        x: t_us,
        y: waveforms.P_tx.map((v) => v * 1e3),
        name: "P_tx (mW)",
        line: { color: "#22d3ee", width: 1.5 },
        yaxis: "y",
      },
      {
        type: "scatter",
        mode: "lines",
        x: t_us,
        y: waveforms.P_rx.map((v) => v * 1e6),
        name: "P_rx (µW)",
        line: { color: "#fbbf24", width: 1.5 },
        yaxis: "y2",
      },
      {
        type: "scatter",
        mode: "lines",
        x: t_us,
        y: waveforms.V_rx,
        name: "V_rx (V)",
        line: { color: "#a78bfa", width: 1.2, dash: "dot" },
        yaxis: "y3",
      },
    ];
  }, [waveforms]);

  const layout = useMemo(
    () => ({
      grid: { rows: 3, columns: 1, pattern: "independent" as const },
      xaxis: { title: { text: "time (µs)" } },
      yaxis: { title: { text: "P_tx (mW)" }, domain: [0.68, 1] },
      yaxis2: { title: { text: "P_rx (µW)" }, domain: [0.34, 0.66], anchor: "x" as const },
      yaxis3: { title: { text: "V_rx (V)" }, domain: [0, 0.32], anchor: "x" as const },
      showlegend: false,
    }),
    [],
  );

  if (!waveforms) {
    return (
      <Card>
        <p className="text-sm text-slate-400">
          Run a simulation to see waveforms.
        </p>
      </Card>
    );
  }

  return (
    <Card>
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">
        Waveforms ({waveforms.modulation ?? "—"})
      </h3>
      <PlotCanvas data={data} layout={layout} aspect="16/11" />
    </Card>
  );
}

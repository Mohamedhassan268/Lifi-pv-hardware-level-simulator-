/**
 * BitsComparePanel — TX vs RX bits, with errors highlighted. Simple stem-style
 * scatter; the eye diagram is a richer Phase-2 follow-up.
 */

import { useMemo } from "react";
import type { Data } from "plotly.js";

import { Card } from "@/primitives/Card";
import { PlotCanvas } from "@/primitives/PlotCanvas";
import { useResultsStore } from "@/store/resultsStore";

export function BitsComparePanel() {
  const waveforms = useResultsStore((s) => s.waveforms);
  const metrics = useResultsStore((s) => s.metrics);

  const data: Data[] = useMemo(() => {
    if (!waveforms?.bits_tx?.length) return [];
    const idx = waveforms.bits_tx.map((_, i) => i);
    const errIdx: number[] = [];
    waveforms.bits_tx.forEach((b, i) => {
      if (waveforms.bits_rx[i] !== undefined && b !== waveforms.bits_rx[i]) {
        errIdx.push(i);
      }
    });
    return [
      {
        type: "scatter",
        mode: "markers",
        x: idx,
        y: waveforms.bits_tx,
        name: "TX",
        marker: { color: "#22d3ee", size: 8, symbol: "square" },
      },
      {
        type: "scatter",
        mode: "markers",
        x: idx,
        y: waveforms.bits_rx,
        name: "RX",
        marker: { color: "#fbbf24", size: 8, symbol: "circle-open" },
      },
      ...(errIdx.length
        ? [
            {
              type: "scatter" as const,
              mode: "markers" as const,
              x: errIdx,
              y: errIdx.map(() => 1.15),
              name: "errors",
              marker: { color: "#f43f5e", size: 10, symbol: "x" as const },
            },
          ]
        : []),
    ];
  }, [waveforms]);

  const layout = useMemo(
    () => ({
      xaxis: { title: { text: "bit index" } },
      yaxis: { title: { text: "bit value" }, range: [-0.3, 1.3] as [number, number] },
    }),
    [],
  );

  if (!waveforms?.bits_tx?.length) {
    return (
      <Card>
        <p className="text-sm text-slate-400">No bit stream yet.</p>
      </Card>
    );
  }

  const ber = metrics?.ber;
  const nErr = metrics?.ber_n_errors ?? 0;
  const nBits = metrics?.ber_n_bits ?? waveforms.bits_tx.length;

  return (
    <Card>
      <div className="mb-3 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          TX vs RX bits
        </h3>
        <p className="font-mono text-xs text-slate-400">
          {nErr} / {nBits} errors
          {typeof ber === "number" && ` · BER ${ber.toExponential(2)}`}
        </p>
      </div>
      <PlotCanvas data={data} layout={layout} aspect="16/6" />
    </Card>
  );
}

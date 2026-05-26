/**
 * SpectrumPanel — Hann-windowed FFT of V_rx, one-sided dB magnitude.
 *
 * Pure frontend (no backend changes). The FFT itself lives in lib/dsp.ts.
 * We log-scale the x-axis when the signal has a useful bandwidth so the
 * carrier / sub-carrier structure is visible.
 */

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

import { inferSampleRate, powerSpectrumDb } from "@/lib/dsp";
import { Card } from "@/primitives/Card";
import { PlotCanvas } from "@/primitives/PlotCanvas";
import { useResultsStore } from "@/store/resultsStore";

export function SpectrumPanel() {
  const waveforms = useResultsStore((s) => s.waveforms);

  const { data, layout, hint } = useMemo(() => {
    if (!waveforms || !waveforms.time?.length) {
      return {
        data: [] as Data[],
        layout: {} as Partial<Layout>,
        hint: "Run a simulation to see the spectrum.",
      };
    }
    const fs = inferSampleRate(waveforms.time);
    if (!fs) {
      return {
        data: [] as Data[],
        layout: {} as Partial<Layout>,
        hint: "Could not infer sample rate from waveforms.",
      };
    }

    const { freq, dB } = powerSpectrumDb(waveforms.V_rx, fs);

    // Drop DC bin to avoid a misleading spike on log axes.
    const xs = Array.from(freq.slice(1));
    const ys = Array.from(dB.slice(1));

    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: xs,
          y: ys,
          line: { color: "#a78bfa", width: 1.2 },
          name: "|V_rx|² (dB)",
          showlegend: false,
        },
      ] as Data[],
      layout: {
        xaxis: { title: { text: "frequency (Hz)" }, type: "log" as const },
        yaxis: { title: { text: "power (dB)" } },
      },
      hint: `fs = ${(fs / 1000).toFixed(1)} kHz · N = ${waveforms.V_rx.length}`,
    };
  }, [waveforms]);

  if (data.length === 0) {
    return (
      <Card>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
          V_rx spectrum
        </h3>
        <p className="text-sm text-slate-400">{hint}</p>
      </Card>
    );
  }

  return (
    <Card>
      <header className="mb-3 flex items-baseline justify-between gap-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          V_rx spectrum
        </h3>
        <span className="text-[11px] text-slate-500">{hint}</span>
      </header>
      <PlotCanvas data={data} layout={layout} aspect="16/9" />
    </Card>
  );
}

/**
 * EyeDiagramPanel — overlay V_rx folded modulo 2·T_bit.
 *
 * Each bit period yields one trace; all traces share the same x-axis (time
 * within the 2-UI window, in µs). A clean signal looks like a single open
 * "eye" — noise and ISI close it.
 *
 * Pure frontend: no backend changes. We derive T_bit from the configured
 * data_rate_bps. We chunk the waveform into segments of length floor(2 * fs * T_bit)
 * and emit each as its own Plotly scatter trace at low alpha.
 */

import { useMemo } from "react";
import type { Data } from "plotly.js";

import { inferSampleRate } from "@/lib/dsp";
import { computeEyeMetrics } from "@/lib/eyeMetrics";
import { Card } from "@/primitives/Card";
import { PlotCanvas } from "@/primitives/PlotCanvas";
import { useConfigStore } from "@/store/configStore";
import { useResultsStore } from "@/store/resultsStore";

const MAX_TRACES = 80; // cap for plotly performance

export function EyeDiagramPanel() {
  const waveforms = useResultsStore((s) => s.waveforms);
  const dataRateBps = useConfigStore(
    (s) => (s.config.data_rate_bps as number | undefined) ?? null,
  );

  const { data, layout, hint } = useMemo(() => {
    if (!waveforms || !waveforms.time?.length || !dataRateBps) {
      return { data: [] as Data[], layout: {}, hint: "Run a simulation to see the eye diagram." };
    }
    const fs = inferSampleRate(waveforms.time);
    if (!fs) {
      return { data: [] as Data[], layout: {}, hint: "Could not infer sample rate from waveforms." };
    }

    const tBit = 1 / dataRateBps;
    const segLen = Math.max(4, Math.floor(2 * fs * tBit)); // 2 UIs wide
    const vrx = waveforms.V_rx;
    const N = vrx.length;
    const total = Math.floor(N / segLen);
    if (total < 2) {
      return {
        data: [] as Data[],
        layout: {},
        hint: "Too few samples per bit period for an eye — increase t_stop_s or bit count.",
      };
    }

    const stride = Math.max(1, Math.ceil(total / MAX_TRACES));
    const xAxis: number[] = new Array(segLen);
    for (let i = 0; i < segLen; i++) xAxis[i] = (i / fs) * 1e6; // µs

    const traces: Data[] = [];
    for (let s = 0; s + segLen <= N; s += segLen * stride) {
      const y = vrx.slice(s, s + segLen);
      traces.push({
        type: "scatter",
        mode: "lines",
        x: xAxis,
        y,
        showlegend: false,
        line: { color: "rgba(103,232,249,0.18)", width: 1 },
        hoverinfo: "skip",
      });
    }

    return {
      data: traces,
      layout: {
        xaxis: { title: { text: "time within 2·T_bit (µs)" } },
        yaxis: { title: { text: "V_rx (V)" } },
      },
      hint: `${traces.length} segments overlaid · T_bit = ${(tBit * 1e6).toFixed(1)} µs`,
    };
  }, [waveforms, dataRateBps]);

  if (data.length === 0) {
    return (
      <Card>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
          Eye diagram
        </h3>
        <p className="text-sm text-slate-400">{hint}</p>
      </Card>
    );
  }

  return (
    <Card>
      <header className="mb-3 flex items-baseline justify-between gap-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Eye diagram
        </h3>
        <span className="text-[11px] text-slate-500">{hint}</span>
      </header>
      <EyeMetricsStrip />
      <PlotCanvas data={data} layout={layout} aspect="16/9" />
    </Card>
  );
}

function EyeMetricsStrip() {
  const waveforms = useResultsStore((s) => s.waveforms);
  const dataRateBps = useConfigStore(
    (s) => (s.config.data_rate_bps as number | undefined) ?? null,
  );
  const metrics = useMemo(() => {
    if (!waveforms || !dataRateBps) return null;
    return computeEyeMetrics(
      waveforms.time,
      waveforms.V_rx,
      waveforms.bits_tx,
      dataRateBps,
    );
  }, [waveforms, dataRateBps]);

  if (!metrics) return null;

  const chip = (label: string, value: string) => (
    <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px]">
      <span className="text-slate-500">{label}</span>{" "}
      <span className="font-mono text-slate-200">{value}</span>
    </span>
  );

  return (
    <div className="mb-3 flex flex-wrap gap-1.5">
      {chip("opening", `${(metrics.eyeOpeningV * 1000).toFixed(1)} mV`)}
      {chip("eye SNR", `${metrics.eyeSnrDb.toFixed(1)} dB`)}
      {chip("jitter σ", `${(metrics.jitterSigmaSec * 1e9).toFixed(1)} ns`)}
      {chip("edges", `${metrics.edgeCount}`)}
    </div>
  );
}

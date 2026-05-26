/**
 * ConstellationPanel — what we can plot today depends on the modulation:
 *
 *   OOK / OOK_Manchester / PWM_ASK — amplitude-modulated binary schemes.
 *   We sample V_rx at the centre of each bit period and scatter those
 *   amplitudes on a 1-D axis, colour-coded by the transmitted bit. With a
 *   clean link the two clouds separate; with noise/ISI they overlap. The
 *   midline is the implicit decision threshold.
 *
 *   OFDM / BFSK — true complex / I-Q constellations need the demodulator's
 *   intermediate symbols, which the backend does not currently expose. We
 *   render a placeholder pointing at the follow-up backend work.
 *
 * Pure frontend (no backend changes) for the amplitude case.
 */

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

import { inferSampleRate } from "@/lib/dsp";
import { Card } from "@/primitives/Card";
import { PlotCanvas } from "@/primitives/PlotCanvas";
import { useConfigStore } from "@/store/configStore";
import { useResultsStore } from "@/store/resultsStore";

const AMPLITUDE_SCHEMES = new Set(["OOK", "OOK_Manchester", "PWM_ASK"]);

export function ConstellationPanel() {
  const waveforms = useResultsStore((s) => s.waveforms);
  const dataRateBps = useConfigStore(
    (s) => (s.config.data_rate_bps as number | undefined) ?? null,
  );

  const modulation = (waveforms?.modulation as string | undefined) ?? "";

  const view = useMemo(() => {
    if (!waveforms || !waveforms.time?.length) {
      return { kind: "empty" as const, hint: "Run a simulation to see the decision scatter." };
    }
    if (!AMPLITUDE_SCHEMES.has(modulation)) {
      return {
        kind: "unsupported" as const,
        hint: `Constellation for ${modulation || "this scheme"} requires backend exposure of complex symbols. Tracked as a follow-up.`,
      };
    }
    if (!dataRateBps) {
      return { kind: "empty" as const, hint: "Configure data rate to build the decision scatter." };
    }

    const fs = inferSampleRate(waveforms.time);
    if (!fs) return { kind: "empty" as const, hint: "Could not infer sample rate." };

    const tBit = 1 / dataRateBps;
    const samplesPerBit = Math.max(2, Math.floor(fs * tBit));
    const bits = waveforms.bits_tx;
    const vrx = waveforms.V_rx;
    if (!bits?.length) return { kind: "empty" as const, hint: "No bits_tx in waveforms." };

    // Sample V_rx at the center of each bit period.
    const xZeros: number[] = [];
    const yZeros: number[] = [];
    const xOnes: number[] = [];
    const yOnes: number[] = [];
    const N = Math.min(bits.length, Math.floor(vrx.length / samplesPerBit));
    for (let i = 0; i < N; i++) {
      const idx = Math.floor((i + 0.5) * samplesPerBit);
      if (idx >= vrx.length) break;
      const v = vrx[idx];
      // tiny horizontal jitter so co-located points are visible
      const jitter = (Math.random() - 0.5) * 0.4;
      if (bits[i] === 0) {
        xZeros.push(jitter);
        yZeros.push(v);
      } else {
        xOnes.push(1 + jitter);
        yOnes.push(v);
      }
    }

    const allY = [...yZeros, ...yOnes];
    const threshold =
      allY.length > 0 ? (Math.min(...allY) + Math.max(...allY)) / 2 : 0;

    const data: Data[] = [
      {
        type: "scatter",
        mode: "markers",
        x: xZeros,
        y: yZeros,
        marker: { color: "#22d3ee", size: 6, opacity: 0.6 },
        name: "bit = 0",
      },
      {
        type: "scatter",
        mode: "markers",
        x: xOnes,
        y: yOnes,
        marker: { color: "#fbbf24", size: 6, opacity: 0.6 },
        name: "bit = 1",
      },
      {
        type: "scatter",
        mode: "lines",
        x: [-1, 2],
        y: [threshold, threshold],
        line: { color: "rgba(255,255,255,0.25)", dash: "dot", width: 1 },
        name: "decision threshold",
        hoverinfo: "skip",
      },
    ];
    const layout: Partial<Layout> = {
      xaxis: {
        title: { text: "symbol class (jittered)" },
        tickvals: [0, 1],
        ticktext: ["0", "1"],
        range: [-1, 2],
      },
      yaxis: { title: { text: "V_rx at bit centre (V)" } },
      legend: { orientation: "h", y: 1.12 },
    };
    return {
      kind: "scatter" as const,
      data,
      layout,
      hint: `${N} symbols · ${modulation}`,
    };
  }, [waveforms, modulation, dataRateBps]);

  if (view.kind === "scatter") {
    return (
      <Card>
        <header className="mb-3 flex items-baseline justify-between gap-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Decision scatter
          </h3>
          <span className="text-[11px] text-slate-500">{view.hint}</span>
        </header>
        <PlotCanvas data={view.data} layout={view.layout} aspect="16/9" />
      </Card>
    );
  }

  if (view.kind === "unsupported") {
    return (
      <Card>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
          Constellation
        </h3>
        <p className="text-sm text-slate-400">{view.hint}</p>
      </Card>
    );
  }

  return (
    <Card>
      <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
        Decision scatter
      </h3>
      <p className="text-sm text-slate-400">{view.hint}</p>
    </Card>
  );
}

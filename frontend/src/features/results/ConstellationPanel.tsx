/**
 * ConstellationPanel — three rendering modes selected by modulation:
 *
 *   OOK / OOK_Manchester / PWM_ASK
 *     Amplitude-modulated binary schemes. We sample V_rx at the centre of
 *     each bit period and scatter those amplitudes on a 1-D axis,
 *     colour-coded by the transmitted bit. With a clean link the two
 *     clouds separate; with noise/ISI they overlap. Pure frontend.
 *
 *   OFDM
 *     True complex I/Q constellation. Backend exposes the post-FFT data
 *     carrier symbols (ofdm_iq_real / ofdm_iq_imag) from the demodulator.
 *     With M-QAM the cloud forms an M-point grid; noise blurs each point
 *     into a Gaussian blob, channel response rotates / scales them.
 *
 *   BFSK
 *     Non-coherent BFSK has no I/Q, but the Goertzel decision plane is
 *     the equivalent: per-bit (E0, E1) energy pair. Bit=0 lands above
 *     the diagonal E1=E0, bit=1 below. Backend exposes bfsk_e0 / bfsk_e1.
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
      return { kind: "empty" as const, hint: "Run a simulation to see the constellation." };
    }

    // ---- OFDM I/Q ------------------------------------------------------
    if (modulation === "OFDM") {
      const re = waveforms.ofdm_iq_real;
      const im = waveforms.ofdm_iq_imag;
      if (!re?.length || !im?.length) {
        return {
          kind: "empty" as const,
          hint: "OFDM symbols not received yet — run a Python-engine OFDM simulation.",
        };
      }
      const n = Math.min(re.length, im.length);
      // Symmetric axis range: pad slightly so outer points aren't clipped.
      let mag = 0;
      for (let i = 0; i < n; i++) {
        const a = Math.abs(re[i]);
        const b = Math.abs(im[i]);
        if (a > mag) mag = a;
        if (b > mag) mag = b;
      }
      mag = mag > 0 ? mag * 1.15 : 1;
      const data: Data[] = [
        {
          type: "scatter",
          mode: "markers",
          x: re.slice(0, n),
          y: im.slice(0, n),
          marker: { color: "#fbbf24", size: 4, opacity: 0.55 },
          name: "received I/Q",
          hovertemplate: "I=%{x:.3f}<br>Q=%{y:.3f}<extra></extra>",
        },
        {
          type: "scatter",
          mode: "lines",
          x: [-mag, mag],
          y: [0, 0],
          line: { color: "rgba(255,255,255,0.15)", width: 1 },
          showlegend: false,
          hoverinfo: "skip",
        },
        {
          type: "scatter",
          mode: "lines",
          x: [0, 0],
          y: [-mag, mag],
          line: { color: "rgba(255,255,255,0.15)", width: 1 },
          showlegend: false,
          hoverinfo: "skip",
        },
      ];
      const layout: Partial<Layout> = {
        xaxis: { title: { text: "I" }, range: [-mag, mag], zeroline: false },
        yaxis: {
          title: { text: "Q" },
          range: [-mag, mag],
          zeroline: false,
          scaleanchor: "x",
          scaleratio: 1,
        },
        showlegend: false,
      };
      return {
        kind: "iq" as const,
        data,
        layout,
        hint: `${n} symbols · OFDM`,
      };
    }

    // ---- BFSK Goertzel energy plane -----------------------------------
    if (modulation === "BFSK") {
      const e0 = waveforms.bfsk_e0;
      const e1 = waveforms.bfsk_e1;
      const bits = waveforms.bits_tx;
      if (!e0?.length || !e1?.length) {
        return {
          kind: "empty" as const,
          hint: "BFSK energies not received yet — run a BFSK simulation.",
        };
      }
      const n = Math.min(e0.length, e1.length);
      // Split into bit=0 / bit=1 clouds if we have bits_tx.
      const x0: number[] = [];
      const y0: number[] = [];
      const x1: number[] = [];
      const y1: number[] = [];
      // bits_tx may have a different length (subsampled to 512); align to
      // the shorter axis by truncating.
      const m = bits?.length ? Math.min(n, bits.length) : n;
      for (let i = 0; i < m; i++) {
        if (bits && bits[i] === 1) {
          x1.push(e0[i]);
          y1.push(e1[i]);
        } else {
          x0.push(e0[i]);
          y0.push(e1[i]);
        }
      }
      // any extra energies past bits_tx — colour them grey
      const xExtra = m < n ? e0.slice(m, n) : [];
      const yExtra = m < n ? e1.slice(m, n) : [];

      // Diagonal threshold E1 = E0
      const all = [...e0.slice(0, n), ...e1.slice(0, n)];
      const lo = Math.min(...all);
      const hi = Math.max(...all);
      const pad = (hi - lo) * 0.05 || hi * 0.05 || 1e-9;

      const data: Data[] = [
        {
          type: "scatter",
          mode: "markers",
          x: x0,
          y: y0,
          marker: { color: "#22d3ee", size: 7, opacity: 0.7 },
          name: "bit = 0 (E1 > E0)",
          hovertemplate: "E0=%{x:.2e}<br>E1=%{y:.2e}<extra></extra>",
        },
        {
          type: "scatter",
          mode: "markers",
          x: x1,
          y: y1,
          marker: { color: "#fbbf24", size: 7, opacity: 0.7 },
          name: "bit = 1 (E0 > E1)",
          hovertemplate: "E0=%{x:.2e}<br>E1=%{y:.2e}<extra></extra>",
        },
        ...(xExtra.length
          ? [
              {
                type: "scatter" as const,
                mode: "markers" as const,
                x: xExtra,
                y: yExtra,
                marker: { color: "#94a3b8", size: 5, opacity: 0.5 },
                name: "unlabelled",
                hovertemplate: "E0=%{x:.2e}<br>E1=%{y:.2e}<extra></extra>",
              },
            ]
          : []),
        {
          type: "scatter",
          mode: "lines",
          x: [lo - pad, hi + pad],
          y: [lo - pad, hi + pad],
          line: { color: "rgba(255,255,255,0.25)", dash: "dot", width: 1 },
          name: "E1 = E0 (decision)",
          hoverinfo: "skip",
        },
      ];
      const layout: Partial<Layout> = {
        xaxis: { title: { text: `E(f0) — Goertzel energy at f0` } },
        yaxis: {
          title: { text: `E(f1) — Goertzel energy at f1` },
          scaleanchor: "x",
          scaleratio: 1,
        },
        legend: { orientation: "h", y: 1.12 },
      };
      return {
        kind: "energies" as const,
        data,
        layout,
        hint: `${n} symbols · BFSK decision plane`,
      };
    }

    // ---- OOK / Manchester / PWM_ASK 1-D decision scatter -------------
    if (!AMPLITUDE_SCHEMES.has(modulation)) {
      return {
        kind: "unsupported" as const,
        hint: `Constellation for ${modulation || "this scheme"} not modelled yet.`,
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

    const xZeros: number[] = [];
    const yZeros: number[] = [];
    const xOnes: number[] = [];
    const yOnes: number[] = [];
    const N = Math.min(bits.length, Math.floor(vrx.length / samplesPerBit));
    for (let i = 0; i < N; i++) {
      const idx = Math.floor((i + 0.5) * samplesPerBit);
      if (idx >= vrx.length) break;
      const v = vrx[idx];
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

  if (view.kind === "iq" || view.kind === "energies" || view.kind === "scatter") {
    const title =
      view.kind === "iq"
        ? "I/Q constellation"
        : view.kind === "energies"
          ? "BFSK decision plane"
          : "Decision scatter";
    return (
      <Card>
        <header className="mb-3 flex items-baseline justify-between gap-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            {title}
          </h3>
          <span className="text-[11px] text-slate-500">{view.hint}</span>
        </header>
        <PlotCanvas data={view.data} layout={view.layout} aspect="1/1" />
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
        Constellation
      </h3>
      <p className="text-sm text-slate-400">{view.hint}</p>
    </Card>
  );
}

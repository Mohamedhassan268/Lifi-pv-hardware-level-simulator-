/**
 * MetricsCard — headline BER / SNR / P_rx numbers from the WebSocket
 * `metrics` event. Each value fade-swaps via an AnimatePresence key.
 */

import { motion } from "framer-motion";

import { analyticalBer, wilsonCi } from "@/lib/dsp";
import { DUR, EASE } from "@/lib/motion";
import { Card } from "@/primitives/Card";
import { useConfigStore } from "@/store/configStore";
import { useResultsStore, type Metrics } from "@/store/resultsStore";

type Row = {
  key: keyof Metrics | "throughput_kbps";
  label: string;
  unit: string;
  fmt: (v: number) => string;
};

const ROWS: Row[] = [
  { key: "ber", label: "BER", unit: "", fmt: (v) => v.toExponential(2) },
  { key: "snr_est_dB", label: "SNR", unit: "dB", fmt: (v) => v.toFixed(1) },
  { key: "throughput_kbps", label: "Throughput", unit: "kbps", fmt: (v) => v.toFixed(2) },
  { key: "P_rx_avg_uW", label: "P_rx (avg)", unit: "µW", fmt: (v) => v.toFixed(1) },
  { key: "I_ph_avg_uA", label: "I_ph (avg)", unit: "µA", fmt: (v) => v.toFixed(1) },
];

export function MetricsCard() {
  const metrics = useResultsStore((s) => s.metrics);
  const dataRateBps = useConfigStore(
    (s) => (s.config.data_rate_bps as number | undefined) ?? null,
  );
  const modulation = useConfigStore(
    (s) => (s.config.modulation as string | undefined) ?? "OOK",
  );

  // Throughput = data_rate × (1 − BER), converted to kbps. Only meaningful
  // once both BER and the configured rate are known.
  const throughputKbps =
    typeof metrics?.ber === "number" && typeof dataRateBps === "number"
      ? (dataRateBps * (1 - metrics.ber)) / 1000
      : undefined;

  // Phase 1C: analytical BER companion (theory) and Phase 2C: Wilson CI.
  const snrDb = metrics?.snr_est_dB;
  const theoreticalBer =
    typeof snrDb === "number"
      ? analyticalBer(modulation, Math.pow(10, snrDb / 10))
      : NaN;
  const measuredBer = metrics?.ber;
  const ciBounds =
    typeof metrics?.ber_n_errors === "number" &&
    typeof metrics?.ber_n_bits === "number" &&
    metrics.ber_n_bits > 0
      ? wilsonCi(metrics.ber_n_errors, metrics.ber_n_bits)
      : null;

  return (
    <Card>
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-300">
        Metrics
      </h3>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3">
        {ROWS.map((row) => {
          const val =
            row.key === "throughput_kbps"
              ? throughputKbps
              : metrics?.[row.key as keyof Metrics];
          const display =
            typeof val === "number" && Number.isFinite(val)
              ? row.fmt(val)
              : "—";
          return (
            <div key={row.key} className="flex flex-col">
              <dt className="text-xs uppercase tracking-wider text-slate-400">
                {row.label}
              </dt>
              <motion.dd
                key={`${row.key}-${display}`}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: DUR.fast, ease: EASE }}
                className="font-mono text-lg text-beam-300"
              >
                {display}
                {row.unit ? <span className="ml-1 text-sm text-slate-400">{row.unit}</span> : null}
              </motion.dd>
            </div>
          );
        })}
      </dl>
      {/* Measured vs theoretical BER + Wilson CI */}
      {(typeof measuredBer === "number" || Number.isFinite(theoreticalBer)) && (
        <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.02] p-3">
          <p className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">
            BER vs theory
          </p>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <dt className="text-slate-400">Measured</dt>
            <dd className="text-right font-mono text-beam-300">
              {typeof measuredBer === "number"
                ? measuredBer === 0
                  ? "0"
                  : measuredBer.toExponential(2)
                : "—"}
              {ciBounds && (
                <span className="ml-1 text-[10px] text-slate-500">
                  [{ciBounds[0].toExponential(1)}, {ciBounds[1].toExponential(1)}]
                </span>
              )}
            </dd>
            <dt className="text-slate-400">Theory ({modulation})</dt>
            <dd className="text-right font-mono text-slate-300">
              {Number.isFinite(theoreticalBer) ? theoreticalBer.toExponential(2) : "—"}
            </dd>
            {Number.isFinite(theoreticalBer) &&
              typeof measuredBer === "number" &&
              measuredBer > 0 &&
              theoreticalBer > 0 && (
                <>
                  <dt className="text-slate-400">Δ</dt>
                  <dd className="text-right font-mono text-slate-300">
                    {(10 * Math.log10(measuredBer / theoreticalBer)).toFixed(2)} dB
                  </dd>
                </>
              )}
          </dl>
        </div>
      )}

      {/* Energy harvest: raw PV cell power vs load-regulated DC-DC output.
          They diverge — the converter output is lower and stays flat under
          regulation while raw harvest rises. */}
      {(() => {
        const feat = metrics?.features;
        const raw = numFeat(feat?.harvested_power_W);
        if (raw == null) return null;
        const out = numFeat(feat?.dcdc_output_power_W);
        const eff = numFeat(feat?.dcdc_efficiency);
        return (
          <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.02] p-3">
            <p className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">
              Energy harvest
            </p>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              <dt className="text-slate-400">Raw PV cell</dt>
              <dd className="text-right font-mono text-harvest-300">
                {(raw * 1e6).toFixed(1)} µW
              </dd>
              <dt className="text-slate-400">DC-DC output</dt>
              <dd className="text-right font-mono text-harvest-300">
                {out != null ? `${(out * 1e6).toFixed(1)} µW` : "—"}
              </dd>
              {eff != null && (
                <>
                  <dt className="text-slate-400">Efficiency</dt>
                  <dd className="text-right font-mono text-slate-300">
                    {(eff * 100).toFixed(1)} %
                  </dd>
                </>
              )}
            </dl>
          </div>
        );
      })()}

      {metrics?.engine && (
        <p className="mt-4 text-xs text-slate-500">engine: {metrics.engine}</p>
      )}
    </Card>
  );
}

function numFeat(v: number | string | null | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

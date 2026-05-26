/**
 * ResultsMiniPanel — inline result card shown over the schematic canvas
 * after a simulation completes. Shows the headline metrics (BER, SNR,
 * P_rx) and an "Open full results" button that routes to the existing
 * EngineRoute where waveforms / bit-compare panels live.
 *
 * Visibility: shows while pipelineStore.running, and persists after the
 * run completes while resultsStore.metrics is populated. Closes itself
 * if the user starts a new run or opens a category modal.
 */

import { AnimatePresence, motion } from "framer-motion";

import { DUR, EASE } from "@/lib/motion";
import { Button } from "@/primitives/Button";
import { useBuilderUIStore } from "@/store/builderUIStore";
import { usePipelineStore } from "@/store/pipelineStore";
import { useResultsStore } from "@/store/resultsStore";
import { useUIStore } from "@/store/uiStore";

export function ResultsMiniPanel() {
  const running = usePipelineStore((s) => s.running);
  const steps = usePipelineStore((s) => s.steps);
  const metrics = useResultsStore((s) => s.metrics);
  const waveforms = useResultsStore((s) => s.waveforms);
  const openCategory = useBuilderUIStore((s) => s.openCategory);
  const setRoute = useUIStore((s) => s.setRoute);

  const visible = (running || metrics != null) && openCategory == null;
  // "Open full results" stays disabled until BOTH metrics and waveforms have
  // landed — otherwise the user can route to EngineRoute mid-stream and see
  // empty panels.
  const canOpenFull = !running && metrics != null && waveforms != null;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ y: 24, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 24, opacity: 0 }}
          transition={{ duration: DUR.base, ease: EASE }}
          className="absolute bottom-6 right-6 z-10 w-[340px] rounded-2xl border border-white/10
                     bg-slate-950/85 p-4 shadow-2xl backdrop-blur"
        >
          <header className="flex items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-slate-100">
              {running ? "Simulating…" : "Results"}
            </h3>
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              {metrics?.engine ?? "python"}
            </span>
          </header>

          <div className="mt-3 grid grid-cols-3 gap-3">
            <Metric label="BER" value={fmtBer(metrics?.ber)} accent="beam" />
            <Metric label="SNR" value={fmt(metrics?.snr_est_dB, "dB")} />
            <Metric label="P_rx" value={fmt(metrics?.P_rx_avg_uW, "µW")} accent="harvest" />
          </div>

          <ul className="mt-3 space-y-1">
            {(["TX", "Channel", "RX"] as const).map((n) => (
              <li key={n} className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400">{n}</span>
                <span
                  className={`uppercase tracking-wider ${
                    steps[n].status === "done"
                      ? "text-emerald-400"
                      : steps[n].status === "running"
                        ? "text-beam-300"
                        : steps[n].status === "error"
                          ? "text-rose-400"
                          : "text-slate-600"
                  }`}
                >
                  {steps[n].status}
                </span>
              </li>
            ))}
          </ul>

          <div className="mt-4">
            <Button
              variant="ghost"
              className="w-full"
              onClick={() => setRoute("engine")}
              disabled={!canOpenFull}
              title={canOpenFull ? undefined : "Waiting for metrics and waveforms…"}
            >
              {canOpenFull ? "Open full results →" : "Waiting for data…"}
            </Button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function Metric({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "beam" | "harvest";
}) {
  const color =
    accent === "beam"
      ? "text-beam-300"
      : accent === "harvest"
        ? "text-harvest-300"
        : "text-slate-100";
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2">
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className={`mt-0.5 text-sm font-semibold ${color}`}>{value}</p>
    </div>
  );
}

function fmt(v: number | null | undefined, unit: string): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return `${v.toFixed(2)} ${unit}`;
}

function fmtBer(v: number | null | undefined): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  return v.toExponential(1);
}

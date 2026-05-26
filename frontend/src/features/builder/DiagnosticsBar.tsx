/**
 * DiagnosticsBar — bottom instrument strip. Reads from configStore +
 * pipelineStore + resultsStore and renders a hairline-segmented status line
 * with the live numerics an engineer expects to see at a glance: pipeline
 * state, modulation, data rate, distance, P_rx, SNR, BER, clock.
 *
 * Format is intentionally compact: label-inst (caps) for the column name and
 * a monospace tabular-nums readout below. No data hidden behind hovers.
 */

import { useEffect, useState } from "react";

import { usePipelineStore } from "@/store/pipelineStore";
import { useResultsStore } from "@/store/resultsStore";
import { useConfigStore } from "@/store/configStore";

export function DiagnosticsBar() {
  const running = usePipelineStore((s) => s.running);
  const pipelineError = usePipelineStore((s) => s.error);
  const steps = usePipelineStore((s) => s.steps);
  const metrics = useResultsStore((s) => s.metrics);
  const config = useConfigStore((s) => s.config);
  const presetName = useConfigStore((s) => s.presetName);

  const distance = num(config.distance_m);
  const rate = num(config.data_rate_bps);
  const modulation = (config.modulation as string) ?? "—";
  const engine = metrics?.engine ?? (config.simulation_engine as string) ?? "python";
  const pRx = metrics?.P_rx_avg_uW;
  const snr = metrics?.snr_est_dB;
  const ber = metrics?.ber;
  const status = pipelineError ? "fail" : running ? "run" : metrics ? "done" : "idle";

  const completedSteps = (Object.values(steps) as { status: string }[]).filter(
    (s) => s.status === "ok",
  ).length;

  const now = useNowEverySecond();

  return (
    <footer className="flex items-stretch border-t border-hair bg-slate-950/95 readout text-[11px] text-slate-300">
      <Cell label="status" tone={statusTone(status)}>
        {labelForStatus(status)}
        {running && (
          <span className="text-slate-500"> · {completedSteps}/3</span>
        )}
      </Cell>
      <Cell label="preset">
        <span className={presetName ? "text-slate-200" : "text-slate-500"}>
          {presetName ?? "—"}
        </span>
      </Cell>
      <Cell label="engine">{engine}</Cell>
      <Cell label="mod">{modulation}</Cell>
      <Cell label="rate" unit="bps">
        {fmt(rate, 0)}
      </Cell>
      <Cell label="d" unit="m">
        {fmt(distance, 3)}
      </Cell>
      <Cell label="P_rx" unit="µW">
        {fmt(pRx, 2)}
      </Cell>
      <Cell label="SNR" unit="dB" tone={snrTone(snr)}>
        {fmt(snr, 1)}
      </Cell>
      <Cell label="BER" tone={berTone(ber)}>
        {fmtBer(ber)}
      </Cell>
      <span className="flex-1" />
      <Cell label="utc" align="right">
        {now}
      </Cell>
    </footer>
  );
}

function Cell({
  label,
  unit,
  children,
  tone,
  align,
}: {
  label: string;
  unit?: string;
  children: React.ReactNode;
  tone?: "ok" | "warn" | "fail" | "run";
  align?: "left" | "right";
}) {
  const toneClass =
    tone === "ok"
      ? "text-emerald-300"
      : tone === "warn"
        ? "text-harvest-300"
        : tone === "fail"
          ? "text-rose-400"
          : tone === "run"
            ? "text-beam-300"
            : "text-slate-100";

  return (
    <div
      className={
        "flex flex-col justify-center gap-px border-r border-hair px-2.5 py-1 " +
        (align === "right" ? "items-end text-right" : "items-start text-left")
      }
    >
      <span className="label-inst text-[9px]">
        {label}
        {unit ? <span className="ml-1 text-slate-600">{unit}</span> : null}
      </span>
      <span className={"text-[11px] leading-tight " + toneClass}>{children}</span>
    </div>
  );
}

function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function fmt(v: number | undefined | null, digits: number): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function fmtBer(v: number | undefined | null): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  if (v < 1e-4) return v.toExponential(1);
  return v.toFixed(4);
}

function labelForStatus(s: "idle" | "run" | "done" | "fail"): string {
  return s === "run" ? "RUNNING" : s === "done" ? "READY" : s === "fail" ? "FAILED" : "IDLE";
}

function statusTone(s: "idle" | "run" | "done" | "fail") {
  return s === "run" ? "run" : s === "done" ? "ok" : s === "fail" ? "fail" : undefined;
}

function snrTone(s: number | undefined | null): "ok" | "warn" | "fail" | undefined {
  if (s === null || s === undefined) return undefined;
  if (s >= 13) return "ok";
  if (s >= 8) return "warn";
  return "fail";
}

function berTone(b: number | undefined | null): "ok" | "warn" | "fail" | undefined {
  if (b === null || b === undefined) return undefined;
  if (b === 0 || b < 1e-4) return "ok";
  if (b < 1e-2) return "warn";
  return "fail";
}

function useNowEverySecond(): string {
  const [s, setS] = useState(() => formatNow());
  useEffect(() => {
    const id = window.setInterval(() => setS(formatNow()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return s;
}

function formatNow(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
}

/**
 * useSimulationSocket — opens /ws/pipeline on demand, streams events into the
 * pipelineStore and resultsStore. Returns a stable `runSimulation(config)`
 * callable plus a `cancel()` helper.
 *
 * Lifecycle:
 *   - Lazy-creates a WebSocket on first runSimulation() call
 *   - Replaces an existing run if runSimulation() is called while one is
 *     in flight (close + reconnect)
 *   - Cleans up on unmount
 */

import { useCallback, useEffect, useRef } from "react";

import type { ConfigDict } from "@/api/client";
import { wsUrl } from "@/api/ws";
import { useMessagesStore } from "@/store/messagesStore";
import { usePipelineStore, type StepName } from "@/store/pipelineStore";
import { useResultsStore } from "@/store/resultsStore";
import { useUIStore } from "@/store/uiStore";
import {
  useStandardsStore,
  type ComplianceResult,
} from "@/store/standardsStore";

type ServerMsg =
  | { type: "step"; name: StepName; status: "running" | "done" | "error"; message: string; duration_s?: number }
  | { type: "metrics"; [k: string]: unknown }
  | { type: "waveforms"; [k: string]: unknown }
  | ({ type: "compliance" } & ComplianceResult)
  | { type: "done"; engine: string; session_dir: string }
  | { type: "error"; message: string };

export function useSimulationSocket() {
  const wsRef = useRef<WebSocket | null>(null);

  const startRun = usePipelineStore((s) => s.startRun);
  const updateStep = usePipelineStore((s) => s.updateStep);
  const finishRun = usePipelineStore((s) => s.finishRun);
  const failRun = usePipelineStore((s) => s.failRun);
  const setMetrics = useResultsStore((s) => s.setMetrics);
  const setWaveforms = useResultsStore((s) => s.setWaveforms);
  const clearResults = useResultsStore((s) => s.clear);
  const setComplianceResult = useStandardsStore((s) => s.setResult);

  const closeSocket = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      ws.close();
    }
    wsRef.current = null;
  }, []);

  useEffect(() => closeSocket, [closeSocket]);

  const runSimulation = useCallback(
    (config: ConfigDict) => {
      closeSocket();
      clearResults();
      setComplianceResult(null);
      startRun();

      // Results + live log now have content — reveal their tabs.
      const reveal = useUIStore.getState().reveal;
      reveal("engine");
      reveal("inspector");

      const ws = new WebSocket(wsUrl("/ws/pipeline"));
      wsRef.current = ws;

      const activeProfile = useStandardsStore.getState().activeProfileId;

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            type: "start",
            config,
            ...(activeProfile ? { profile_id: activeProfile } : {}),
          }),
        );
      };

      ws.onmessage = (event) => {
        let msg: ServerMsg;
        try {
          msg = JSON.parse(event.data) as ServerMsg;
        } catch {
          return;
        }

        // Mirror every WS event into the messages store so the Inspector
        // can show a live log. Keep summaries short.
        const append = useMessagesStore.getState().append;
        append({
          level: msg.type === "error" ? "error" : "info",
          source: "ws/pipeline",
          type: msg.type,
          message: summarizeMsg(msg),
          detail: msg,
        });

        switch (msg.type) {
          case "step":
            updateStep(msg.name, {
              status: msg.status,
              message: msg.message,
              duration_s: msg.duration_s,
            });
            break;
          case "metrics": {
            // Narrow the write — strip the `type` discriminator and any
            // unexpected keys that would pollute the metrics card state.
            const m = msg as Record<string, unknown>;
            setMetrics({
              engine: m.engine as string | undefined,
              ber: numOrNull(m.ber),
              ber_n_errors: numOrNull(m.ber_n_errors),
              ber_n_bits: numOrNull(m.ber_n_bits),
              snr_est_dB: numOrNull(m.snr_est_dB),
              P_rx_avg_uW: numOrNull(m.P_rx_avg_uW),
              I_ph_avg_uA: numOrNull(m.I_ph_avg_uA),
              G_ch: numOrNull(m.G_ch),
              noise_breakdown: pickNoiseBreakdown(m.noise_breakdown),
            });
            break;
          }
          case "waveforms": {
            const w = msg as Record<string, unknown>;
            setWaveforms({
              time: asNumArray(w.time),
              P_tx: asNumArray(w.P_tx),
              P_rx: asNumArray(w.P_rx),
              I_ph: asNumArray(w.I_ph),
              V_rx: asNumArray(w.V_rx),
              bits_tx: asNumArray(w.bits_tx),
              bits_rx: asNumArray(w.bits_rx),
              modulation: typeof w.modulation === "string" ? w.modulation : undefined,
              ofdm_iq_real: w.ofdm_iq_real ? asNumArray(w.ofdm_iq_real) : undefined,
              ofdm_iq_imag: w.ofdm_iq_imag ? asNumArray(w.ofdm_iq_imag) : undefined,
              bfsk_e0: w.bfsk_e0 ? asNumArray(w.bfsk_e0) : undefined,
              bfsk_e1: w.bfsk_e1 ? asNumArray(w.bfsk_e1) : undefined,
            });
            break;
          }
          case "compliance": {
            const { type, ...rest } = msg;
            void type;
            setComplianceResult(rest);
            break;
          }
          case "done": {
            finishRun(msg.session_dir);
            // A completed run unlocks the post-hoc analysis tools.
            const r = useUIStore.getState().reveal;
            r("sweeps");
            r("ac");
            ws.close();
            break;
          }
          case "error":
            failRun(msg.message);
            ws.close();
            break;
        }
      };

      ws.onerror = () => failRun("WebSocket error — is the backend running?");
      ws.onclose = (e) => {
        // If the socket closes before a `done` event we should mark the run failed.
        const { running } = usePipelineStore.getState();
        if (running) failRun(e.reason || "connection closed");
      };
    },
    [
      closeSocket,
      clearResults,
      setComplianceResult,
      startRun,
      updateStep,
      setMetrics,
      setWaveforms,
      finishRun,
      failRun,
    ],
  );

  const cancel = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "cancel" }));
    }
    closeSocket();
    failRun("cancelled");
  }, [closeSocket, failRun]);

  return { runSimulation, cancel };
}

// One-line summary used by the Inspector messages log.
function summarizeMsg(msg: ServerMsg): string {
  switch (msg.type) {
    case "step":
      return `${msg.name}: ${msg.status}${msg.message ? ` — ${msg.message}` : ""}`;
    case "metrics": {
      const m = msg as Record<string, unknown>;
      const ber = typeof m.ber === "number" ? m.ber.toExponential(2) : "—";
      const snr = typeof m.snr_est_dB === "number"
        ? `${(m.snr_est_dB as number).toFixed(1)} dB`
        : "—";
      return `metrics: BER=${ber}, SNR=${snr}`;
    }
    case "waveforms": {
      const w = msg as Record<string, unknown>;
      const n = Array.isArray(w.time) ? w.time.length : 0;
      return `waveforms: ${n} samples`;
    }
    case "compliance":
      return `compliance: ${msg.passed ? "PASS" : "FAIL"}`;
    case "done":
      return `done (engine=${msg.engine})`;
    case "error":
      return `error: ${msg.message}`;
    default:
      return "(unknown event)";
  }
}

// Narrowing helpers — the backend WS sends loosely-typed JSON, and we'd
// rather drop unexpected fields than write them blindly into the store.
function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function asNumArray(v: unknown): number[] {
  return Array.isArray(v) ? v.filter((x): x is number => typeof x === "number") : [];
}

function pickNoiseBreakdown(
  v: unknown,
): NonNullable<import("@/store/resultsStore").Metrics["noise_breakdown"]> | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  const keys = [
    "shot_A2",
    "thermal_A2",
    "ambient_A2",
    "amplifier_A2",
    "adc_A2",
    "processing_A2",
    "total_A2",
    "total_std_A",
  ] as const;
  const out: Record<string, number> = {};
  for (const k of keys) {
    const n = numOrNull(o[k]);
    if (n === null) return null;
    out[k] = n;
  }
  return out as never;
}

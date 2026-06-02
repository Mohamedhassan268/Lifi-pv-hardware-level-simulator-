/**
 * useDuplexRun — runs a two-system (A + B) simulation as two sequential
 * pipeline passes over /ws/pipeline, collecting each system's final metrics
 * into duplexStore. Reuses the same WS protocol as useSimulationSocket but
 * promise-wraps a single pass so the two runs can be awaited in order.
 */

import { useCallback } from "react";

import type { ConfigDict } from "@/api/client";
import { wsUrl } from "@/api/ws";
import { useDuplexStore, type SysMetrics } from "@/store/duplexStore";
import { useUIStore } from "@/store/uiStore";

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Run one pipeline pass; resolve with its final metrics, reject on error. */
function runOnce(config: ConfigDict): Promise<SysMetrics> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl("/ws/pipeline"));
    let settled = false;
    let metrics: SysMetrics = { ber: null, snr_est_dB: null, P_rx_avg_uW: null };

    const done = (fn: () => void) => {
      if (settled) return;
      settled = true;
      try {
        ws.close();
      } catch {
        // ignore
      }
      fn();
    };

    ws.onopen = () => ws.send(JSON.stringify({ type: "start", config }));
    ws.onmessage = (event) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(event.data as string) as Record<string, unknown>;
      } catch {
        return;
      }
      if (msg.type === "metrics") {
        metrics = {
          engine: typeof msg.engine === "string" ? msg.engine : undefined,
          ber: numOrNull(msg.ber),
          snr_est_dB: numOrNull(msg.snr_est_dB),
          P_rx_avg_uW: numOrNull(msg.P_rx_avg_uW),
        };
      } else if (msg.type === "done") {
        done(() => resolve(metrics));
      } else if (msg.type === "error") {
        done(() => reject(new Error(typeof msg.message === "string" ? msg.message : "pipeline error")));
      }
    };
    ws.onerror = () => done(() => reject(new Error("WebSocket error — is the backend running?")));
    ws.onclose = () => done(() => reject(new Error("connection closed before completion")));
  });
}

export function useDuplexRun() {
  const runDuplex = useCallback(async (configA: ConfigDict, configB: ConfigDict) => {
    const d = useDuplexStore.getState();
    if (d.running) return;
    d.start();
    useUIStore.getState().reveal("engine");

    try {
      d.setCurrent("A");
      d.setResult("A", await runOnce({ ...configA, simulation_engine: "python" }));
      d.setCurrent("B");
      d.setResult("B", await runOnce({ ...configB, simulation_engine: "python" }));
      d.finish();
    } catch (e) {
      d.fail(e instanceof Error ? e.message : String(e));
    }
  }, []);

  return { runDuplex };
}

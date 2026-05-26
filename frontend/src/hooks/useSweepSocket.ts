/**
 * useSweepSocket — opens /ws/sweep on demand and streams points into
 * sweepStore. Returns runSweep(kind, preset, params) and cancel().
 */

import { useCallback, useEffect, useRef } from "react";

import { wsUrl } from "@/api/ws";
import {
  useSweepStore,
  type SweepKind,
  type SweepPoint,
  type ToleranceRow,
} from "@/store/sweepStore";

interface ThermalParams {
  temps_C: number[];
}

interface MonteCarloParams {
  n_runs: number;
  tolerance_spec?: Record<string, number>;
  auto_tolerance?: boolean;
  seed?: number;
}

interface BerSnrParams {
  n_points: number;
  ambient_lux_lo: number;
  ambient_lux_hi: number;
}

interface BerDistanceParams {
  n_points: number;
  distance_lo_m: number;
  distance_hi_m: number;
}

type SweepParams =
  | ThermalParams
  | MonteCarloParams
  | BerSnrParams
  | BerDistanceParams;

type ServerMsg =
  | { type: "started"; kind: SweepKind; total: number }
  | { type: "point"; i: number; total: number; data: SweepPoint }
  | { type: "done"; kind: SweepKind; count: number }
  | { type: "error"; message: string }
  | { type: "tolerance_resolved"; rows: ToleranceRow[] };

export function useSweepSocket() {
  const wsRef = useRef<WebSocket | null>(null);

  const start = useSweepStore((s) => s.start);
  const pushPoint = useSweepStore((s) => s.pushPoint);
  const finish = useSweepStore((s) => s.finish);
  const fail = useSweepStore((s) => s.fail);
  const setToleranceRows = useSweepStore((s) => s.setToleranceRows);

  const closeSocket = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState !== WebSocket.CLOSED) ws.close();
    wsRef.current = null;
  }, []);

  useEffect(() => closeSocket, [closeSocket]);

  const runSweep = useCallback(
    (kind: SweepKind, preset: string, params: SweepParams) => {
      closeSocket();
      start(kind, 0);

      const ws = new WebSocket(wsUrl("/ws/sweep"));
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: "start", kind, preset, params }));
      };

      ws.onmessage = (event) => {
        let msg: ServerMsg;
        try {
          msg = JSON.parse(event.data) as ServerMsg;
        } catch {
          return;
        }
        switch (msg.type) {
          case "started":
            start(msg.kind, msg.total);
            break;
          case "point":
            pushPoint(msg.data);
            break;
          case "tolerance_resolved":
            setToleranceRows(msg.rows);
            break;
          case "done":
            finish();
            ws.close();
            break;
          case "error":
            fail(msg.message);
            ws.close();
            break;
        }
      };

      ws.onerror = () => fail("WebSocket error — is the backend running?");
      ws.onclose = (e) => {
        const { running } = useSweepStore.getState();
        if (running) fail(e.reason || "connection closed");
      };
    },
    [closeSocket, start, pushPoint, finish, fail],
  );

  const cancel = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "cancel" }));
    }
    closeSocket();
    fail("cancelled");
  }, [closeSocket, fail]);

  return { runSweep, cancel };
}

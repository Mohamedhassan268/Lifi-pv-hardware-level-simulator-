/**
 * useAcSocket — opens /ws/ac on demand, sends the config + sweep params,
 * and writes the batch ac_result into acStore. The AC run is fast (<50 ms
 * server-side), so we don't stream per-point.
 */

import { useCallback, useEffect, useRef } from "react";

import type { ConfigDict } from "@/api/client";
import { wsUrl } from "@/api/ws";
import { useAcStore, type AcResult } from "@/store/acStore";

interface AcParams {
  f_start_hz?: number;
  f_stop_hz?: number;
  n_points?: number;
}

type ServerMsg =
  | { type: "started"; n_points: number }
  | ({ type: "ac_result" } & AcResult)
  | { type: "done" }
  | { type: "error"; message: string };

export function useAcSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const start = useAcStore((s) => s.start);
  const setResult = useAcStore((s) => s.setResult);
  const fail = useAcStore((s) => s.fail);

  const closeSocket = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState !== WebSocket.CLOSED) ws.close();
    wsRef.current = null;
  }, []);

  useEffect(() => closeSocket, [closeSocket]);

  const runAc = useCallback(
    (config: ConfigDict, params: AcParams = {}) => {
      closeSocket();
      start();
      const ws = new WebSocket(wsUrl("/ws/ac"));
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: "start", config, params }));
      };

      ws.onmessage = (event) => {
        let msg: ServerMsg;
        try {
          msg = JSON.parse(event.data) as ServerMsg;
        } catch {
          return;
        }
        if (msg.type === "ac_result") {
          const { type, ...rest } = msg;
          void type;
          setResult(rest);
        } else if (msg.type === "error") {
          fail(msg.message);
          ws.close();
        } else if (msg.type === "done") {
          ws.close();
        }
      };

      ws.onerror = () => fail("WebSocket error — is the backend running?");
      ws.onclose = (e) => {
        const { running } = useAcStore.getState();
        if (running) fail(e.reason || "connection closed");
      };
    },
    [closeSocket, start, setResult, fail],
  );

  return { runAc };
}

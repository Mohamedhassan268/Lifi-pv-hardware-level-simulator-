/**
 * useGlassBoxPipeline — owns the /ws/pipeline WebSocket connection for
 * stepwise + probe-fetch operation.
 *
 * The existing engine route uses its own WS hook for continuous runs; this
 * hook is the new entrypoint for the glass-box workflow:
 *   - sends {type:"start", mode:"stepwise"|"continuous", capture_probes:"all", ...}
 *   - listens for "step"/"paused"/"probes_available" JSON messages
 *   - listens for BINARY probe_data frames, decodes them, writes them to the
 *     pipelineExecutionStore probe cache
 *   - exposes step()/resume()/cancel() actions and a fetchProbe(id) action
 *     that requests a specific probe over the same socket
 */

import { useCallback, useEffect, useRef } from "react";

import { getWsBase } from "@/api/backend";
import { decodeBinaryFrame } from "@/api/binaryFrame";
import { useConfigStore } from "@/store/configStore";
import { usePipelineExecutionStore, type ExecutionMode } from "@/store/pipelineExecutionStore";
import type { ProbeStage } from "@/types/probes";

export interface UseGlassBoxPipeline {
  start: (mode: ExecutionMode, opts?: { profileId?: string }) => void;
  step: () => void;
  cancel: () => void;
  fetchProbe: (probeId: string) => void;
}

export function useGlassBoxPipeline(): UseGlassBoxPipeline {
  const wsRef = useRef<WebSocket | null>(null);
  const config = useConfigStore((s) => s.config);

  const startRun = usePipelineExecutionStore((s) => s.startRun);
  const setRunning = usePipelineExecutionStore((s) => s.setRunning);
  const setPausedAt = usePipelineExecutionStore((s) => s.setPausedAt);
  const addAvailableProbes = usePipelineExecutionStore((s) => s.addAvailableProbes);
  const cacheArrayProbe = usePipelineExecutionStore((s) => s.cacheArrayProbe);
  const cacheScalarProbe = usePipelineExecutionStore((s) => s.cacheScalarProbe);
  const reservePendingFetch = usePipelineExecutionStore((s) => s.reservePendingFetch);

  // Clean up on unmount.
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  const ensureSocket = useCallback((onOpen: (ws: WebSocket) => void) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      onOpen(wsRef.current);
      return;
    }
    const ws = new WebSocket(`${getWsBase()}/ws/pipeline`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.addEventListener("open", () => onOpen(ws));
    ws.addEventListener("message", (ev) => handleMessage(ev));
    ws.addEventListener("close", () => {
      setRunning(false);
      setPausedAt(null);
    });
    ws.addEventListener("error", () => {
      setRunning(false);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setRunning, setPausedAt]);

  function handleMessage(ev: MessageEvent) {
    if (ev.data instanceof ArrayBuffer) {
      try {
        const { header, payload } = decodeBinaryFrame(ev.data);
        if (header.type === "probe_data") {
          cacheArrayProbe(
            header.id,
            payload,
            header.shape,
            header.units,
            (header.stage as ProbeStage) || "RX",
            header.config_hash,
            header.suggest_decimate,
          );
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn("decodeBinaryFrame failed", e);
      }
      return;
    }

    // JSON text frame
    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(ev.data as string);
    } catch {
      return;
    }

    switch (msg.type) {
      case "paused":
        setPausedAt((msg.after as ProbeStage) ?? null);
        if (Array.isArray(msg.available_probes)) {
          addAvailableProbes(
            (msg.after as ProbeStage) ?? "RX",
            msg.available_probes as string[],
          );
        }
        break;
      case "probes_available":
        if (Array.isArray(msg.probe_ids)) {
          addAvailableProbes(
            (msg.stage as ProbeStage) ?? "RX",
            msg.probe_ids as string[],
          );
        }
        break;
      case "done":
        setRunning(false);
        setPausedAt(null);
        if (Array.isArray(msg.available_probes)) {
          addAvailableProbes("RX", msg.available_probes as string[]);
        }
        break;
      case "error":
        setRunning(false);
        setPausedAt(null);
        // eslint-disable-next-line no-console
        console.error("/ws/pipeline error:", msg.message);
        break;
      case "probe_data":
        // Scalar probes arrive as JSON.
        if (msg.kind === "scalar" && typeof msg.id === "string") {
          cacheScalarProbe(
            msg.id,
            Number(msg.value),
            String(msg.units ?? ""),
            (msg.stage as ProbeStage) ?? "RX",
            String(msg.config_hash ?? ""),
          );
        }
        break;
      default:
        // step / metrics / waveforms / compliance — handled by other hooks.
        break;
    }
  }

  const start = useCallback<UseGlassBoxPipeline["start"]>((mode, opts) => {
    const cfg = config ?? {};
    // Cheap client-side hash mirror of cosim.probes.config_hash.
    // We can't recompute the Python hash perfectly, so we let the server's
    // hash (returned in "metrics" / "done") become the authoritative tag.
    const placeholder = `local-${Date.now().toString(16)}`;
    startRun(placeholder);
    setRunning(true);
    ensureSocket((ws) => {
      ws.send(
        JSON.stringify({
          type: "start",
          mode,
          config: cfg,
          capture_probes: "all",
          ...(opts?.profileId ? { profile_id: opts.profileId } : {}),
        }),
      );
    });
  }, [config, ensureSocket, setRunning, startRun]);

  const step = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "step" }));
  }, []);

  const cancel = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "cancel" }));
  }, []);

  const fetchProbe = useCallback((probeId: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const rid = reservePendingFetch(probeId);
    wsRef.current.send(
      JSON.stringify({ type: "probe_fetch", id: probeId, request_id: rid }),
    );
  }, [reservePendingFetch]);

  return { start, step, cancel, fetchProbe };
}

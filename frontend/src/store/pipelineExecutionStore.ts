/**
 * pipelineExecutionStore — execution-control + probe state for the glass-box UI.
 *
 * Separate from pipelineStore (results-mirror) so that the result-graph code
 * doesn't accidentally get coupled to step/pause/probe-fetch concerns.
 *
 * Holds:
 *   - mode: "continuous" | "stepwise"
 *   - pausedAt: stage we're paused after, or null when running/idle
 *   - availableProbes: probe IDs the backend has finished capturing this run
 *   - registry: full ProbeSpec list fetched once from /api/probes
 *   - probeData: cached {arrays, scalars} keyed by (probe_id, config_hash)
 *   - openProbeWindows: list of probe IDs the user has opened a scope for
 *   - currentConfigHash: stamps every cached probe with the config it came from;
 *                       used to mark windows stale after a config edit
 */

import { create } from "zustand";

import type { ProbePayload } from "@/api/binaryFrame";
import type { ProbeSpec, ProbeStage } from "@/types/probes";

export type ExecutionMode = "continuous" | "stepwise";

interface ProbeCacheEntry {
  configHash: string;
  units: string;
  stage: ProbeStage;
  array?: ProbePayload;
  shape?: number[];
  scalar?: number;
  suggestDecimate?: boolean;
  fetchedAt: number;
}

interface PipelineExecutionState {
  // Execution control
  mode: ExecutionMode;
  running: boolean;
  pausedAt: ProbeStage | null;

  // Probe discovery
  registry: ProbeSpec[];
  availableProbes: string[];
  currentConfigHash: string | null;

  // Probe cache (id -> entry); cleared on new run with different config_hash
  probeData: Record<string, ProbeCacheEntry>;

  // UI: open scope windows. Order matters for tiling.
  openProbeWindows: string[];

  // Pending probe_fetch requests: request_id -> probe_id
  pendingFetches: Record<number, string>;
  nextRequestId: number;

  // ---- actions ----
  setMode: (m: ExecutionMode) => void;
  setRegistry: (specs: ProbeSpec[]) => void;
  startRun: (configHash: string) => void;
  setRunning: (running: boolean) => void;
  setPausedAt: (stage: ProbeStage | null) => void;
  addAvailableProbes: (stage: ProbeStage, ids: string[]) => void;
  cacheArrayProbe: (
    id: string, payload: ProbePayload, shape: number[],
    units: string, stage: ProbeStage, configHash: string,
    suggestDecimate?: boolean,
  ) => void;
  cacheScalarProbe: (
    id: string, value: number, units: string, stage: ProbeStage, configHash: string,
  ) => void;
  openProbe: (id: string) => void;
  closeProbe: (id: string) => void;
  reservePendingFetch: (id: string) => number;
  resolvePendingFetch: (requestId: number) => string | undefined;
  reset: () => void;
}

const initialState = {
  mode: "continuous" as ExecutionMode,
  running: false,
  pausedAt: null,
  registry: [] as ProbeSpec[],
  availableProbes: [] as string[],
  currentConfigHash: null as string | null,
  probeData: {} as Record<string, ProbeCacheEntry>,
  openProbeWindows: [] as string[],
  pendingFetches: {} as Record<number, string>,
  nextRequestId: 1,
};

export const usePipelineExecutionStore = create<PipelineExecutionState>((set, get) => ({
  ...initialState,

  setMode(m) {
    set({ mode: m });
  },

  setRegistry(specs) {
    set({ registry: specs });
  },

  startRun(configHash) {
    const prev = get().currentConfigHash;
    set({
      running: true,
      pausedAt: null,
      availableProbes: [],
      // Drop the cache if the config changed — previously-cached probes are stale.
      probeData: prev === configHash ? get().probeData : {},
      currentConfigHash: configHash,
      pendingFetches: {},
    });
  },

  setRunning(running) {
    set({ running });
  },

  setPausedAt(stage) {
    set({ pausedAt: stage });
  },

  addAvailableProbes(_stage, ids) {
    set((s) => {
      const merged = new Set([...s.availableProbes, ...ids]);
      return { availableProbes: Array.from(merged) };
    });
  },

  cacheArrayProbe(id, payload, shape, units, stage, configHash, suggestDecimate) {
    set((s) => ({
      probeData: {
        ...s.probeData,
        [id]: {
          configHash,
          units,
          stage,
          array: payload,
          shape,
          suggestDecimate,
          fetchedAt: Date.now(),
        },
      },
    }));
  },

  cacheScalarProbe(id, value, units, stage, configHash) {
    set((s) => ({
      probeData: {
        ...s.probeData,
        [id]: {
          configHash, units, stage, scalar: value, fetchedAt: Date.now(),
        },
      },
    }));
  },

  openProbe(id) {
    set((s) =>
      s.openProbeWindows.includes(id)
        ? s
        : { openProbeWindows: [...s.openProbeWindows, id] },
    );
  },

  closeProbe(id) {
    set((s) => ({
      openProbeWindows: s.openProbeWindows.filter((p) => p !== id),
    }));
  },

  reservePendingFetch(id) {
    const rid = get().nextRequestId;
    set((s) => ({
      pendingFetches: { ...s.pendingFetches, [rid]: id },
      nextRequestId: rid + 1,
    }));
    return rid;
  },

  resolvePendingFetch(requestId) {
    const id = get().pendingFetches[requestId];
    if (!id) return undefined;
    set((s) => {
      const next = { ...s.pendingFetches };
      delete next[requestId];
      return { pendingFetches: next };
    });
    return id;
  },

  reset() {
    set({ ...initialState, registry: get().registry });
  },
}));

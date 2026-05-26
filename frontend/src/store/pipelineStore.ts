/**
 * pipelineStore — mirrors the 3-step pipeline state from cosim/pipeline.py.
 * The WebSocket hook writes here; PipelineCards reads.
 */

import { create } from "zustand";

import type { Status } from "@/primitives/StatusDot";

export type StepName = "TX" | "Channel" | "RX";

export interface PipelineStep {
  name: StepName;
  status: Status;
  message: string;
  duration_s?: number;
}

interface PipelineState {
  running: boolean;
  steps: Record<StepName, PipelineStep>;
  error: string | null;
  sessionDir: string | null;

  startRun: () => void;
  updateStep: (name: StepName, patch: Partial<PipelineStep>) => void;
  finishRun: (sessionDir: string | null) => void;
  failRun: (message: string) => void;
  reset: () => void;
}

const initialSteps = (): Record<StepName, PipelineStep> => ({
  TX: { name: "TX", status: "idle", message: "" },
  Channel: { name: "Channel", status: "idle", message: "" },
  RX: { name: "RX", status: "idle", message: "" },
});

export const usePipelineStore = create<PipelineState>((set) => ({
  running: false,
  steps: initialSteps(),
  error: null,
  sessionDir: null,

  startRun() {
    set({ running: true, steps: initialSteps(), error: null, sessionDir: null });
  },

  updateStep(name, patch) {
    set((s) => ({
      steps: { ...s.steps, [name]: { ...s.steps[name], ...patch } },
    }));
  },

  finishRun(sessionDir) {
    set({ running: false, sessionDir });
  },

  failRun(message) {
    set({ running: false, error: message });
  },

  reset() {
    set({ running: false, steps: initialSteps(), error: null, sessionDir: null });
  },
}));

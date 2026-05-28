/**
 * messagesStore — ring buffer of WebSocket events emitted by the backend
 * during a simulation run. Used by the InspectorRoute's Messages tab and
 * any future "live log" view.
 *
 * Kept small (MAX 200 entries) so the React DevTools / Zustand snapshot stays
 * cheap. Older entries are dropped when the buffer is full.
 */

import { create } from "zustand";

const MAX = 200;

export type LogLevel = "info" | "warn" | "error" | "debug";

export interface LogEntry {
  id: number;
  ts: number;            // epoch ms
  level: LogLevel;
  source: string;        // e.g. "ws/pipeline", "validator", "client"
  type: string;          // event subtype, e.g. "step", "metrics", "compliance"
  message: string;       // short human-readable summary
  detail?: unknown;      // raw payload for the expanded row
}

interface MessagesState {
  entries: LogEntry[];
  nextId: number;
  append: (e: Omit<LogEntry, "id" | "ts">) => void;
  clear: () => void;
}

export const useMessagesStore = create<MessagesState>((set, get) => ({
  entries: [],
  nextId: 1,
  append: (e) => {
    const id = get().nextId;
    const entry: LogEntry = { ...e, id, ts: Date.now() };
    set((s) => {
      const next = s.entries.length >= MAX
        ? [...s.entries.slice(s.entries.length - MAX + 1), entry]
        : [...s.entries, entry];
      return { entries: next, nextId: id + 1 };
    });
  },
  clear: () => set({ entries: [], nextId: 1 }),
}));

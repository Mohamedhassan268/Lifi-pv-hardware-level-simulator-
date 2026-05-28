/**
 * MessagesTab — tails the WebSocket event log captured in messagesStore.
 * Most recent first; click any row to expand the raw payload.
 */

import { useState } from "react";

import { Card } from "@/primitives/Card";
import { type LogEntry, useMessagesStore } from "@/store/messagesStore";

const LEVEL_CLASS: Record<LogEntry["level"], string> = {
  error: "text-rose-300",
  warn: "text-amber-200",
  info: "text-slate-300",
  debug: "text-slate-500",
};

export function MessagesTab() {
  const entries = useMessagesStore((s) => s.entries);
  const clear = useMessagesStore((s) => s.clear);
  const [expanded, setExpanded] = useState<number | null>(null);

  if (entries.length === 0) {
    return (
      <Card>
        <p className="text-sm text-slate-400">
          No backend events captured yet. Run a simulation to start the log.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>{entries.length} event{entries.length === 1 ? "" : "s"}</span>
        <button
          type="button"
          onClick={clear}
          className="rounded border border-slate-700 px-2 py-1 hover:border-slate-500 hover:text-slate-200"
        >
          Clear
        </button>
      </div>
      <Card>
        <ul className="divide-y divide-slate-800/80">
          {[...entries].reverse().map((entry) => {
            const isOpen = expanded === entry.id;
            return (
              <li key={entry.id} className="py-2">
                <button
                  type="button"
                  onClick={() => setExpanded(isOpen ? null : entry.id)}
                  className="w-full text-left"
                >
                  <div className="flex items-center gap-3 font-mono text-xs">
                    <span className="text-slate-500">{formatTs(entry.ts)}</span>
                    <span className={`uppercase ${LEVEL_CLASS[entry.level]}`}>
                      {entry.level}
                    </span>
                    <span className="text-slate-400">{entry.source}</span>
                    <span className="text-slate-500">·</span>
                    <span className="text-slate-300">{entry.type}</span>
                    <span className="ml-2 truncate text-slate-200">
                      {entry.message}
                    </span>
                  </div>
                </button>
                {isOpen && entry.detail !== undefined ? (
                  <pre className="mt-2 max-h-72 overflow-auto rounded bg-slate-900/60 p-2 text-[11px] text-slate-300">
                    {safeJson(entry.detail)}
                  </pre>
                ) : null}
              </li>
            );
          })}
        </ul>
      </Card>
    </div>
  );
}

function formatTs(ts: number): string {
  const d = new Date(ts);
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${d
    .getMilliseconds()
    .toString()
    .padStart(3, "0")}`;
}

function safeJson(v: unknown): string {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

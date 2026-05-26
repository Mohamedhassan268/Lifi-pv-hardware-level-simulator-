/**
 * StatusDot — color-coded indicator matching gui/widgets.py StatusIndicator
 * (idle | running | done | error).
 */

export type Status = "idle" | "running" | "done" | "error";

const COLORS: Record<Status, string> = {
  idle: "bg-slate-500",
  running: "bg-beam-400 animate-pulse",
  done: "bg-emerald-400",
  error: "bg-rose-500",
};

export function StatusDot({ status, size = 8 }: { status: Status; size?: number }) {
  return (
    <span
      aria-label={status}
      className={`inline-block rounded-full ${COLORS[status]}`}
      style={{ width: size, height: size }}
    />
  );
}

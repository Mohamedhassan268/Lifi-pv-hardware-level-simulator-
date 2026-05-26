/**
 * SweepProgress — thin progress bar + count. Lives next to SweepControls so
 * the user always sees how far the sweep has gotten without scrolling.
 */

import { motion } from "framer-motion";

import { DUR, EASE } from "@/lib/motion";
import { Card } from "@/primitives/Card";
import { useSweepStore } from "@/store/sweepStore";

export function SweepProgress() {
  const total = useSweepStore((s) => s.total);
  const done = useSweepStore((s) => s.points.length);
  const running = useSweepStore((s) => s.running);
  const error = useSweepStore((s) => s.error);
  const kind = useSweepStore((s) => s.kind);

  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;

  return (
    <Card>
      <header className="mb-3 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Progress
        </h3>
        <p className="font-mono text-xs text-slate-400">
          {done} / {total || "—"}
          {kind && ` · ${kind === "thermal" ? "thermal" : "Monte Carlo"}`}
        </p>
      </header>

      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
        <motion.div
          className="h-full bg-beam-400 shadow-[0_0_12px_rgba(34,211,238,0.5)] will-change-transform"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: DUR.fast, ease: EASE }}
        />
      </div>

      {error && (
        <p className="mt-3 text-sm text-rose-400">{error}</p>
      )}

      {!running && total > 0 && !error && done === total && (
        <p className="mt-3 text-xs text-emerald-300">Sweep complete.</p>
      )}
    </Card>
  );
}

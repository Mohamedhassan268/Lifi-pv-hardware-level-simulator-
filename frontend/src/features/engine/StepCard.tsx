/**
 * StepCard — single pipeline step (TX/Channel/RX). Shares `layoutId` so the
 * 3 cards animate via FLIP when the running step changes position.
 */

import { motion } from "framer-motion";

import { DUR, EASE } from "@/lib/motion";
import { StatusDot, type Status } from "@/primitives/StatusDot";

interface StepCardProps {
  name: string;
  status: Status;
  message: string;
  duration_s?: number;
  active: boolean;
}

export function StepCard({ name, status, message, duration_s, active }: StepCardProps) {
  return (
    <motion.li
      layout
      layoutId={`step-${name}`}
      transition={{ duration: DUR.base, ease: EASE }}
      className={
        "relative flex flex-col gap-2 rounded-2xl border p-5 will-change-transform " +
        (active
          ? "border-beam-400/60 bg-beam-400/[0.06] shadow-[0_0_24px_-8px_rgba(34,211,238,0.45)]"
          : "border-white/10 bg-white/[0.03]")
      }
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
          {name}
        </span>
        <StatusDot status={status} />
      </div>
      <p className="text-sm text-slate-200">{message || "—"}</p>
      {typeof duration_s === "number" && duration_s > 0 && (
        <p className="font-mono text-xs text-slate-500">
          {duration_s.toFixed(2)} s
        </p>
      )}
    </motion.li>
  );
}

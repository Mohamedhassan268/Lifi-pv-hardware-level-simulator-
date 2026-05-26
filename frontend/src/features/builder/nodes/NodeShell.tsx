/**
 * NodeShell — shared card chrome for the three category nodes (TX / Channel
 * / RX). Renders the title, subtitle, step status dot, and chip strip.
 * The unique SVG illustration is passed as `children`.
 *
 * States driven by `data.configured`:
 *   - true:  full-colour card + chips + accent shadow
 *   - false: dashed muted card, no chips, italic "Not configured" subtitle
 */

import { Handle, Position } from "@xyflow/react";
import { motion } from "framer-motion";
import { type ReactNode } from "react";

import { usePipelineStore, type StepName } from "@/store/pipelineStore";

export interface NodeShellProps {
  title: string;
  subtitle: string;
  chips: string[];
  configured: boolean;
  step?: StepName;
  accent: "beam" | "slate" | "harvest";
  children: ReactNode; // the SVG illustration
  showHandles?: { left?: boolean; right?: boolean };
}

export function NodeShell({
  title,
  subtitle,
  chips,
  configured,
  step,
  accent,
  children,
  showHandles = { left: true, right: true },
}: NodeShellProps) {
  const stepStatus = usePipelineStore((s) =>
    step ? s.steps[step].status : "idle",
  );

  const borderClass = !configured
    ? "border-white/10 border-dashed bg-slate-950/70"
    : accent === "beam"
      ? "border-beam-400/40 bg-slate-950/85 shadow-[0_0_42px_-12px_rgba(34,211,238,0.45)]"
      : accent === "harvest"
        ? "border-harvest-400/40 bg-slate-950/85 shadow-[0_0_42px_-12px_rgba(251,191,36,0.4)]"
        : "border-white/15 bg-slate-950/85";

  const statusDot =
    stepStatus === "running"
      ? "bg-beam-300 animate-pulse"
      : stepStatus === "done"
        ? "bg-emerald-400"
        : stepStatus === "error"
          ? "bg-rose-400"
          : "bg-slate-600";

  return (
    <motion.div
      layout
      className={`relative w-[260px] rounded-2xl border px-4 py-3 backdrop-blur ${borderClass}`}
    >
      {showHandles.left && (
        <Handle
          type="target"
          position={Position.Left}
          className="!h-2 !w-2 !border-0 !bg-white/30"
        />
      )}

      <header className="flex items-center justify-between gap-2">
        <h3
          className={`text-sm font-semibold ${
            configured ? "text-slate-100" : "text-slate-400"
          }`}
        >
          {title}
        </h3>
        {step && configured && (
          <span className={`inline-block h-2 w-2 rounded-full ${statusDot}`} />
        )}
      </header>

      <p
        className={`mt-0.5 text-[10px] uppercase tracking-wider ${
          configured ? "text-slate-500" : "italic text-slate-600"
        }`}
      >
        {subtitle}
      </p>

      {/* Illustration area */}
      <div className="mt-2 flex items-center justify-center">{children}</div>

      {configured && chips.length > 0 && (
        <ul className="mt-2 flex flex-wrap gap-1">
          {chips.map((c) => (
            <li
              key={c}
              className="rounded-md border border-white/10 bg-white/[0.04] px-1.5 py-0.5 text-[10px] text-slate-300"
            >
              {c}
            </li>
          ))}
        </ul>
      )}

      {showHandles.right && (
        <Handle
          type="source"
          position={Position.Right}
          className="!h-2 !w-2 !border-0 !bg-white/30"
        />
      )}
    </motion.div>
  );
}

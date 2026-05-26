/**
 * InspectorSection — collapsible accordion section for the docked inspector.
 *
 * Instrument-style chrome: hairline borders, near-square corners, dense
 * padding, monospace formula line. Each section can optionally show an enable
 * toggle and a contribution-percentage readout in its header.
 */

import { useState, type ReactNode } from "react";

interface InspectorSectionProps {
  index?: string;
  title: string;
  formula?: string;
  enabled?: boolean;
  onEnabledChange?: (v: boolean) => void;
  contribution?: number | null;
  defaultOpen?: boolean;
  children: ReactNode;
}

export function InspectorSection({
  index,
  title,
  formula,
  enabled,
  onEnabledChange,
  contribution,
  defaultOpen = false,
  children,
}: InspectorSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const canToggleEnable = typeof enabled === "boolean" && onEnabledChange;
  const off = canToggleEnable && enabled === false;

  return (
    <section
      className={
        "border-l-2 " +
        (off
          ? "border-l-slate-700/60 bg-slate-900/30"
          : enabled
            ? "border-l-beam-400/60 bg-slate-900/40"
            : "border-l-slate-700/60 bg-slate-900/30")
      }
    >
      <header
        className="flex cursor-pointer items-center gap-2 border-b border-hair px-2 py-1.5 hover:bg-white/[0.02]"
        onClick={() => setOpen(!open)}
      >
        <span className="readout w-3 select-none text-[10px] text-slate-500">
          {open ? "▾" : "▸"}
        </span>
        {index && (
          <span className="readout w-4 text-[10px] text-slate-500">{index}</span>
        )}
        <span className="label-inst flex-1 truncate">{title}</span>
        {typeof contribution === "number" && contribution >= 0 && (
          <span className="readout w-10 text-right text-[10px] text-slate-400">
            {contribution.toFixed(0)}%
          </span>
        )}
        {canToggleEnable && (
          <button
            type="button"
            role="switch"
            aria-checked={enabled}
            onClick={(e) => {
              e.stopPropagation();
              onEnabledChange?.(!enabled);
            }}
            className={
              "relative h-3.5 w-7 shrink-0 border transition-colors " +
              (enabled
                ? "border-beam-400/60 bg-beam-400/30"
                : "border-white/15 bg-white/[0.05]")
            }
          >
            <span
              className={
                "absolute top-0.5 h-2 w-2 transition-transform " +
                (enabled ? "bg-beam-300" : "bg-white/60")
              }
              style={{ transform: `translateX(${enabled ? 16 : 2}px)` }}
            />
          </button>
        )}
      </header>

      {open && (
        <div className="px-2 py-2">
          {formula && (
            <p className="readout mb-2 border-b border-hair pb-1.5 text-[10px] text-slate-500">
              <span className="text-slate-600">fn</span>{" "}
              <span className="text-slate-400">{formula}</span>
            </p>
          )}
          <div className={"space-y-2 " + (off ? "opacity-40" : "")}>{children}</div>
        </div>
      )}
    </section>
  );
}

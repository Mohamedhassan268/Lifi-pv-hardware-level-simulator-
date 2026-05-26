/**
 * ProbeWindow — single probe scope panel.
 *
 * Owns:
 *   - the header (probe ID, stage, units, sample count, config_hash)
 *   - the ProbeScope visual
 *   - the "Equation" affordance which surfaces the linked ScienceCard
 *   - a close button that removes itself from pipelineExecutionStore
 *
 * The window does NOT fetch probe data — that's driven by the pipeline WS
 * handler which calls `cacheArrayProbe` in the store. The window only renders
 * whatever's cached. If the probe isn't cached yet (user opened the window
 * but the fetch hasn't returned), the window shows a "Loading…" state.
 */

import { useMemo, useState } from "react";

import { usePipelineExecutionStore } from "@/store/pipelineExecutionStore";

import { ProbeScope } from "./ProbeScope";
import { ScienceCard } from "@/components/ScienceCard";

interface ProbeWindowProps {
  probeId: string;
  /** Optional time-axis array shared across all probes of a run. */
  timeAxis?: Float64Array | null;
}

export function ProbeWindow({ probeId, timeAxis = null }: ProbeWindowProps) {
  const close = usePipelineExecutionStore((s) => s.closeProbe);
  const entry = usePipelineExecutionStore((s) => s.probeData[probeId]);
  const spec = usePipelineExecutionStore((s) =>
    s.registry.find((p) => p.id === probeId),
  );
  const currentHash = usePipelineExecutionStore((s) => s.currentConfigHash);

  const [showEquation, setShowEquation] = useState(false);

  const stale = useMemo(() => {
    if (!entry || !currentHash) return false;
    return entry.configHash !== currentHash;
  }, [entry, currentHash]);

  if (!spec) {
    return (
      <PanelShell title={probeId} subtitle="unknown probe" onClose={() => close(probeId)} />
    );
  }

  // Probe is open but no data yet — fetch will arrive over WS.
  if (!entry) {
    return (
      <PanelShell
        title={spec.label}
        subtitle={`${spec.id} · ${spec.units}`}
        stage={spec.stage}
        onClose={() => close(probeId)}
      >
        <div className="flex h-[220px] items-center justify-center text-xs text-slate-500">
          Awaiting probe data…
        </div>
      </PanelShell>
    );
  }

  return (
    <PanelShell
      title={spec.label}
      subtitle={`${spec.id} · units: ${entry.units}${
        entry.shape ? ` · N=${entry.shape[entry.shape.length - 1]}` : ""
      }`}
      stage={spec.stage}
      stale={stale}
      onClose={() => close(probeId)}
      action={
        spec.equation_key ? (
          <button
            type="button"
            onClick={() => setShowEquation((v) => !v)}
            className="px-2 py-0.5 text-[10px] uppercase tracking-[0.15em] text-cyan-300 hover:text-cyan-100"
          >
            {showEquation ? "Hide eq." : "Equation"}
          </button>
        ) : null
      }
    >
      {entry.scalar !== undefined ? (
        <div className="flex h-[220px] flex-col items-center justify-center gap-2">
          <div className="font-mono text-3xl text-cyan-200 tabular-nums">
            {formatScalar(entry.scalar)}
          </div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
            {entry.units}
          </div>
        </div>
      ) : entry.array ? (
        <ProbeScope
          x={timeAxis}
          y={entry.array}
          units={entry.units}
          label={spec.label}
        />
      ) : null}

      {entry.suggestDecimate && entry.array && (
        <div className="border-t border-slate-800 px-2 py-1 text-[10px] uppercase tracking-[0.15em] text-amber-300">
          Decimated view — full {entry.array.length.toLocaleString()} samples retained for export.
        </div>
      )}

      {showEquation && spec.equation_key && (
        <div className="border-t border-slate-800 p-2">
          <ScienceCard equationKey={spec.equation_key} compact />
        </div>
      )}
    </PanelShell>
  );
}

// ---------------------------------------------------------------------------
// PanelShell — engineering-instrument aesthetic (monospace, hairline chrome)
// ---------------------------------------------------------------------------

interface PanelShellProps {
  title: string;
  subtitle?: string;
  stage?: string;
  stale?: boolean;
  onClose: () => void;
  action?: React.ReactNode;
  children?: React.ReactNode;
}

function PanelShell({
  title, subtitle, stage, stale, onClose, action, children,
}: PanelShellProps) {
  return (
    <div className="border border-slate-800/80 bg-slate-950/80 font-mono text-slate-200">
      <header className="flex items-center gap-3 border-b border-slate-800 px-3 py-1.5">
        {stage && (
          <span className="text-[9px] uppercase tracking-[0.2em] text-slate-500">
            {stage}
          </span>
        )}
        <div className="flex-1">
          <div className="text-xs text-slate-100">{title}</div>
          {subtitle && (
            <div className="text-[10px] uppercase tracking-[0.12em] text-slate-500">
              {subtitle}
            </div>
          )}
        </div>
        {stale && (
          <span className="border border-amber-500/30 px-1.5 py-0.5 text-[9px] uppercase tracking-[0.18em] text-amber-300">
            Stale
          </span>
        )}
        {action}
        <button
          type="button"
          onClick={onClose}
          className="text-slate-500 hover:text-slate-200"
          aria-label="Close probe"
        >
          ×
        </button>
      </header>
      {children}
    </div>
  );
}

function formatScalar(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1e6 || abs < 1e-3) return v.toExponential(4);
  return v.toPrecision(6);
}

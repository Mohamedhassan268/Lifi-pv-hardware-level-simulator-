/**
 * ScienceCard — render an equation from the backend registry with live values.
 *
 * Fetches `POST /api/equations/{key}` with the current SystemConfig, then
 * renders:
 *   1. The LaTeX form (KaTeX, serif math)
 *   2. A symbol table: symbol | name | value | units | source
 *   3. Citation list
 *
 * The aesthetic stays engineering-instrument: monospace chrome around the
 * KaTeX block, hairline separators, tabular numerics, no consumer polish.
 *
 * This proves to the user that the same equation displayed in the UI is the
 * one the engine is actually evaluating — the symbol values come from the
 * exact same Python functions (`channel.los_gain`, `noise.shot_noise_variance`,
 * etc.) that the pipeline calls.
 */

import "katex/dist/katex.min.css";

import katex from "katex";
import { useEffect, useMemo, useState } from "react";

import { getBackendBase } from "@/api/backend";
import { useConfigStore } from "@/store/configStore";

interface SymbolRow {
  symbol: string;
  name: string;
  units: string;
  source: string;
  value: number | null;
}

interface EquationPayload {
  key: string;
  title: string;
  latex: string;
  plain: string;
  symbols: SymbolRow[];
  references: string[];
}

export interface ScienceCardProps {
  equationKey: string;
  /** Compact mode hides title and references — used inside ProbeWindow tabs. */
  compact?: boolean;
}

export function ScienceCard({ equationKey, compact = false }: ScienceCardProps) {
  const config = useConfigStore((s) => s.config);
  const [payload, setPayload] = useState<EquationPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setPayload(null);
    (async () => {
      try {
        const res = await fetch(`${getBackendBase()}/api/equations/${equationKey}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config ?? {}),
        });
        if (!res.ok) {
          setError(`HTTP ${res.status}`);
          return;
        }
        const data = (await res.json()) as EquationPayload;
        if (!cancelled) setPayload(data);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [equationKey, config]);

  if (error) {
    return (
      <div className="border border-red-900/40 bg-red-950/40 p-2 text-[10px] uppercase tracking-[0.15em] text-red-300">
        ScienceCard error: {error}
      </div>
    );
  }
  if (!payload) {
    return (
      <div className="border border-slate-800/80 bg-slate-950/60 p-3 font-mono text-[10px] uppercase tracking-[0.15em] text-slate-500">
        Resolving {equationKey}…
      </div>
    );
  }

  return (
    <div className="border border-slate-800/80 bg-slate-950/60 font-mono text-slate-200">
      {!compact && (
        <header className="border-b border-slate-800 px-3 py-1.5">
          <div className="text-[9px] uppercase tracking-[0.22em] text-slate-500">
            {payload.key}
          </div>
          <div className="text-xs text-slate-100">{payload.title}</div>
        </header>
      )}

      <div className="overflow-x-auto bg-slate-950 px-3 py-3 text-slate-100">
        <KatexBlock latex={payload.latex} />
      </div>

      <table className="w-full table-fixed border-t border-slate-800 text-[11px]">
        <thead className="text-[9px] uppercase tracking-[0.18em] text-slate-500">
          <tr className="border-b border-slate-800">
            <th className="px-2 py-1 text-left">Sym</th>
            <th className="px-2 py-1 text-left">Name</th>
            <th className="px-2 py-1 text-right">Value</th>
            <th className="px-2 py-1 text-left">Units</th>
            <th className="px-2 py-1 text-left">Source</th>
          </tr>
        </thead>
        <tbody className="text-slate-300">
          {payload.symbols.map((row) => (
            <tr key={row.symbol} className="border-b border-slate-900/80">
              <td className="px-2 py-1 font-medium text-cyan-200">{row.symbol}</td>
              <td className="px-2 py-1">{row.name}</td>
              <td className="px-2 py-1 text-right tabular-nums">
                {row.value === null ? "—" : formatValue(row.value)}
              </td>
              <td className="px-2 py-1 text-slate-400">{row.units}</td>
              <td className="px-2 py-1 text-[10px] text-slate-500">{row.source}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {!compact && payload.references.length > 0 && (
        <footer className="border-t border-slate-800 px-3 py-1.5 text-[10px] text-slate-500">
          {payload.references.map((r) => (
            <div key={r}>· {r}</div>
          ))}
        </footer>
      )}
    </div>
  );
}

function formatValue(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1e6 || abs < 1e-3) return v.toExponential(4);
  return v.toPrecision(6);
}

/**
 * Thin KaTeX renderer — uses katex.renderToString directly so we never
 * pull in react-katex (UMD/CommonJS, breaks Vite dev pre-bundling).
 */
function KatexBlock({ latex }: { latex: string }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(latex, {
        displayMode: true,
        throwOnError: false,
        output: "html",
      });
    } catch (e) {
      return `<span class="text-red-300">${String(e)}</span>`;
    }
  }, [latex]);
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

/**
 * ToleranceTable — read-only table of the auto-derived tolerance spec
 * (one row per SystemConfig field, with the source component and a
 * "datasheet/process/estimate" tag). Visible while a Monte Carlo run
 * with `auto_tolerance: true` is in flight or just completed.
 */

import { Card } from "@/primitives/Card";
import { useSweepStore } from "@/store/sweepStore";

export function ToleranceTable() {
  const rows = useSweepStore((s) => s.toleranceRows);

  if (!rows || rows.length === 0) return null;

  return (
    <Card>
      <header className="mb-3 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Tolerance spec (auto)
        </h3>
        <span className="text-[11px] text-slate-500">{rows.length} fields</span>
      </header>
      <table className="w-full text-left">
        <thead>
          <tr className="text-[10px] uppercase tracking-wider text-slate-500">
            <th className="pb-1">Field</th>
            <th className="pb-1 text-right">±%</th>
            <th className="pb-1">Source</th>
            <th className="pb-1">Via</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.field} className="border-t border-white/5 text-xs text-slate-300">
              <td className="py-1.5 font-mono">{r.field}</td>
              <td className="py-1.5 text-right font-mono">
                {(r.fractional * 100).toFixed(1)}
              </td>
              <td className="py-1.5">
                <SourceChip source={r.source} />
              </td>
              <td className="py-1.5 text-slate-400">{r.via}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function SourceChip({ source }: { source: string }) {
  const color =
    source === "datasheet"
      ? "border-emerald-400/40 text-emerald-300"
      : source === "process"
        ? "border-beam-400/40 text-beam-300"
        : "border-white/20 text-slate-400";
  return (
    <span
      className={`inline-block rounded border bg-white/[0.04] px-1.5 py-0.5 text-[10px] uppercase tracking-wider ${color}`}
    >
      {source}
    </span>
  );
}

/**
 * ComplianceBanner — PASS/FAIL banner shown above the ResultsPanel in
 * the EngineRoute when a standards profile was active for the last run.
 * Lists each criterion's outcome and surfaces the failing ones.
 */

import { motion } from "framer-motion";

import { DUR, EASE } from "@/lib/motion";
import { useStandardsStore, type CriterionResult } from "@/store/standardsStore";

export function ComplianceBanner() {
  const result = useStandardsStore((s) => s.lastResult);
  if (!result) return null;

  const tone = result.passed
    ? "border-emerald-400/40 bg-emerald-400/[0.08]"
    : "border-rose-400/40 bg-rose-400/[0.08]";
  const dot = result.passed ? "bg-emerald-400" : "bg-rose-400";

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: DUR.base, ease: EASE }}
      className={`mb-6 rounded-2xl border p-5 backdrop-blur-sm ${tone}`}
      role="status"
    >
      <header className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className={`inline-block h-2.5 w-2.5 rounded-full ${dot}`} />
          <h3 className="text-sm font-semibold uppercase tracking-wider">
            {result.passed ? "Compliant" : "Non-compliant"}
          </h3>
          <span className="text-xs text-slate-300">{result.profile_label}</span>
        </div>
        <span className="text-[11px] text-slate-400">{result.spec_section}</span>
      </header>

      <ul className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {result.results.map((r) => (
          <CriterionRow key={r.label} r={r} />
        ))}
      </ul>
    </motion.div>
  );
}

function CriterionRow({ r }: { r: CriterionResult }) {
  return (
    <li
      className={`flex items-start justify-between gap-3 rounded-lg border px-3 py-2 text-xs ${
        r.ok
          ? "border-emerald-400/30 bg-white/[0.02]"
          : "border-rose-400/40 bg-rose-400/[0.05]"
      }`}
    >
      <div className="min-w-0">
        <p className="font-mono text-slate-100">{r.label}</p>
        {r.detail && <p className="mt-0.5 text-[10px] text-slate-400">{r.detail}</p>}
      </div>
      <div className="shrink-0 text-right">
        <p className={`text-[10px] font-semibold uppercase tracking-wider ${
          r.ok ? "text-emerald-300" : "text-rose-300"
        }`}>
          {r.ok ? "PASS" : "FAIL"}
        </p>
        {r.actual !== null && (
          <p className="font-mono text-[11px] text-slate-200">
            {fmt(r.actual)}
            {r.bound !== null && (
              <span className="text-slate-500"> / {fmt(r.bound)}</span>
            )}
          </p>
        )}
      </div>
    </li>
  );
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) < 1e-3 && v !== 0) return v.toExponential(2);
  return v.toFixed(2);
}

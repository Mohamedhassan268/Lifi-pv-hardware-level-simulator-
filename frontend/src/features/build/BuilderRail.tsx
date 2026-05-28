/**
 * BuilderRail — right-side persistent panel for the Builder. Two cards:
 *   1. Live link-budget summary (reuses LinkBudgetTable with source="draft")
 *   2. Validation issues list (debounced /api/config/validate)
 *
 * Issues are grouped by level (ERROR / WARNING / INFO) and color-coded.
 * ERROR-level issues gate the Run button via `useBuilderValidity`.
 */

import { motion } from "framer-motion";

import type { ValidationIssue, ValidationIssueLevel } from "@/api/client";
import { KicadExportCard } from "@/features/build/KicadExportCard";
import { LinkBudgetTable } from "@/features/setup/LinkBudgetTable";
import { useDebouncedValidate } from "@/hooks/useDebouncedValidate";
import { DUR, EASE } from "@/lib/motion";
import { Card } from "@/primitives/Card";
import { useConfigStore } from "@/store/configStore";

const LEVEL_STYLES: Record<
  ValidationIssueLevel,
  { border: string; bg: string; text: string; badge: string; label: string }
> = {
  error: {
    border: "border-rose-500/30",
    bg: "bg-rose-500/5",
    text: "text-rose-300",
    badge: "bg-rose-500/20 text-rose-200",
    label: "ERROR",
  },
  warning: {
    border: "border-amber-500/30",
    bg: "bg-amber-500/5",
    text: "text-amber-200",
    badge: "bg-amber-500/20 text-amber-100",
    label: "WARN",
  },
  info: {
    border: "border-sky-500/30",
    bg: "bg-sky-500/5",
    text: "text-sky-200",
    badge: "bg-sky-500/20 text-sky-100",
    label: "INFO",
  },
};

function IssueRow({ issue }: { issue: ValidationIssue }) {
  const style = LEVEL_STYLES[issue.level];
  return (
    <li
      className={`rounded-lg border ${style.border} ${style.bg} px-3 py-2 font-mono text-xs ${style.text}`}
    >
      <div className="flex items-start gap-2">
        <span
          className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${style.badge}`}
        >
          {style.label}
        </span>
        <div className="flex-1 space-y-1">
          <div className="font-semibold opacity-80">{issue.field}</div>
          <div className="opacity-90">{issue.message}</div>
          {issue.suggestion ? (
            <div className="text-[11px] opacity-70">
              <span className="opacity-50">→</span> {issue.suggestion}
            </div>
          ) : null}
        </div>
      </div>
    </li>
  );
}

export function BuilderRail() {
  const draft = useConfigStore((s) => s.draft);
  const validation = useDebouncedValidate(draft);

  const issues = validation.issues;
  const hasAny = issues.length > 0;
  const errorCount = issues.filter((i) => i.level === "error").length;
  const warnCount = issues.filter((i) => i.level === "warning").length;
  const infoCount = issues.filter((i) => i.level === "info").length;

  return (
    <div className="space-y-5">
      <LinkBudgetTable source="draft" />

      <KicadExportCard />

      <Card>
        <div className="mb-3 flex items-baseline justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Validation
          </h3>
          {hasAny ? (
            <span className="text-[11px] font-mono text-slate-400">
              {errorCount > 0 ? `${errorCount}E ` : ""}
              {warnCount > 0 ? `${warnCount}W ` : ""}
              {infoCount > 0 ? `${infoCount}i` : ""}
            </span>
          ) : null}
        </div>
        {!hasAny ? (
          <motion.p
            key="valid"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: DUR.fast, ease: EASE }}
            className="text-sm text-emerald-300"
          >
            ✓ Config looks good.
          </motion.p>
        ) : (
          <motion.ul
            key="issues"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: DUR.fast, ease: EASE }}
            className="space-y-2"
          >
            {issues.map((issue, i) => (
              <IssueRow key={`${issue.rule_id || i}-${issue.field}`} issue={issue} />
            ))}
          </motion.ul>
        )}
      </Card>
    </div>
  );
}

export function useBuilderValidity() {
  const draft = useConfigStore((s) => s.draft);
  const validation = useDebouncedValidate(draft);
  return validation.valid;
}

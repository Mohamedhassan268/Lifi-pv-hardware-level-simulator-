/**
 * ValidationTab — full-screen view of the structured validator output.
 * Groups issues by rule family (the part of rule_id before the first dot)
 * and renders them in collapsible sections. Reuses the same data source as
 * BuilderRail (`useDebouncedValidate` on the draft config) so the two views
 * stay in sync.
 */

import { useMemo } from "react";

import type { ValidationIssue, ValidationIssueLevel } from "@/api/client";
import { useDebouncedValidate } from "@/hooks/useDebouncedValidate";
import { Card } from "@/primitives/Card";
import { useConfigStore } from "@/store/configStore";

const LEVEL_RANK: Record<ValidationIssueLevel, number> = {
  error: 0,
  warning: 1,
  info: 2,
};

const LEVEL_BADGE: Record<ValidationIssueLevel, string> = {
  error: "bg-rose-500/20 text-rose-200",
  warning: "bg-amber-500/20 text-amber-100",
  info: "bg-sky-500/20 text-sky-100",
};

function familyOf(rule_id?: string): string {
  if (!rule_id) return "other";
  const dot = rule_id.indexOf(".");
  return dot >= 0 ? rule_id.slice(0, dot) : rule_id;
}

export function ValidationTab() {
  // Prefer draft when the Builder is open; otherwise validate the live config.
  const draft = useConfigStore((s) => s.draft);
  const config = useConfigStore((s) => s.config);
  const target = draft ?? config;
  const validation = useDebouncedValidate(target);

  const groups = useMemo(() => {
    const map = new Map<string, ValidationIssue[]>();
    for (const issue of validation.issues) {
      const f = familyOf(issue.rule_id);
      if (!map.has(f)) map.set(f, []);
      map.get(f)!.push(issue);
    }
    for (const list of map.values()) {
      list.sort((a, b) => LEVEL_RANK[a.level] - LEVEL_RANK[b.level]);
    }
    return Array.from(map.entries()).sort(([, a], [, b]) => {
      const sevA = Math.min(...a.map((i) => LEVEL_RANK[i.level]));
      const sevB = Math.min(...b.map((i) => LEVEL_RANK[i.level]));
      return sevA - sevB;
    });
  }, [validation.issues]);

  if (validation.issues.length === 0) {
    return (
      <Card>
        <p className="text-sm text-emerald-300">
          ✓ No validation issues against the current config.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Summary
          </h3>
          <div className="flex gap-3 text-xs">
            <Stat label="errors" count={countBy(validation.issues, "error")} tone="rose" />
            <Stat label="warnings" count={countBy(validation.issues, "warning")} tone="amber" />
            <Stat label="info" count={countBy(validation.issues, "info")} tone="sky" />
          </div>
        </div>
      </Card>

      {groups.map(([family, list]) => (
        <Card key={family}>
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            {family} ({list.length})
          </h4>
          <ul className="space-y-2">
            {list.map((issue, i) => (
              <li
                key={`${issue.rule_id || i}-${issue.field}`}
                className="rounded-lg border border-slate-700/60 bg-slate-800/30 px-3 py-2"
              >
                <div className="flex items-start gap-2">
                  <span
                    className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${LEVEL_BADGE[issue.level]}`}
                  >
                    {issue.level.toUpperCase()}
                  </span>
                  <div className="flex-1 space-y-1">
                    <div className="font-mono text-xs text-slate-300">
                      {issue.field}
                      {issue.rule_id ? (
                        <span className="ml-2 text-slate-500">
                          [{issue.rule_id}]
                        </span>
                      ) : null}
                    </div>
                    <div className="text-sm text-slate-200">{issue.message}</div>
                    {issue.suggestion ? (
                      <div className="text-xs text-slate-400">
                        <span className="opacity-50">→</span> {issue.suggestion}
                      </div>
                    ) : null}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      ))}
    </div>
  );
}

function Stat({
  label,
  count,
  tone,
}: {
  label: string;
  count: number;
  tone: "rose" | "amber" | "sky";
}) {
  const cls =
    tone === "rose"
      ? "text-rose-300"
      : tone === "amber"
        ? "text-amber-200"
        : "text-sky-200";
  return (
    <span className={`font-mono ${count === 0 ? "text-slate-500" : cls}`}>
      {count} {label}
    </span>
  );
}

function countBy(
  issues: ValidationIssue[],
  level: ValidationIssueLevel,
): number {
  return issues.reduce((n, i) => n + (i.level === level ? 1 : 0), 0);
}

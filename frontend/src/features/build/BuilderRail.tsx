/**
 * BuilderRail — right-side persistent panel for the Builder. Two cards:
 *   1. Live link-budget summary (reuses LinkBudgetTable with source="draft")
 *   2. Validation error list (debounced /api/config/validate)
 */

import { motion } from "framer-motion";

import { LinkBudgetTable } from "@/features/setup/LinkBudgetTable";
import { useDebouncedValidate } from "@/hooks/useDebouncedValidate";
import { DUR, EASE } from "@/lib/motion";
import { Card } from "@/primitives/Card";
import { useConfigStore } from "@/store/configStore";

export function BuilderRail() {
  const draft = useConfigStore((s) => s.draft);
  const validation = useDebouncedValidate(draft);

  return (
    <div className="space-y-5">
      <LinkBudgetTable source="draft" />

      <Card>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">
          Validation
        </h3>
        {validation.valid ? (
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
            key="errors"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: DUR.fast, ease: EASE }}
            className="space-y-2"
          >
            {validation.errors.map((err, i) => (
              <li
                key={i}
                className="rounded-lg border border-rose-500/30 bg-rose-500/5 px-3 py-2 font-mono text-xs text-rose-300"
              >
                {err}
              </li>
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

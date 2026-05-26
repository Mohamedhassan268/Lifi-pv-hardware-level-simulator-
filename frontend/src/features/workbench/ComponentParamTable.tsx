/**
 * ComponentParamTable — right-side detail panel: shows every parameter from
 * the component's get_parameters(). Mirrors the param table in
 * gui/tab_component_library.py.
 */

import { motion } from "framer-motion";
import { useEffect, useState } from "react";

import { api, type ComponentDetail } from "@/api/client";
import { DUR, EASE } from "@/lib/motion";
import { Button } from "@/primitives/Button";
import { Card } from "@/primitives/Card";
import { useConfigStore } from "@/store/configStore";

interface Props {
  part: string | null;
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") {
    const abs = Math.abs(v);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return v.toExponential(3);
    return Number.isInteger(v) ? v.toString() : v.toFixed(4);
  }
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

export function ComponentParamTable({ part }: Props) {
  const [detail, setDetail] = useState<ComponentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const patch = useConfigStore((s) => s.patch);
  const currentSlotValue = useConfigStore((s) =>
    detail?.config_field ? (s.config[detail.config_field] as string | undefined) : undefined,
  );

  useEffect(() => {
    if (!part) {
      setDetail(null);
      setError(null);
      return;
    }
    let cancelled = false;
    api
      .getComponent(part)
      .then((d) => {
        if (!cancelled) {
          setDetail(d);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [part]);

  if (!part) {
    return (
      <Card>
        <p className="text-sm text-slate-400">
          Select a part on the left to see its datasheet parameters.
        </p>
      </Card>
    );
  }

  const isCurrent =
    detail?.config_field && currentSlotValue === detail.part;
  const canApply = detail?.config_field && !isCurrent;

  return (
    <Card>
      <header className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h3 className="font-mono text-lg text-beam-300">{part}</h3>
          <p className="text-xs uppercase tracking-wider text-slate-500">
            {detail?.category} {detail?.class && `· ${detail.class}`}
          </p>
        </div>
        {detail?.config_field && (
          <Button
            disabled={!canApply}
            onClick={() => detail && patch({ [detail.config_field]: detail.part })}
          >
            {isCurrent ? "In use" : `Use as ${detail.config_field}`}
          </Button>
        )}
      </header>

      {error && <p className="mb-3 text-sm text-rose-400">{error}</p>}

      {detail && (
        <motion.dl
          key={detail.part}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: DUR.fast, ease: EASE }}
          className="grid grid-cols-1 gap-2 sm:grid-cols-2"
        >
          {Object.entries(detail.parameters).map(([k, v]) => (
            <div
              key={k}
              className="flex flex-col rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2"
            >
              <dt className="text-[10px] uppercase tracking-wider text-slate-500">
                {k}
              </dt>
              <dd className="font-mono text-sm text-slate-200">{formatValue(v)}</dd>
            </div>
          ))}
        </motion.dl>
      )}
    </Card>
  );
}

/**
 * LinkBudgetTable — calls POST /api/link-budget whenever the relevant config
 * fields change, and displays the result.
 *
 * Uses a short debounce so dragging a slider doesn't fire 60 requests/sec.
 */

import { motion } from "framer-motion";
import { useEffect, useState } from "react";

import type { LinkBudgetResponse } from "@/api/client";
import { api } from "@/api/client";
import { DUR, EASE } from "@/lib/motion";
import { Card } from "@/primitives/Card";
import { useLinkBudgetInput, type ConfigSource } from "@/store/configStore";

function useDebounced<T>(value: T, delayMs: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return v;
}

const ROWS: Array<{ key: keyof LinkBudgetResponse; label: string; unit: string; digits: number }> = [
  { key: "channel_gain", label: "Channel gain H(0)", unit: "", digits: 5 },
  { key: "p_rx_uW", label: "Received power", unit: "µW", digits: 2 },
  { key: "i_ph_uA", label: "Photocurrent", unit: "µA", digits: 2 },
  { key: "lambertian_order", label: "Lambertian order m", unit: "", digits: 2 },
  { key: "snr_estimate_dB", label: "SNR estimate", unit: "dB", digits: 1 },
];

interface LinkBudgetTableProps {
  source?: ConfigSource;
}

export function LinkBudgetTable({ source = "live" }: LinkBudgetTableProps = {}) {
  const input = useLinkBudgetInput(source);
  const debounced = useDebounced(input, 80);
  const [result, setResult] = useState<LinkBudgetResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .linkBudget(debounced)
      .then((r) => {
        if (!cancelled) {
          setResult(r);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [debounced]);

  return (
    <Card>
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-300">
        Link Budget
      </h3>
      {error && <p className="text-sm text-rose-400">{error}</p>}
      <dl className="grid grid-cols-1 gap-3">
        {ROWS.map((row) => (
          <div key={row.key} className="flex items-baseline justify-between">
            <dt className="text-sm text-slate-400">{row.label}</dt>
            <motion.dd
              key={result ? `${row.key}-${result[row.key]}` : `${row.key}-loading`}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: DUR.fast, ease: EASE }}
              className="font-mono text-sm text-beam-300"
            >
              {result
                ? `${result[row.key].toFixed(row.digits)}${row.unit ? " " + row.unit : ""}`
                : "—"}
            </motion.dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}

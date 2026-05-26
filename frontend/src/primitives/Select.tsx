/**
 * Select — labelled dropdown. Tailwind-themed native <select>.
 */

import { type ChangeEvent } from "react";

interface SelectProps {
  label: string;
  value: string | undefined;
  onChange: (v: string) => void;
  options: Array<string | { value: string; label: string }>;
  disabled?: boolean;
  hint?: string;
}

export function Select({ label, value, onChange, options, disabled, hint }: SelectProps) {
  return (
    <label className={"block " + (disabled ? "opacity-50" : "")}>
      <span className="mb-1 block text-xs uppercase tracking-wider text-slate-400">
        {label}
      </span>
      <select
        value={value ?? ""}
        disabled={disabled}
        onChange={(e: ChangeEvent<HTMLSelectElement>) => onChange(e.target.value)}
        className="w-full rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2
                   text-sm text-slate-100
                   focus:outline-none focus:ring-2 focus:ring-beam-400/40
                   disabled:cursor-not-allowed"
      >
        {options.map((opt) => {
          const v = typeof opt === "string" ? opt : opt.value;
          const l = typeof opt === "string" ? opt : opt.label;
          return (
            <option key={v} value={v} className="bg-slate-900">
              {l}
            </option>
          );
        })}
      </select>
      {hint && <p className="mt-1 text-[10px] text-slate-500">{hint}</p>}
    </label>
  );
}

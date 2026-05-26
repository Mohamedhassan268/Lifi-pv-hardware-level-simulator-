/**
 * NumberField — instrument-style number input. Square corners, hairline
 * border, monospace + tabular-nums value, uppercase tracked label, unit shown
 * to the right of the input (column-aligned across stacked fields).
 */

import { type ChangeEvent } from "react";

interface NumberFieldProps {
  label: string;
  unit?: string;
  value: number | undefined;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  hint?: string;
}

export function NumberField({
  label,
  unit,
  value,
  onChange,
  step,
  min,
  max,
  disabled,
  hint,
}: NumberFieldProps) {
  return (
    <label className={"block " + (disabled ? "opacity-50" : "")}>
      <div className="readout mb-0.5 text-[10px] text-slate-500">
        <span className="select-all">{label}</span>
      </div>
      <div className="flex items-stretch border border-hair bg-slate-950 focus-within:border-beam-400/50">
        <input
          type="number"
          value={value ?? ""}
          step={step}
          min={min}
          max={max}
          disabled={disabled}
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v)) onChange(v);
          }}
          className="readout min-w-0 flex-1 bg-transparent px-2 py-1 text-[12px] text-slate-100
                     outline-none disabled:cursor-not-allowed"
        />
        {unit && (
          <span className="readout flex select-none items-center border-l border-hair bg-white/[0.02] px-2 text-[10px] text-slate-400">
            {unit}
          </span>
        )}
      </div>
      {hint && <p className="mt-0.5 text-[10px] text-slate-500">{hint}</p>}
    </label>
  );
}

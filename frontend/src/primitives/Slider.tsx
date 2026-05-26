/**
 * Slider — native range input with Tailwind theming. Native ranges deliver
 * the smoothest drag perf on every platform; we avoid Framer Motion on the
 * thumb so 60 Hz drag events don't fight the animation system.
 */

import { type ChangeEvent } from "react";

interface SliderProps {
  label: string;
  unit?: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}

export function Slider({
  label,
  unit,
  min,
  max,
  step = (max - min) / 100,
  value,
  onChange,
  format,
}: SliderProps) {
  const display = format ? format(value) : value.toFixed(step < 1 ? 2 : 0);
  return (
    <label className="block">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-wider text-slate-400">
          {label}
        </span>
        <span className="font-mono text-sm text-beam-300">
          {display}
          {unit ? ` ${unit}` : ""}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e: ChangeEvent<HTMLInputElement>) =>
          onChange(parseFloat(e.target.value))
        }
        className="
          h-1 w-full cursor-pointer appearance-none rounded-full bg-white/10
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-beam-400
          [&::-webkit-slider-thumb]:shadow-[0_0_0_4px_rgba(34,211,238,0.18)]
          [&::-webkit-slider-thumb]:transition-shadow
          [&::-webkit-slider-thumb]:hover:shadow-[0_0_0_6px_rgba(34,211,238,0.28)]
          [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4
          [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-beam-400
          [&::-moz-range-thumb]:border-0
        "
      />
    </label>
  );
}

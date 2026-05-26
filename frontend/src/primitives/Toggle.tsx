/**
 * Toggle — instrument-style enable switch. Square corners, no spring;
 * the knob is a small block that snaps between OFF and ON positions.
 */

interface ToggleProps {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  hint?: string;
}

export function Toggle({ label, value, onChange, disabled, hint }: ToggleProps) {
  return (
    <label
      className={
        "flex items-center justify-between gap-2 border border-hair bg-slate-950 px-2 py-1.5 " +
        (disabled ? "opacity-50" : "cursor-pointer hover:bg-white/[0.02]")
      }
    >
      <span className="flex min-w-0 flex-col">
        <span className="readout truncate text-[11px] text-slate-200">{label}</span>
        {hint && (
          <span className="truncate text-[10px] text-slate-500">{hint}</span>
        )}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={value}
        disabled={disabled}
        onClick={() => !disabled && onChange(!value)}
        className={
          "relative h-3.5 w-7 shrink-0 border transition-colors " +
          (value
            ? "border-beam-400/60 bg-beam-400/30"
            : "border-white/15 bg-white/[0.05]")
        }
      >
        <span
          className={
            "absolute top-0.5 h-2 w-2 transition-transform " +
            (value ? "bg-beam-300" : "bg-white/60")
          }
          style={{ transform: `translateX(${value ? 16 : 2}px)` }}
        />
      </button>
    </label>
  );
}

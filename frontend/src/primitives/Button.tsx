/**
 * Button — instrument-style action. Square corners, hairline border, no
 * glow shadow, no spring. Two variants:
 *   - primary : filled cyan
 *   - ghost   : hairline-bordered translucent
 */

import { type ButtonHTMLAttributes } from "react";

type Variant = "primary" | "ghost";

export function Button({
  children,
  variant = "primary",
  className = "",
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  const base =
    "inline-flex items-center justify-center gap-1.5 px-2.5 py-1 text-[11px] " +
    "uppercase tracking-wider font-medium select-none transition-colors";
  const styles =
    variant === "primary"
      ? "bg-beam-400 text-slate-950 hover:bg-beam-300"
      : "border border-hair bg-white/[0.03] text-slate-200 hover:bg-white/[0.07]";
  const disabledStyles = disabled ? " opacity-40 cursor-not-allowed" : "";

  return (
    <button
      disabled={disabled}
      className={`${base} ${styles}${disabledStyles} ${className}`}
      {...(rest as object)}
    >
      {children}
    </button>
  );
}

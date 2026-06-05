/**
 * Schematic symbols — one SVG glyph per component kind, drawn in a fixed
 * 64×40 viewBox so PartNode can lay them out uniformly. Pin handles are
 * positioned by PartNode (by port role), so these glyphs are purely the
 * recognisable body of each part.
 *
 * `symbolFor(category, ctype)` maps a component to its glyph key; PartNode
 * stores the key on node data so the canvas stays declarative.
 */

import type { ReactNode } from "react";

export type SymbolKey =
  | "amp"
  | "comparator"
  | "resistor"
  | "capacitor"
  | "source"
  | "drive"
  | "channel"
  | "diode"
  | "transistor"
  | "led"
  | "photodiode"
  | "mosfet"
  | "solar_cell"
  | "output"
  | "ground"
  | "vcc"
  | "vee"
  | "vref"
  | "mcu"
  | "multimeter"
  | "scope"
  | "generic";

const STROKE = "#cbd5e1"; // slate-300
const ACCENT = "#67e8f9"; // beam

const W = 64;
const H = 40;

function frame(children: ReactNode) {
  // Scalable: fills its container width (PartNode sizes it per component), so
  // small parts (R/C, power flags) render compact and active parts render large.
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" preserveAspectRatio="xMidYMid meet">
      {children}
    </svg>
  );
}

const SYMBOLS: Record<SymbolKey, ReactNode> = {
  // Op-amp / instrumentation amp: triangle pointing right, +/- on inputs.
  amp: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <polygon points="18,6 18,34 48,20" fill="rgba(103,232,249,0.06)" />
      <line x1="6" y1="13" x2="18" y2="13" />
      <line x1="6" y1="27" x2="18" y2="27" />
      <line x1="48" y1="20" x2="58" y2="20" />
      {/* supply-rail stubs (VCC up, VEE down) */}
      <line x1="30" y1="11" x2="30" y2="3" />
      <line x1="30" y1="29" x2="30" y2="37" />
      <text x="21" y="16" fontSize="7" fill={STROKE} stroke="none">+</text>
      <text x="22" y="31" fontSize="8" fill={STROKE} stroke="none">−</text>
    </g>,
  ),

  // Comparator: triangle with a hysteresis glyph inside.
  comparator: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <polygon points="18,6 18,34 48,20" fill="rgba(192,132,252,0.07)" />
      <line x1="6" y1="13" x2="18" y2="13" />
      <line x1="6" y1="27" x2="18" y2="27" />
      <line x1="48" y1="20" x2="58" y2="20" />
      {/* supply-rail stubs (VCC up, VEE down) */}
      <line x1="30" y1="11" x2="30" y2="3" />
      <line x1="30" y1="29" x2="30" y2="37" />
      <path d="M26,22 h4 v-4 h4" stroke="#c084fc" strokeWidth={1.2} />
    </g>,
  ),

  // Resistor: zig-zag.
  resistor: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="20" x2="14" y2="20" />
      <polyline points="14,20 18,12 24,28 30,12 36,28 42,12 46,20" />
      <line x1="46" y1="20" x2="60" y2="20" />
    </g>,
  ),

  // Capacitor: two parallel plates.
  capacitor: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="20" x2="28" y2="20" />
      <line x1="28" y1="8" x2="28" y2="32" />
      <line x1="36" y1="8" x2="36" y2="32" />
      <line x1="36" y1="20" x2="60" y2="20" />
    </g>,
  ),

  // Data source / modulator: a clock-like digital waveform in a box.
  drive: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <rect x="10" y="8" width="40" height="24" rx="3" fill="rgba(103,232,249,0.06)" />
      <polyline points="14,26 14,14 22,14 22,26 30,26 30,14 38,14 38,26 46,26"
        stroke={ACCENT} strokeWidth={1.2} />
      <line x1="50" y1="20" x2="60" y2="20" />
    </g>,
  ),

  // Optical channel: a light cone from the emitter (cyan) widening to the
  // receiver (pink) — the one beam both the transmitted and received light cross.
  channel: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="2" y1="20" x2="12" y2="20" />
      <line x1="52" y1="20" x2="62" y2="20" />
      <polygon points="12,20 50,7 50,33" fill="rgba(103,232,249,0.15)" stroke="none" />
      <g stroke={ACCENT} strokeWidth={1}>
        <line x1="12" y1="20" x2="50" y2="11" />
        <line x1="12" y1="20" x2="50" y2="20" />
        <line x1="12" y1="20" x2="50" y2="29" />
      </g>
      <circle cx="12" cy="20" r="3" fill={ACCENT} stroke="none" />
      <rect x="48" y="12" width="4" height="16" rx="1" fill="#f472b6" stroke="none" />
    </g>,
  ),

  // DC source: circle with + / −.
  source: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="20" x2="18" y2="20" />
      <circle cx="32" cy="20" r="13" fill="rgba(251,191,36,0.06)" />
      <line x1="46" y1="20" x2="60" y2="20" />
      <text x="28" y="17" fontSize="8" fill={STROKE} stroke="none">+</text>
      <text x="29" y="30" fontSize="9" fill={STROKE} stroke="none">−</text>
    </g>,
  ),

  // Diode: triangle + bar.
  diode: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="20" x2="24" y2="20" />
      <polygon points="24,12 24,28 38,20" fill="rgba(203,213,225,0.12)" />
      <line x1="38" y1="12" x2="38" y2="28" />
      <line x1="38" y1="20" x2="60" y2="20" />
    </g>,
  ),

  // NPN BJT: base bar, collector/emitter legs, emitter arrow pointing out.
  transistor: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="6" y1="20" x2="24" y2="20" />
      <line x1="24" y1="10" x2="24" y2="30" />
      <line x1="24" y1="16" x2="44" y2="7" />
      <line x1="44" y1="7" x2="44" y2="2" />
      <line x1="24" y1="24" x2="44" y2="33" />
      <line x1="44" y1="33" x2="44" y2="38" />
      <polygon points="44,33 38,30 39,34" fill={STROKE} stroke="none" />
    </g>,
  ),

  // LED: diode + two emission arrows.
  led: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="22" x2="22" y2="22" />
      <polygon points="22,14 22,30 36,22" fill="rgba(251,191,36,0.12)" />
      <line x1="36" y1="14" x2="36" y2="30" />
      <line x1="36" y1="22" x2="56" y2="22" />
      <g stroke={ACCENT} strokeWidth={1.2}>
        <line x1="40" y1="10" x2="46" y2="4" />
        <polyline points="46,4 43,5 45,8" fill="none" />
        <line x1="46" y1="12" x2="52" y2="6" />
        <polyline points="52,6 49,7 51,10" fill="none" />
      </g>
    </g>,
  ),

  // Photodiode: diode + two incoming arrows.
  photodiode: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="22" x2="22" y2="22" />
      <polygon points="22,14 22,30 36,22" fill="rgba(244,114,182,0.12)" />
      <line x1="36" y1="14" x2="36" y2="30" />
      <line x1="36" y1="22" x2="56" y2="22" />
      <g stroke="#f472b6" strokeWidth={1.2}>
        <line x1="50" y1="4" x2="44" y2="10" />
        <polyline points="44,10 47,9 45,6" fill="none" />
        <line x1="56" y1="6" x2="50" y2="12" />
        <polyline points="50,12 53,11 51,8" fill="none" />
      </g>
    </g>,
  ),

  // N-MOSFET: gate bar + channel with drain/source/arrow.
  mosfet: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="20" x2="18" y2="20" />
      <line x1="18" y1="10" x2="18" y2="30" />
      <line x1="22" y1="10" x2="22" y2="30" />
      <line x1="22" y1="12" x2="40" y2="12" />
      <line x1="40" y1="6" x2="40" y2="34" />
      <line x1="22" y1="28" x2="40" y2="28" />
      <line x1="40" y1="12" x2="52" y2="12" />
      <line x1="40" y1="28" x2="52" y2="28" />
      <polyline points="30,24 34,28 30,32" stroke={ACCENT} />
    </g>,
  ),

  // Solar cell: photodiode-style cell with sun rays.
  solar_cell: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="24" x2="20" y2="24" />
      <line x1="20" y1="14" x2="20" y2="34" />
      <line x1="28" y1="10" x2="28" y2="38" strokeWidth={2.4} />
      <line x1="28" y1="24" x2="44" y2="24" />
      <rect x="44" y="14" width="14" height="20" fill="rgba(251,191,36,0.10)" />
      <g stroke="#fbbf24" strokeWidth={1.1}>
        <line x1="48" y1="10" x2="46" y2="5" />
        <line x1="54" y1="10" x2="55" y2="5" />
      </g>
    </g>,
  ),

  // Output terminal: an arrow into a labelled flag (the V(dout) probe point).
  output: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="20" x2="22" y2="20" />
      <polygon points="22,12 46,12 54,20 46,28 22,28" fill="rgba(52,211,153,0.10)" stroke="#34d399" />
      <text x="28" y="24" fontSize="8" fill="#34d399" stroke="none">OUT</text>
    </g>,
  ),

  // Multimeter probe: lead into a round meter showing 'V' (measurement-only).
  multimeter: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="20" x2="22" y2="20" />
      <circle cx="40" cy="20" r="13" fill="rgba(103,232,249,0.08)" stroke={ACCENT} />
      <text x="35" y="24" fontSize="11" fill={ACCENT} stroke="none">V</text>
    </g>,
  ),

  // Scope probe: lead into a screen with a small trace (measurement-only).
  scope: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="4" y1="20" x2="18" y2="20" />
      <rect x="18" y="7" width="42" height="26" rx="2" fill="rgba(103,232,249,0.06)" stroke={ACCENT} />
      <polyline points="22,20 27,12 31,28 35,12 39,28 43,12 47,28 51,20 58,20"
                stroke="#34d399" strokeWidth={1.2} fill="none" />
    </g>,
  ),

  // Ground: standard 3-bar symbol.
  ground: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <line x1="32" y1="6" x2="32" y2="20" />
      <line x1="20" y1="20" x2="44" y2="20" />
      <line x1="25" y1="26" x2="39" y2="26" />
      <line x1="29" y1="32" x2="35" y2="32" />
    </g>,
  ),

  // VCC power flag: bar + stem down to the (bottom) pin, labelled.
  vcc: frame(
    <g fill="none" stroke="#fbbf24" strokeWidth={1.5}>
      <line x1="20" y1="13" x2="44" y2="13" />
      <line x1="32" y1="13" x2="32" y2="34" />
      <text x="22" y="10" fontSize="8" fill="#fbbf24" stroke="none">VCC</text>
    </g>,
  ),

  // VEE / negative rail flag: stem up to the (top) pin, bar below, labelled.
  vee: frame(
    <g fill="none" stroke="#94a3b8" strokeWidth={1.5}>
      <line x1="32" y1="6" x2="32" y2="27" />
      <line x1="20" y1="27" x2="44" y2="27" />
      <text x="22" y="37" fontSize="8" fill="#94a3b8" stroke="none">VEE</text>
    </g>,
  ),

  // VREF mid-rail flag: bar + stem down to the (bottom) pin, labelled.
  vref: frame(
    <g fill="none" stroke="#c084fc" strokeWidth={1.5}>
      <line x1="20" y1="13" x2="44" y2="13" />
      <line x1="32" y1="13" x2="32" y2="34" />
      <text x="20" y="10" fontSize="8" fill="#c084fc" stroke="none">VREF</text>
    </g>,
  ),

  // Microcontroller (ESP32 / Arduino): a chip with leads and an MCU label.
  mcu: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <rect x="16" y="7" width="32" height="26" rx="2" fill="rgba(103,232,249,0.07)" />
      <line x1="6" y1="14" x2="16" y2="14" stroke={ACCENT} />
      <line x1="6" y1="26" x2="16" y2="26" />
      <line x1="48" y1="14" x2="58" y2="14" stroke="#34d399" />
      <line x1="48" y1="26" x2="58" y2="26" />
      <text x="20" y="23" fontSize="7" fill={STROKE} stroke="none">MCU</text>
    </g>,
  ),

  generic: frame(
    <g fill="none" stroke={STROKE} strokeWidth={1.5}>
      <rect x="14" y="10" width="36" height="20" rx="3" fill="rgba(203,213,225,0.05)" />
      <line x1="4" y1="20" x2="14" y2="20" />
      <line x1="50" y1="20" x2="60" y2="20" />
    </g>,
  ),
};

export function symbolFor(category: string, ctype: string): SymbolKey {
  if (ctype === "R") return "resistor";
  if (ctype === "C") return "capacitor";
  if (ctype === "V") return "source";
  if (ctype === "DRIVE") return "drive";
  if (ctype === "CHANNEL") return "channel";
  if (ctype === "OUT") return "output";
  if (ctype === "GND") return "ground";
  if (ctype === "VCC") return "vcc";
  if (ctype === "VEE") return "vee";
  if (ctype === "VREF") return "vref";
  if (ctype === "MCU") return "mcu";
  switch (category) {
    case "Amplifier":
      return "amp";
    case "Comparator":
      return "comparator";
    case "BJT":
      return "transistor";
    case "Zener":
      return "diode";
    case "Photodiode":
      return "photodiode";
    case "Solar Cell":
      return "solar_cell";
    case "LED":
      return "led";
    case "MOSFET":
      return "mosfet";
    default:
      return "generic";
  }
}

export function Symbol({ symbol }: { symbol: SymbolKey }) {
  return <>{SYMBOLS[symbol] ?? SYMBOLS.generic}</>;
}

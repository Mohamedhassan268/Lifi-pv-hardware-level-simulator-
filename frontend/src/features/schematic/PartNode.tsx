/**
 * PartNode — a placed component on the schematic canvas. Each part renders its
 * SPICE ports as React Flow Handles anchored to the symbol's drawn terminals
 * (Proteus-style): the pin dots sit on the actual stubs of the glyph, with a
 * per-symbol anchor map keyed by port role. Roles with no explicit anchor fall
 * back to an evenly-distributed edge slot.
 *
 * Each Handle id IS the port name, so schematicStore can resolve nets from
 * React Flow connections without extra bookkeeping.
 */

import { Handle, Position, type NodeProps } from "@xyflow/react";

import type { SpicePort } from "@/api/client";
import { Symbol, type SymbolKey } from "@/features/schematic/symbols";
import type { PartNodeData } from "@/store/schematicStore";

const ROLE_COLOR: Record<string, string> = {
  signal_in: "#67e8f9",
  signal_out: "#34d399",
  supply_pos: "#fbbf24",
  supply_neg: "#94a3b8",
  ref: "#c084fc",
  optical_in: "#f472b6",
  optical_out: "#f472b6",
  signal: "#67e8f9",
  output: "#34d399",
  ground: "#94a3b8",
};

// Fallback side for a role when a symbol declares no explicit anchor for it.
const ROLE_SIDE: Record<string, Position> = {
  signal_in: Position.Left,
  optical_in: Position.Left,
  ref: Position.Left,
  signal: Position.Left,
  output: Position.Left,
  ground: Position.Top,
  signal_out: Position.Right,
  optical_out: Position.Right,
  supply_pos: Position.Top,
  supply_neg: Position.Bottom,
};

// Power-rail flags point their single pin toward the components they sit above
// (VCC/VREF) or below (VEE/GND), overriding the role-based fallback side.
const MARKER_SIDE: Record<string, Position> = {
  VCC: Position.Bottom,
  VREF: Position.Bottom,
  VEE: Position.Top,
  GND: Position.Top,
};

// ---------------------------------------------------------------------------
// Per-symbol pin anchors. Coordinates are in each glyph's 64×40 viewBox (see
// symbols.tsx); `side` is the direction the pin stub points (drives edge
// routing + dot offset). Anchors are listed per role in pin order, so the i-th
// port of a given role takes the i-th anchor (e.g. INP then INN for the two
// signal_in pins of an op-amp). Symbols/roles not listed use the edge fallback.
// ---------------------------------------------------------------------------
type Size = "xs" | "sm" | "md" | "lg" | "xl";
interface Anchor {
  x: number;
  y: number;
  side: Position;
}
type SymbolPins = Partial<Record<string, Anchor[]>>;

const L = Position.Left;
const R = Position.Right;
const T = Position.Top;
const B = Position.Bottom;

const SYMBOL_PINS: Partial<Record<SymbolKey, SymbolPins>> = {
  // Op-amp / instrumentation amp triangle: two inputs left, output right,
  // rails on the stubs added top/bottom; REF (INA322 only) bottom-right.
  amp: {
    signal_in: [{ x: 6, y: 13, side: L }, { x: 6, y: 27, side: L }],
    signal_out: [{ x: 58, y: 20, side: R }],
    supply_pos: [{ x: 30, y: 3, side: T }],
    supply_neg: [{ x: 30, y: 37, side: B }],
    ref: [{ x: 44, y: 37, side: B }],
  },
  comparator: {
    signal_in: [{ x: 6, y: 13, side: L }, { x: 6, y: 27, side: L }],
    signal_out: [{ x: 58, y: 20, side: R }],
    supply_pos: [{ x: 30, y: 3, side: T }],
    supply_neg: [{ x: 30, y: 37, side: B }],
  },
  resistor: {
    signal: [{ x: 4, y: 20, side: L }],
    signal_out: [{ x: 60, y: 20, side: R }],
  },
  capacitor: {
    signal: [{ x: 4, y: 20, side: L }],
    signal_out: [{ x: 60, y: 20, side: R }],
  },
  source: {
    signal: [{ x: 4, y: 20, side: L }],
    signal_out: [{ x: 60, y: 20, side: R }],
  },
  drive: {
    signal_out: [{ x: 60, y: 20, side: R }],
  },
  channel: {
    optical_in: [{ x: 2, y: 20, side: L }],
    optical_out: [{ x: 62, y: 20, side: R }],
  },
  // Diode-based detectors: anode (signal_out) on the triangle lead (left),
  // cathode (signal) on the bar lead (right), optical input from the top.
  photodiode: {
    signal_out: [{ x: 4, y: 22, side: L }],
    signal: [{ x: 56, y: 22, side: R }],
    optical_in: [{ x: 50, y: 4, side: T }],
  },
  solar_cell: {
    signal_out: [{ x: 4, y: 24, side: L }],
    signal: [{ x: 58, y: 24, side: R }],
    optical_in: [{ x: 50, y: 5, side: T }],
  },
  led: {
    signal_in: [{ x: 4, y: 22, side: L }],
    signal: [{ x: 56, y: 22, side: R }],
  },
  // N-MOSFET: gate left, drain right-top, source right-bottom.
  mosfet: {
    signal_in: [{ x: 4, y: 20, side: L }],
    signal_out: [{ x: 52, y: 12, side: R }],
    signal: [{ x: 52, y: 28, side: R }],
  },
  // NPN BJT: base left, collector top, emitter bottom (ports: collector, base,
  // emitter -> roles signal, signal_in, signal, so signal[0]=collector top and
  // signal[1]=emitter bottom).
  transistor: {
    signal: [{ x: 44, y: 3, side: T }, { x: 44, y: 37, side: B }],
    signal_in: [{ x: 6, y: 20, side: L }],
  },
  // Diode / Zener: anode lead left, cathode (bar) lead right.
  diode: {
    signal: [{ x: 4, y: 20, side: L }, { x: 60, y: 20, side: R }],
  },
  mcu: {
    signal_in: [{ x: 6, y: 14, side: L }],
    signal_out: [{ x: 58, y: 14, side: R }],
  },
  output: {
    output: [{ x: 4, y: 20, side: L }],
  },
  ground: {
    ground: [{ x: 32, y: 6, side: T }],
  },
  vcc: {
    supply_pos: [{ x: 32, y: 34, side: B }],
  },
  vee: {
    supply_neg: [{ x: 32, y: 6, side: T }],
  },
  vref: {
    ref: [{ x: 32, y: 34, side: B }],
  },
};

const SIZE_BY_SYMBOL: Partial<Record<SymbolKey, Size>> = {
  resistor: "xs", capacitor: "xs",
  ground: "xs", vcc: "xs", vee: "xs", vref: "xs", output: "xs",
  source: "sm", diode: "sm", led: "sm", photodiode: "sm", mosfet: "sm",
  amp: "md", comparator: "md", drive: "md", generic: "md",
  channel: "lg", solar_cell: "lg",
  mcu: "xl",
};

function sizeOf(symbol: SymbolKey): Size {
  return SIZE_BY_SYMBOL[symbol] ?? "md";
}

const SYM_W: Record<Size, number> = { xs: 34, sm: 46, md: 58, lg: 74, xl: 94 };

// An anchor on a glyph edge, for ports a symbol doesn't explicitly place.
function edgeAnchor(side: Position, frac: number): Anchor {
  switch (side) {
    case Position.Left:
      return { x: 1, y: frac * 40, side };
    case Position.Right:
      return { x: 63, y: frac * 40, side };
    case Position.Top:
      return { x: frac * 64, y: 1, side };
    default:
      return { x: frac * 64, y: 39, side };
  }
}

// Resolve every port to a glyph-space anchor: explicit per-symbol anchors
// first, then evenly-distributed edge slots for any leftover roles.
function resolveAnchors(d: PartNodeData): { port: SpicePort; anchor: Anchor }[] {
  const symbol = (d.symbol as SymbolKey) ?? "generic";
  const pinMap = SYMBOL_PINS[symbol] ?? {};
  const roleCount: Record<string, number> = {};

  const out = d.ports.map((p) => {
    const idx = roleCount[p.role] ?? 0;
    roleCount[p.role] = idx + 1;
    const anchor = pinMap[p.role]?.[idx] ?? null;
    return { port: p, anchor };
  });

  // Fallback: spread unplaced ports along their role's edge.
  const fallbackSide = (role: string) =>
    MARKER_SIDE[d.ctype] ?? ROLE_SIDE[role] ?? Position.Left;
  const bySide = new Map<Position, number[]>();
  out.forEach((o, i) => {
    if (o.anchor) return;
    const s = fallbackSide(o.port.role);
    bySide.set(s, [...(bySide.get(s) ?? []), i]);
  });
  for (const [side, idxs] of bySide) {
    idxs.forEach((i, k) => {
      out[i].anchor = edgeAnchor(side, (k + 1) / (idxs.length + 1));
    });
  }

  return out as { port: SpicePort; anchor: Anchor }[];
}

export function PartNode({ data, selected }: NodeProps) {
  const d = data as PartNodeData;
  const size = sizeOf((d.symbol as SymbolKey) ?? "generic");
  const w = SYM_W[size];
  const h = (w * 40) / 64; // preserve the 64×40 glyph aspect

  const anchors = resolveAnchors(d);
  const handleType = (role: string) =>
    role === "signal_out" || role === "optical_out" ? "source" : "target";

  return (
    // No card: the bare symbol sits directly on the canvas grid (Proteus-style).
    // Selection is shown by a glow tracing the glyph, not a bordered rectangle.
    <div className="flex flex-col items-center">
      {/* Symbol box: pins are positioned in this glyph-space (64×40) region so
          the dots land on the drawn terminals. */}
      <div
        className={"relative" + (selected ? " drop-shadow-[0_0_7px_rgba(34,211,238,0.9)]" : "")}
        style={{ width: w, height: h }}
      >
        <Symbol symbol={(d.symbol as SymbolKey) ?? "generic"} />
        {anchors.map(({ port, anchor }) => (
          <Handle
            key={port.name}
            id={port.name}
            type={handleType(port.role)}
            position={anchor.side}
            style={{
              left: `${(anchor.x / 64) * 100}%`,
              top: `${(anchor.y / 40) * 100}%`,
              transform: "translate(-50%, -50%)",
              width: 9,
              height: 9,
              border: 0,
              background: ROLE_COLOR[port.role] ?? "#67e8f9",
            }}
            isConnectable
          />
        ))}
      </div>

      <div className="mt-1 text-center">
        <div
          className={
            (size === "xs" || size === "sm" ? "text-[10px] " : "text-xs ") +
            (selected ? "text-beam-200" : "text-slate-200")
          }
        >
          {d.label}
        </div>
        <div className="mt-0.5 font-mono text-[9px] tracking-wider text-slate-500">
          {d.value ? d.value : (d.partCode ?? d.ctype)}
        </div>
      </div>
    </div>
  );
}

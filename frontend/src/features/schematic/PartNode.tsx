/**
 * PartNode — a generic placed component on the schematic canvas. Renders one
 * Handle per SPICE port, positioned by role: signal inputs on the left,
 * outputs on the right, supplies on top/bottom, ref/optical on the left.
 *
 * Each Handle id IS the port name, so schematicStore can resolve nets from
 * React Flow connections without extra bookkeeping.
 */

import { Handle, Position, type NodeProps } from "@xyflow/react";

import { Symbol, type SymbolKey } from "@/features/schematic/symbols";
import type { PartNodeData } from "@/store/schematicStore";

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

// Power-rail flags point their single pin toward the components they sit above
// (VCC/VREF) or below (VEE/GND), overriding the role-based side.
const MARKER_SIDE: Record<string, Position> = {
  VCC: Position.Bottom,
  VREF: Position.Bottom,
  VEE: Position.Top,
  GND: Position.Top,
};

// Node footprint scales with the part: passives and power flags stay compact,
// active parts (amps, MOSFETs, the MCU, channel) render larger.
const COMPACT = new Set(["R", "C", "V", "VCC", "VEE", "VREF", "GND", "OUT"]);
const MEDIUM = new Set(["DRIVE"]);

function sizeOf(ctype: string): "sm" | "md" | "lg" {
  if (COMPACT.has(ctype)) return "sm";
  if (MEDIUM.has(ctype)) return "md";
  return "lg";
}

const BOX_MIN = { sm: "min-w-[64px]", md: "min-w-[104px]", lg: "min-w-[128px]" };
const SYM_W = { sm: 40, md: 56, lg: 72 };
const PAD = { sm: "px-2 py-1.5", md: "px-2.5 py-2", lg: "px-3 py-2" };

export function PartNode({ data, selected }: NodeProps) {
  const d = data as PartNodeData;
  const size = sizeOf(d.ctype);
  const sideOf = (role: string) =>
    MARKER_SIDE[d.ctype] ?? ROLE_SIDE[role] ?? Position.Left;
  const bySide = (side: Position) => d.ports.filter((p) => sideOf(p.role) === side);

  const left = bySide(Position.Left);
  const right = bySide(Position.Right);
  const top = bySide(Position.Top);
  const bottom = bySide(Position.Bottom);

  // Whether each port acts as a connection source or target. React Flow needs
  // a type; we expose every pin as both by rendering source+target stacked is
  // overkill — outputs are sources, everything else is a target-capable source.
  const handleType = (role: string) =>
    role === "signal_out" || role === "optical_out" ? "source" : "target";

  const renderHandles = (ports: typeof left, side: Position) =>
    ports.map((p, i) => {
      const pct = ((i + 1) / (ports.length + 1)) * 100;
      const isVert = side === Position.Top || side === Position.Bottom;
      const style = isVert
        ? { left: `${pct}%`, background: ROLE_COLOR[p.role] ?? "#67e8f9" }
        : { top: `${pct}%`, background: ROLE_COLOR[p.role] ?? "#67e8f9" };
      return (
        <Handle
          key={`${side}-${p.name}`}
          id={p.name}
          type={handleType(p.role)}
          position={side}
          style={{ ...style, width: 9, height: 9, border: "0" }}
          isConnectable
        />
      );
    });

  return (
    <div
      className={
        `relative ${BOX_MIN[size]} ${PAD[size]} rounded-xl border backdrop-blur ` +
        (selected
          ? "border-beam-400/70 bg-slate-900/90 shadow-[0_0_30px_-10px_rgba(34,211,238,0.6)]"
          : "border-white/15 bg-slate-900/80")
      }
    >
      {renderHandles(left, Position.Left)}
      {renderHandles(right, Position.Right)}
      {renderHandles(top, Position.Top)}
      {renderHandles(bottom, Position.Bottom)}

      <div className="flex flex-col items-center">
        <div style={{ width: SYM_W[size] }}>
          <Symbol symbol={(d.symbol as SymbolKey) ?? "generic"} />
        </div>
        <div className="mt-1 text-center">
          <div className={size === "sm" ? "text-[10px] text-slate-200" : "text-xs text-slate-100"}>
            {d.label}
          </div>
          <div className="mt-0.5 font-mono text-[9px] tracking-wider text-slate-500">
            {d.value ? d.value : (d.partCode ?? d.ctype)}
          </div>
        </div>
      </div>
    </div>
  );
}

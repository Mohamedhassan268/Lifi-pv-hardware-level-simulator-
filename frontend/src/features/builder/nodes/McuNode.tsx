/**
 * McuNode — the node controller block (ESP32 / Arduino). Reads the receiver's
 * output (left handle); the chip body with leads is drawn inline. Tier A: a
 * representational block carrying the MCU parameters — not a pipeline stage.
 */

import { type Node, type NodeProps } from "@xyflow/react";

import { NodeShell } from "@/features/builder/nodes/NodeShell";
import type { BlockData } from "@/features/builder/nodes/nodeTypes";

export function McuNode({ data }: NodeProps<Node<BlockData>>) {
  const { configured } = data;
  const stroke = configured ? "#67e8f9" : "#475569";
  const body = configured ? "rgba(103,232,249,0.08)" : "transparent";

  return (
    <NodeShell
      title={data.title}
      subtitle={data.subtitle}
      chips={data.chips}
      configured={configured}
      accent="beam"
      showHandles={{ left: true, right: false }}
    >
      <svg width="120" height="64" viewBox="0 0 120 64" aria-hidden>
        {/* Chip body */}
        <rect x={36} y={14} width={48} height={36} rx={3} fill={body} stroke={stroke} strokeWidth={1.5} />
        {/* Leads — left (from RX) and right */}
        {[22, 32, 42].map((y) => (
          <line key={`l${y}`} x1={26} y1={y} x2={36} y2={y} stroke={stroke} strokeWidth={1.2} />
        ))}
        {[22, 32, 42].map((y) => (
          <line key={`r${y}`} x1={84} y1={y} x2={94} y2={y} stroke={stroke} strokeWidth={1.2} />
        ))}
        <text x={45} y={36} fontSize={9} fill={configured ? "#cbd5e1" : "#64748b"} stroke="none">
          MCU
        </text>
      </svg>
    </NodeShell>
  );
}

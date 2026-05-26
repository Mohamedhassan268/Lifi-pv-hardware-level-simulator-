/**
 * TxNode — TX block illustration: LED die + lens + emanating beam dashes.
 *
 * When the TX step is running, the beam dashes pulse outward.
 */

import { motion } from "framer-motion";
import { type Node, type NodeProps } from "@xyflow/react";

import { NodeShell } from "@/features/builder/nodes/NodeShell";
import type { BlockData } from "@/features/builder/nodes/nodeTypes";
import { usePipelineStore } from "@/store/pipelineStore";

export function TxNode({ data }: NodeProps<Node<BlockData>>) {
  const running = usePipelineStore((s) => s.steps.TX.status === "running");
  const { configured } = data;

  return (
    <NodeShell
      title={data.title}
      subtitle={data.subtitle}
      chips={data.chips}
      configured={configured}
      step={data.step}
      accent="beam"
      showHandles={{ left: false, right: true }}
    >
      <svg width="180" height="64" viewBox="0 0 180 64" aria-hidden>
        {/* LED body (dome on left) */}
        <defs>
          <radialGradient id="led-glow" cx="0.3" cy="0.5" r="0.7">
            <stop offset="0%" stopColor={configured ? "#67e8f9" : "#475569"} stopOpacity="0.95" />
            <stop offset="60%" stopColor={configured ? "#22d3ee" : "#334155"} stopOpacity="0.5" />
            <stop offset="100%" stopColor="transparent" />
          </radialGradient>
          <linearGradient id="lens-grad" x1="0" y1="0.5" x2="1" y2="0.5">
            <stop offset="0%" stopColor="rgba(148,163,184,0.6)" />
            <stop offset="100%" stopColor="rgba(148,163,184,0.1)" />
          </linearGradient>
        </defs>

        {/* PCB pad */}
        <rect x={4} y={42} width={36} height={8} rx={1} fill={configured ? "#334155" : "#1e293b"} />
        {/* LED die */}
        <rect x={14} y={26} width={16} height={20} rx={2} fill={configured ? "#0f172a" : "#1e293b"} stroke={configured ? "#22d3ee" : "#475569"} strokeWidth={1.2} />
        {/* LED dome */}
        <ellipse cx={32} cy={36} rx={14} ry={14} fill="url(#led-glow)" />
        {/* Lens body */}
        <path d="M 56 18 Q 70 36 56 54 L 80 54 Q 90 36 80 18 Z" fill="url(#lens-grad)" stroke={configured ? "rgba(103,232,249,0.5)" : "rgba(148,163,184,0.3)"} strokeWidth={1} />

        {/* Beam dashes — animate outward when TX running */}
        {configured && [0, 1, 2, 3, 4].map((i) => (
          <motion.line
            key={i}
            x1={90 + i * 17}
            y1={36}
            x2={100 + i * 17}
            y2={36}
            stroke="#67e8f9"
            strokeWidth={2}
            strokeLinecap="round"
            initial={{ opacity: running ? 0 : 0.6 }}
            animate={
              running
                ? { opacity: [0, 1, 0], x: [0, 6, 12] }
                : { opacity: 0.6 }
            }
            transition={{
              duration: 1.1,
              repeat: running ? Infinity : 0,
              delay: i * 0.15,
              ease: "easeOut",
            }}
          />
        ))}
      </svg>
    </NodeShell>
  );
}

/**
 * RxNode — receiver illustration. PV cell (busbars + fingers) by default;
 * a photodiode body if config.pv_part looks like a Si/Ge photodiode part.
 *
 * Photons drift in from the left and get absorbed on the surface when the
 * RX step is running.
 */

import { motion } from "framer-motion";
import { type Node, type NodeProps } from "@xyflow/react";
import { useMemo } from "react";

import { NodeShell } from "@/features/builder/nodes/NodeShell";
import type { BlockData } from "@/features/builder/nodes/nodeTypes";
import { useConfigStore } from "@/store/configStore";
import { usePipelineStore } from "@/store/pipelineStore";

const PHOTODIODE_PARTS = new Set(["BPW34", "SFH206K", "VEMD5510", "PhotodiodeFromSPICE"]);

export function RxNode({ data }: NodeProps<Node<BlockData>>) {
  const running = usePipelineStore((s) => s.steps.RX.status === "running");
  const pvPart = useConfigStore((s) => (s.config.pv_part as string | undefined) ?? "");
  const isPhotodiode = PHOTODIODE_PARTS.has(pvPart);
  const { configured } = data;

  // Pre-randomize photon offsets so we don't shuffle on every render
  const photonDelays = useMemo(() => [0, 0.18, 0.36, 0.54, 0.72, 0.9], []);

  return (
    <NodeShell
      title={data.title}
      subtitle={data.subtitle}
      chips={data.chips}
      configured={configured}
      step={data.step}
      accent="harvest"
      showHandles={{ left: true, right: false }}
    >
      <svg width="180" height="64" viewBox="0 0 180 64" aria-hidden>
        <defs>
          <linearGradient id="pv-grad" x1="0.5" y1="0" x2="0.5" y2="1">
            <stop offset="0%" stopColor={configured ? "#7c2d12" : "#1e293b"} />
            <stop offset="100%" stopColor={configured ? "#431407" : "#0f172a"} />
          </linearGradient>
          <radialGradient id="pd-body" cx="0.5" cy="0.5" r="0.6">
            <stop offset="0%" stopColor={configured ? "#0f172a" : "#1e293b"} />
            <stop offset="100%" stopColor={configured ? "#020617" : "#0f172a"} />
          </radialGradient>
        </defs>

        {/* Incoming photons */}
        {configured && photonDelays.map((d, i) => (
          <motion.circle
            key={i}
            r={2}
            fill="#67e8f9"
            initial={{ cx: 4, cy: 32 + (i % 3 - 1) * 8, opacity: 0 }}
            animate={
              running
                ? { cx: [4, 60], opacity: [0, 1, 0] }
                : { cx: 4, opacity: 0.3 }
            }
            transition={{
              duration: 1.0,
              repeat: running ? Infinity : 0,
              delay: d,
              ease: "easeIn",
            }}
            style={{ filter: "drop-shadow(0 0 3px rgba(103,232,249,0.7))" }}
          />
        ))}

        {isPhotodiode ? (
          // Photodiode: round body with two leads
          <g>
            <line x1={130} y1={32} x2={160} y2={20} stroke={configured ? "#94a3b8" : "#475569"} strokeWidth={1.2} />
            <line x1={130} y1={32} x2={160} y2={44} stroke={configured ? "#94a3b8" : "#475569"} strokeWidth={1.2} />
            <circle cx={100} cy={32} r={26} fill="url(#pd-body)" stroke={configured ? "#fbbf24" : "#475569"} strokeWidth={1.5} />
            {/* Active region */}
            <circle cx={100} cy={32} r={10} fill={configured ? "#fef3c7" : "#334155"} opacity={configured ? 0.85 : 1} />
          </g>
        ) : (
          // PV cell: rectangular, with busbars and grid fingers
          <g>
            <rect x={66} y={12} width={96} height={40} rx={3} fill="url(#pv-grad)" stroke={configured ? "#fbbf24" : "#475569"} strokeWidth={1.2} />
            {/* Two horizontal busbars */}
            <line x1={66} y1={24} x2={162} y2={24} stroke={configured ? "#fbbf24" : "#64748b"} strokeWidth={1.1} />
            <line x1={66} y1={40} x2={162} y2={40} stroke={configured ? "#fbbf24" : "#64748b"} strokeWidth={1.1} />
            {/* Vertical fingers */}
            {[72, 84, 96, 108, 120, 132, 144, 156].map((x) => (
              <line
                key={x}
                x1={x}
                y1={14}
                x2={x}
                y2={50}
                stroke={configured ? "rgba(251,191,36,0.55)" : "rgba(148,163,184,0.3)"}
                strokeWidth={0.5}
              />
            ))}
          </g>
        )}
      </svg>
    </NodeShell>
  );
}

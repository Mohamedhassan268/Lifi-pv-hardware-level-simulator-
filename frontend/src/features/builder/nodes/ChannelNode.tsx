/**
 * ChannelNode — optical-link illustration: Gaussian beam envelope between
 * two virtual aperture pegs. Hosts the NoiseOverlay when noise is applied.
 */

import { type Node, type NodeProps } from "@xyflow/react";

import { NoiseOverlay } from "@/features/builder/NoiseOverlay";
import { NodeShell } from "@/features/builder/nodes/NodeShell";
import type { BlockData } from "@/features/builder/nodes/nodeTypes";

export function ChannelNode({ data }: NodeProps<Node<BlockData>>) {
  const { configured } = data;
  const showNoise = Boolean(data.showNoise);

  return (
    <NodeShell
      title={data.title}
      subtitle={data.subtitle}
      chips={data.chips}
      configured={configured}
      step={data.step}
      accent="slate"
    >
      <svg width="200" height="64" viewBox="0 0 200 64" aria-hidden className="overflow-visible">
        <defs>
          <linearGradient id="beam-env" x1="0" y1="0.5" x2="1" y2="0.5">
            <stop offset="0%" stopColor="rgba(103,232,249,0.55)" />
            <stop offset="50%" stopColor="rgba(103,232,249,0.85)" />
            <stop offset="100%" stopColor="rgba(251,191,36,0.55)" />
          </linearGradient>
          <linearGradient id="beam-fill" x1="0" y1="0.5" x2="1" y2="0.5">
            <stop offset="0%" stopColor="rgba(103,232,249,0.05)" />
            <stop offset="100%" stopColor="rgba(251,191,36,0.05)" />
          </linearGradient>
        </defs>

        {/* Aperture pegs */}
        <line x1={8} y1={12} x2={8} y2={52} stroke={configured ? "#67e8f9" : "#475569"} strokeWidth={1.5} />
        <line x1={192} y1={12} x2={192} y2={52} stroke={configured ? "#fbbf24" : "#475569"} strokeWidth={1.5} />

        {/* Gaussian beam envelope */}
        <path
          d="M 8 22 Q 100 8 192 22 L 192 42 Q 100 56 8 42 Z"
          fill={configured ? "url(#beam-fill)" : "transparent"}
          stroke={configured ? "url(#beam-env)" : "rgba(148,163,184,0.35)"}
          strokeWidth={1.2}
          strokeDasharray={configured ? "none" : "3 3"}
        />

        {/* Beam axis */}
        <line
          x1={8}
          y1={32}
          x2={192}
          y2={32}
          stroke={configured ? "rgba(103,232,249,0.45)" : "rgba(148,163,184,0.25)"}
          strokeWidth={0.8}
          strokeDasharray="2 3"
        />

        {/* Noise overlay (visible only when noise category is configured AND noise_enable=true) */}
        {showNoise && <NoiseOverlay />}
      </svg>
    </NodeShell>
  );
}

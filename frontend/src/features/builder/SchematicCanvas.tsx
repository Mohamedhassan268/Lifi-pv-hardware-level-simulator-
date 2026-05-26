/**
 * SchematicCanvas — right-panel live schematic, driven by @xyflow/react.
 *
 * Topology is fixed (TX → Channel → RX). Each node type has its own
 * SVG-rich component:
 *   - TxNode      — LED + lens + emanating beam dashes
 *   - ChannelNode — Gaussian envelope between two aperture pegs
 *                   (hosts the NoiseOverlay when noise is applied)
 *   - RxNode      — PV cell grid (or photodiode body) with incoming photons
 *
 * State sources:
 *   - configStore.config           — live config; chip labels read from here
 *   - builderUIStore.configured    — drives the "Not configured" dashed state
 *   - builderUIStore.schematicRev  — bumped by Apply; forces nodes/edges rebuild
 *   - pipelineStore.steps          — TX/Channel/RX running status during Simulate
 */

import {
  Background,
  ReactFlow,
  type Edge,
  type EdgeMouseHandler,
  type Node,
  type NodeMouseHandler,
} from "@xyflow/react";
import { useCallback, useMemo } from "react";

import { ChannelNode } from "@/features/builder/nodes/ChannelNode";
import { RxNode } from "@/features/builder/nodes/RxNode";
import { TxNode } from "@/features/builder/nodes/TxNode";
import type { BlockData } from "@/features/builder/nodes/nodeTypes";
import { useGlassBoxPipeline } from "@/features/probes/useGlassBoxPipeline";
import { useBuilderUIStore, type BuilderCategory } from "@/store/builderUIStore";
import { useConfigStore } from "@/store/configStore";
import { usePipelineExecutionStore } from "@/store/pipelineExecutionStore";

const nodeTypes = { tx: TxNode, channel: ChannelNode, rx: RxNode };

// Canvas node id → inspector entity. Channel block configures the geometry entity.
const NODE_TO_ENTITY: Record<string, BuilderCategory> = {
  tx: "transmitter",
  channel: "geometry",
  rx: "receiver",
};

// Edge → probe ID mapping. Matches cosim.probes registry edge_id fields.
const EDGE_PROBE: Record<string, string> = {
  "tx-channel": "tx.P_tx",
  "channel-rx": "channel.P_rx",
};

export function SchematicCanvas() {
  const config = useConfigStore((s) => s.config);
  const configured = useBuilderUIStore((s) => s.configured);
  const schematicRev = useBuilderUIStore((s) => s.schematicRev);
  const selectedEntity = useBuilderUIStore((s) => s.selectedEntity);
  const selectEntity = useBuilderUIStore((s) => s.selectEntity);

  const availableProbes = usePipelineExecutionStore((s) => s.availableProbes);
  const openProbe = usePipelineExecutionStore((s) => s.openProbe);
  const { fetchProbe } = useGlassBoxPipeline();

  const { nodes, edges } = useMemo(
    () => buildGraph(config, configured, selectedEntity, availableProbes),
    // schematicRev forces a rebuild on Apply even if config-object identity tricks memo.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [config, configured, schematicRev, selectedEntity, availableProbes],
  );

  const onNodeClick: NodeMouseHandler = useCallback(
    (_, node) => {
      const entity = NODE_TO_ENTITY[node.id];
      if (entity) selectEntity(entity);
    },
    [selectEntity],
  );

  const onEdgeClick: EdgeMouseHandler = useCallback(
    (_, edge) => {
      const probeId = EDGE_PROBE[edge.id];
      if (!probeId) return;
      if (!availableProbes.includes(probeId)) return;
      openProbe(probeId);
      fetchProbe(probeId);
    },
    [availableProbes, fetchProbe, openProbe],
  );

  return (
    <div className="relative h-full w-full bg-gradient-to-br from-slate-950 via-slate-950 to-slate-900/40">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        onNodeClick={onNodeClick}
        onEdgeClick={onEdgeClick}
        zoomOnScroll={false}
        panOnScroll={false}
        panOnDrag={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
        fitView
        fitViewOptions={{ padding: 0.2 }}
      >
        <Background gap={24} size={1} color="rgba(255,255,255,0.05)" />
      </ReactFlow>

      <div className="pointer-events-none absolute left-5 top-5 text-[10px] uppercase tracking-[0.2em] text-slate-500">
        Live schematic — click a wire to probe
      </div>
    </div>
  );
}

function buildGraph(
  c: Record<string, unknown>,
  cfg: Record<BuilderCategory, boolean>,
  selectedEntity: BuilderCategory | null,
  availableProbes: string[],
): { nodes: Node<BlockData>[]; edges: Edge[] } {
  const power = num(c.led_radiated_power_mW, 9.3);
  const bias = num(c.bias_current_A, 0.35);
  const modulation = (c.modulation as string) ?? "OOK";
  const rate = num(c.data_rate_bps, 5000);
  const distance = num(c.distance_m, 0.325);
  const fov = num(c.fov_half_angle_deg, 90);
  const ledHalf = num(c.led_half_angle_deg, 9);
  const reflections = num(c.n_reflections, 0);
  const pv = (c.pv_part as string) ?? "—";
  const gain = num(c.ina_gain_dB, 40);
  const adcBits = num(c.adc_bits, 12);
  const noiseOn = cfg.noise && Boolean(c.noise_enable);

  const channelChips: string[] = [
    `${distance.toFixed(2)} m`,
    `±${ledHalf.toFixed(0)}° beam`,
    `${fov.toFixed(0)}° FoV`,
    reflections > 0 ? `${reflections} refl` : "LOS",
  ];
  if (cfg.noise) channelChips.push(noiseOn ? "noise ON" : "noise off");

  const nodes: Node<BlockData>[] = [
    {
      id: "tx",
      type: "tx",
      position: { x: 30, y: 100 },
      selected: selectedEntity === "transmitter",
      data: {
        title: "Transmitter",
        subtitle: cfg.transmitter ? ((c.led_part as string) ?? "LED") : "Not configured",
        chips: cfg.transmitter
          ? [
              `${power.toFixed(1)} mW`,
              `${(bias * 1000).toFixed(0)} mA`,
              modulation,
              `${(rate / 1000).toFixed(1)} kbps`,
            ]
          : [],
        step: "TX",
        configured: cfg.transmitter,
      },
    },
    {
      id: "channel",
      type: "channel",
      position: { x: 350, y: 100 },
      selected: selectedEntity === "geometry" || selectedEntity === "noise",
      data: {
        title: "Channel",
        subtitle: cfg.geometry ? "Optical link" : "Not configured",
        chips: cfg.geometry ? channelChips : [],
        step: "Channel",
        configured: cfg.geometry,
      },
    },
    {
      id: "rx",
      type: "rx",
      position: { x: 670, y: 100 },
      selected: selectedEntity === "receiver",
      data: {
        title: "Receiver",
        subtitle: cfg.receiver ? pv : "Not configured",
        chips: cfg.receiver
          ? [`${gain.toFixed(0)} dB`, `${adcBits.toFixed(0)}-bit ADC`, `${modulation} demod`]
          : [],
        step: "RX",
        configured: cfg.receiver,
      },
    },
  ];

  const txChEdgeOn = cfg.transmitter && cfg.geometry;
  const chRxEdgeOn = cfg.geometry && cfg.receiver;

  const txChProbe = "tx.P_tx";
  const chRxProbe = "channel.P_rx";
  const txChProbed = availableProbes.includes(txChProbe);
  const chRxProbed = availableProbes.includes(chRxProbe);

  const edges: Edge[] = [
    {
      id: "tx-channel",
      source: "tx",
      target: "channel",
      animated: txChEdgeOn,
      label: txChProbed ? "▣ tx.P_tx" : undefined,
      labelStyle: { fill: "#67e8f9", fontSize: 9, letterSpacing: "0.1em" },
      labelBgPadding: [2, 4],
      labelBgStyle: { fill: "rgba(15,23,42,0.85)" },
      style: {
        stroke: txChProbed
          ? "rgba(103,232,249,0.9)"
          : txChEdgeOn
            ? "rgba(103,232,249,0.55)"
            : "rgba(255,255,255,0.12)",
        strokeWidth: txChProbed ? 2 : 1.5,
        strokeDasharray: txChEdgeOn ? undefined : "4 4",
        cursor: txChProbed ? "pointer" : "default",
      },
      data: { probeId: txChProbe, probed: txChProbed },
    },
    {
      id: "channel-rx",
      source: "channel",
      target: "rx",
      animated: chRxEdgeOn,
      label: chRxProbed ? "▣ channel.P_rx" : undefined,
      labelStyle: { fill: "#fbbf24", fontSize: 9, letterSpacing: "0.1em" },
      labelBgPadding: [2, 4],
      labelBgStyle: { fill: "rgba(15,23,42,0.85)" },
      style: {
        stroke: chRxProbed
          ? "rgba(251,191,36,0.9)"
          : chRxEdgeOn
            ? "rgba(251,191,36,0.5)"
            : "rgba(255,255,255,0.12)",
        strokeWidth: chRxProbed ? 2 : 1.5,
        strokeDasharray: chRxEdgeOn ? undefined : "4 4",
        cursor: chRxProbed ? "pointer" : "default",
      },
      data: { probeId: chRxProbe, probed: chRxProbed },
    },
  ];

  return { nodes, edges };
}

function num(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

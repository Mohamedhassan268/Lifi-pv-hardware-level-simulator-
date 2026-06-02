/**
 * SchematicCanvas — right-panel live schematic, driven by @xyflow/react.
 *
 * Renders one TX → Channel → RX row per system (A, and B when present). When
 * the systems are tied (duplex), cross-links A.TX→B.RX and B.TX→A.RX are drawn
 * between the rows. Each node type has its own SVG-rich component:
 *   - TxNode      — LED + lens + emanating beam dashes
 *   - ChannelNode — Gaussian envelope (hosts the NoiseOverlay)
 *   - RxNode      — PV cell grid (or photodiode body) with incoming photons
 *
 * Clicking a block selects that system + entity in the Inspector.
 *
 * State sources:
 *   - configStore.systems          — per-system live config; chips read here
 *   - builderUIStore.configuredBySystem — drives the "Not configured" state
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
import { McuNode } from "@/features/builder/nodes/McuNode";
import { RxNode } from "@/features/builder/nodes/RxNode";
import { TxNode } from "@/features/builder/nodes/TxNode";
import type { BlockData } from "@/features/builder/nodes/nodeTypes";
import { useGlassBoxPipeline } from "@/features/probes/useGlassBoxPipeline";
import {
  useBuilderUIStore,
  type BuilderCategory,
  type ConfiguredMap,
} from "@/store/builderUIStore";
import { useConfigStore, type CouplingMode, type SystemId } from "@/store/configStore";
import { usePipelineExecutionStore } from "@/store/pipelineExecutionStore";
import type { ConfigDict } from "@/api/client";

const nodeTypes = { tx: TxNode, channel: ChannelNode, rx: RxNode, mcu: McuNode };

// Canvas node base id → inspector entity. Channel configures the geometry entity.
const BASE_TO_ENTITY: Record<string, BuilderCategory> = {
  tx: "transmitter",
  channel: "geometry",
  rx: "receiver",
  mcu: "mcu",
};

// Edge → probe ID mapping (system A only — backend probes aren't system-tagged
// yet). Matches cosim.probes registry edge_id fields.
const EDGE_PROBE: Record<string, string> = {
  "tx-A-channel-A": "tx.P_tx",
  "channel-A-rx-A": "channel.P_rx",
};

const ROW_Y: Record<SystemId, number> = { A: 40, B: 300 };

export function SchematicCanvas() {
  const systems = useConfigStore((s) => s.systems);
  const activeSystem = useConfigStore((s) => s.activeSystem);
  const coupling = useConfigStore((s) => s.coupling);
  const setActiveSystem = useConfigStore((s) => s.setActiveSystem);

  const configuredBySystem = useBuilderUIStore((s) => s.configuredBySystem);
  const schematicRev = useBuilderUIStore((s) => s.schematicRev);
  const selectedEntity = useBuilderUIStore((s) => s.selectedEntity);
  const selectEntity = useBuilderUIStore((s) => s.selectEntity);

  const availableProbes = usePipelineExecutionStore((s) => s.availableProbes);
  const openProbe = usePipelineExecutionStore((s) => s.openProbe);
  const { fetchProbe } = useGlassBoxPipeline();

  const { nodes, edges } = useMemo(
    () =>
      buildScene({
        systems,
        configuredBySystem,
        activeSystem,
        coupling,
        selectedEntity,
        availableProbes,
      }),
    // schematicRev forces a rebuild on Apply even if config-object identity tricks memo.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [systems, configuredBySystem, activeSystem, coupling, schematicRev, selectedEntity, availableProbes],
  );

  const onNodeClick: NodeMouseHandler = useCallback(
    (_, node) => {
      // node.id is `${base}-${system}`, e.g. "tx-A".
      const [base, sys] = splitNodeId(node.id);
      const entity = BASE_TO_ENTITY[base];
      if (!entity) return;
      if (sys && sys !== activeSystem) setActiveSystem(sys);
      selectEntity(entity);
    },
    [activeSystem, setActiveSystem, selectEntity],
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

function splitNodeId(id: string): [string, SystemId | null] {
  const m = /^(.*)-([AB])$/.exec(id);
  return m ? [m[1], m[2] as SystemId] : [id, null];
}

interface SceneArgs {
  systems: { A: ConfigDict; B: ConfigDict | null };
  configuredBySystem: Record<SystemId, ConfiguredMap>;
  activeSystem: SystemId;
  coupling: CouplingMode;
  selectedEntity: BuilderCategory | null;
  availableProbes: string[];
}

const tag = (base: string, sys: SystemId) => `${base}-${sys}`;

function entityMatch(sel: BuilderCategory | null, entity: BuilderCategory): boolean {
  return sel === entity || (entity === "geometry" && sel === "noise");
}

function buildScene(args: SceneArgs): { nodes: Node<BlockData>[]; edges: Edge[] } {
  const { systems, configuredBySystem, activeSystem, coupling, selectedEntity, availableProbes } = args;

  // Shared-channel topology: both systems feed one channel block.
  if (coupling === "shared" && systems.B) {
    return sharedChannelScene(args);
  }

  const present: SystemId[] = systems.B ? ["A", "B"] : ["A"];
  const nodes: Node<BlockData>[] = [];
  const edges: Edge[] = [];

  for (const sys of present) {
    const c = systems[sys]!;
    const conf = configuredBySystem[sys];
    const active = sys === activeSystem;
    const y = ROW_Y[sys];
    const sel = (e: BuilderCategory) => active && entityMatch(selectedEntity, e);

    nodes.push(
      txNode(sys, c, conf, { x: 30, y }, sel("transmitter")),
      channelNode(sys, c, conf, { x: 350, y }, sel("geometry")),
      rxNode(sys, c, conf, { x: 670, y }, sel("receiver")),
      mcuNode(sys, c, conf, { x: 990, y }, sel("mcu")),
    );
    edges.push(
      linkEdge(tag("tx", sys), tag("channel", sys), conf.transmitter && conf.geometry, "#67e8f9", availableProbes),
      linkEdge(tag("channel", sys), tag("rx", sys), conf.geometry && conf.receiver, "#fbbf24", availableProbes),
      linkEdge(tag("rx", sys), tag("mcu", sys), conf.receiver && conf.mcu, "#34d399", []),
    );
  }

  // Duplex cross-links between the two systems (A.TX→B.RX, B.TX→A.RX).
  if (coupling === "duplex" && systems.B) {
    edges.push(crossLink("tx-A", "rx-B", "A→B"), crossLink("tx-B", "rx-A", "B→A"));
  }

  return { nodes, edges };
}

// Both systems' TX feed one channel, which feeds both RX. The shared channel
// uses system A's config (B's channel/geometry fields are kept in sync).
function sharedChannelScene(args: SceneArgs): { nodes: Node<BlockData>[]; edges: Edge[] } {
  const { systems, configuredBySystem, activeSystem, selectedEntity, availableProbes } = args;
  const A = systems.A;
  const B = systems.B!;
  const cA = configuredBySystem.A;
  const cB = configuredBySystem.B;
  const selFor = (sys: SystemId, e: BuilderCategory) => activeSystem === sys && entityMatch(selectedEntity, e);
  const chId = tag("channel", "A"); // shared channel carries A's id

  const nodes: Node<BlockData>[] = [
    txNode("A", A, cA, { x: 30, y: 40 }, selFor("A", "transmitter")),
    txNode("B", B, cB, { x: 30, y: 300 }, selFor("B", "transmitter")),
    channelNode("A", A, cA, { x: 360, y: 170 }, entityMatch(selectedEntity, "geometry")),
    rxNode("A", A, cA, { x: 690, y: 40 }, selFor("A", "receiver")),
    rxNode("B", B, cB, { x: 690, y: 300 }, selFor("B", "receiver")),
    mcuNode("A", A, cA, { x: 1010, y: 40 }, selFor("A", "mcu")),
    mcuNode("B", B, cB, { x: 1010, y: 300 }, selFor("B", "mcu")),
  ];

  const edges: Edge[] = [
    linkEdge(tag("tx", "A"), chId, cA.transmitter && cA.geometry, "#67e8f9", availableProbes),
    linkEdge(tag("tx", "B"), chId, cB.transmitter && cA.geometry, "#67e8f9", []),
    linkEdge(chId, tag("rx", "A"), cA.geometry && cA.receiver, "#fbbf24", availableProbes),
    linkEdge(chId, tag("rx", "B"), cA.geometry && cB.receiver, "#fbbf24", []),
    linkEdge(tag("rx", "A"), tag("mcu", "A"), cA.receiver && cA.mcu, "#34d399", []),
    linkEdge(tag("rx", "B"), tag("mcu", "B"), cB.receiver && cB.mcu, "#34d399", []),
  ];

  return { nodes, edges };
}

function txNode(
  sys: SystemId,
  c: ConfigDict,
  conf: ConfiguredMap,
  position: { x: number; y: number },
  selected: boolean,
): Node<BlockData> {
  const modulation = (c.modulation as string) ?? "OOK";
  return {
    id: tag("tx", sys),
    type: "tx",
    position,
    selected,
    data: {
      title: "Transmitter",
      subtitle: conf.transmitter ? ((c.led_part as string) ?? "LED") : "Not configured",
      chips: conf.transmitter
        ? [
            `${num(c.led_radiated_power_mW, 9.3).toFixed(1)} mW`,
            `${(num(c.bias_current_A, 0.35) * 1000).toFixed(0)} mA`,
            modulation,
            `${(num(c.data_rate_bps, 5000) / 1000).toFixed(1)} kbps`,
          ]
        : [],
      step: "TX",
      configured: conf.transmitter,
    },
  };
}

function channelNode(
  sys: SystemId,
  c: ConfigDict,
  conf: ConfiguredMap,
  position: { x: number; y: number },
  selected: boolean,
): Node<BlockData> {
  const noiseOn = conf.noise && Boolean(c.noise_enable);
  const chips: string[] = [
    `${num(c.distance_m, 0.325).toFixed(2)} m`,
    `±${num(c.led_half_angle_deg, 9).toFixed(0)}° beam`,
    `${num(c.fov_half_angle_deg, 90).toFixed(0)}° FoV`,
    num(c.n_reflections, 0) > 0 ? `${num(c.n_reflections, 0)} refl` : "LOS",
  ];
  if (conf.noise) chips.push(noiseOn ? "noise ON" : "noise off");
  return {
    id: tag("channel", sys),
    type: "channel",
    position,
    selected,
    data: {
      title: "Channel",
      subtitle: conf.geometry ? "Optical link" : "Not configured",
      chips: conf.geometry ? chips : [],
      step: "Channel",
      configured: conf.geometry,
      showNoise: noiseOn,
    },
  };
}

function rxNode(
  sys: SystemId,
  c: ConfigDict,
  conf: ConfiguredMap,
  position: { x: number; y: number },
  selected: boolean,
): Node<BlockData> {
  const modulation = (c.modulation as string) ?? "OOK";
  return {
    id: tag("rx", sys),
    type: "rx",
    position,
    selected,
    data: {
      title: "Receiver",
      subtitle: conf.receiver ? ((c.pv_part as string) ?? "—") : "Not configured",
      chips: conf.receiver
        ? [
            `${num(c.ina_gain_dB, 40).toFixed(0)} dB`,
            `${num(c.adc_bits, 12).toFixed(0)}-bit ADC`,
            `${modulation} demod`,
          ]
        : [],
      step: "RX",
      configured: conf.receiver,
    },
  };
}

function mcuNode(
  sys: SystemId,
  c: ConfigDict,
  conf: ConfiguredMap,
  position: { x: number; y: number },
  selected: boolean,
): Node<BlockData> {
  return {
    id: tag("mcu", sys),
    type: "mcu",
    position,
    selected,
    data: {
      title: "Controller",
      subtitle: conf.mcu ? ((c.mcu_board as string) ?? "MCU") : "Not configured",
      chips: conf.mcu
        ? [
            `${num(c.mcu_clock_MHz, 240).toFixed(0)} MHz`,
            `${num(c.adc_bits, 12).toFixed(0)}-bit ADC`,
          ]
        : [],
      configured: conf.mcu,
    },
  };
}

function linkEdge(
  source: string,
  target: string,
  on: boolean,
  color: string,
  availableProbes: string[],
): Edge {
  const id = `${source}-${target}`;
  const probeId = EDGE_PROBE[id];
  const probed = probeId ? availableProbes.includes(probeId) : false;
  const COLOR_RGB: Record<string, [number, number, number]> = {
    "#67e8f9": [103, 232, 249],
    "#fbbf24": [251, 191, 36],
    "#34d399": [52, 211, 153],
  };
  const [r, g, b] = COLOR_RGB[color] ?? [148, 163, 184];
  return {
    id,
    source,
    target,
    animated: on,
    label: probed && probeId ? `▣ ${probeId}` : undefined,
    labelStyle: { fill: color, fontSize: 9, letterSpacing: "0.1em" },
    labelBgPadding: [2, 4],
    labelBgStyle: { fill: "rgba(15,23,42,0.85)" },
    style: {
      stroke: probed
        ? `rgba(${r},${g},${b},0.9)`
        : on
          ? `rgba(${r},${g},${b},0.55)`
          : "rgba(255,255,255,0.12)",
      strokeWidth: probed ? 2 : 1.5,
      strokeDasharray: on ? undefined : "4 4",
      cursor: probed ? "pointer" : "default",
    },
    data: probeId ? { probeId, probed } : undefined,
  };
}

// Violet, dashed cross-link marking the duplex coupling between the two systems.
function crossLink(source: string, target: string, label: string): Edge {
  return {
    id: `duplex-${source}-${target}`,
    source,
    target,
    animated: true,
    type: "smoothstep",
    label: `duplex ${label}`,
    labelStyle: { fill: "#c084fc", fontSize: 9, letterSpacing: "0.1em" },
    labelBgPadding: [2, 4],
    labelBgStyle: { fill: "rgba(15,23,42,0.85)" },
    style: { stroke: "rgba(192,132,252,0.7)", strokeWidth: 1.5, strokeDasharray: "5 4" },
  };
}

function num(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

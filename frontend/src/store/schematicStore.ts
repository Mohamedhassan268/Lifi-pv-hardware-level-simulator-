/**
 * schematicStore — state for the drag-and-drop schematic editor.
 *
 * Holds React Flow nodes/edges plus a cache of each part's SPICE ports
 * (fetched lazily from /api/components/{part}). `toCircuitGraph()` serializes
 * the canvas into the same shape as backend kicad/circuit_graph.py
 * (components + nets), which the simulate endpoint consumes.
 *
 * A React Flow edge connects (sourceNode, sourceHandle) <-> (targetNode,
 * targetHandle); each handle id is a port name. We resolve nets by union-find
 * over the edges so any pins wired together (directly or transitively) share
 * one net.
 */

import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from "@xyflow/react";
import { create } from "zustand";

import type { SpicePort } from "@/api/client";

export interface PartNodeData extends Record<string, unknown> {
  /** Registry part name, e.g. "INA322", or a generic type "R" / "V". */
  part: string;
  /** SPICE subcircuit / element type, e.g. "INA322", "R". */
  ctype: string;
  /** Friendly label shown as the node title, e.g. "Instrumentation Amp". */
  label: string;
  /** Searchable part code shown as the node subtitle, e.g. "INA322". */
  partCode?: string;
  /** Symbol glyph key (see features/schematic/symbols). */
  symbol?: string;
  ports: SpicePort[];
  /** Free-form value for primitives (resistance, dc volts). */
  value?: string;
}

export interface CircuitGraphComponent {
  ref: string;
  component_type: string;
  value: string;
  pins: Record<number, string>;
}

export interface CircuitGraphNet {
  name: string;
  pins: { component_ref: string; pin_number: number }[];
}

export interface CircuitGraphPayload {
  title: string;
  components: CircuitGraphComponent[];
  nets: CircuitGraphNet[];
}

interface SchematicState {
  nodes: Node<PartNodeData>[];
  edges: Edge[];
  selectedNodeId: string | null;
  refCounter: number;
  /** First pin armed by a Ctrl+click, awaiting a second pin to wire to. */
  pendingPin: { node: string; handle: string } | null;

  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (c: Connection) => void;
  addPart: (data: PartNodeData, position: { x: number; y: number }) => void;
  selectNode: (id: string | null) => void;
  setNodeValue: (id: string, value: string) => void;
  removeSelected: () => void;
  clear: () => void;
  loadGraph: (nodes: Node<PartNodeData>[], edges: Edge[]) => void;
  toCircuitGraph: (title?: string) => CircuitGraphPayload;
  /** Ctrl+click wiring: arm the first pin, then wire to the second. */
  clickPin: (node: string, handle: string) => void;
  clearPending: () => void;
}

// Reference designator prefix per component type.
function refPrefix(ctype: string): string {
  if (ctype === "R") return "R";
  if (ctype === "C") return "C";
  if (ctype === "V") return "V";
  if (ctype === "SOLAR_CELL") return "Xsc";
  return "X"; // subcircuit instances
}

export const useSchematicStore = create<SchematicState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  refCounter: 0,
  pendingPin: null,

  onNodesChange: (changes) =>
    set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) as Node<PartNodeData>[] })),

  onEdgesChange: (changes) =>
    set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),

  onConnect: (c) =>
    set((s) => ({
      edges: addEdge(
        { ...c, type: "smoothstep", style: { stroke: "#94a3b8", strokeWidth: 1.6 } },
        s.edges,
      ),
    })),

  addPart: (data, position) =>
    set((s) => {
      const n = s.refCounter + 1;
      const id = `${refPrefix(data.ctype)}${n}`;
      const node: Node<PartNodeData> = {
        id,
        type: "part",
        position,
        data,
      };
      return { nodes: [...s.nodes, node], refCounter: n, selectedNodeId: id };
    }),

  selectNode: (id) => set({ selectedNodeId: id }),

  setNodeValue: (id, value) =>
    set((s) => ({
      nodes: s.nodes.map((nd) =>
        nd.id === id ? { ...nd, data: { ...nd.data, value } } : nd,
      ),
    })),

  removeSelected: () =>
    set((s) => {
      const id = s.selectedNodeId;
      if (!id) return {};
      return {
        nodes: s.nodes.filter((n) => n.id !== id),
        edges: s.edges.filter((e) => e.source !== id && e.target !== id),
        selectedNodeId: null,
      };
    }),

  clear: () => set({ nodes: [], edges: [], selectedNodeId: null, refCounter: 0 }),

  loadGraph: (nodes, edges) =>
    set({
      nodes,
      edges,
      selectedNodeId: null,
      // Keep refCounter ahead of any numeric suffix already used, so new
      // parts added after loading a preset don't collide with preset refs.
      refCounter: nodes.length,
    }),

  toCircuitGraph: (title = "User circuit") => {
    const { nodes, edges } = get();
    return serializeGraph(nodes, edges, title);
  },

  clickPin: (node, handle) =>
    set((s) => {
      const p = s.pendingPin;
      // First Ctrl+click arms the pin; a second one wires the two together.
      if (!p) return { pendingPin: { node, handle } };
      if (p.node === node && p.handle === handle) return { pendingPin: null }; // re-click cancels
      const newEdge: Edge = {
        id: `e-${p.node}.${p.handle}-${node}.${handle}`,
        source: p.node,
        sourceHandle: p.handle,
        target: node,
        targetHandle: handle,
        type: "smoothstep",
        style: { stroke: "#94a3b8", strokeWidth: 1.6 },
      };
      return { edges: addEdge(newEdge, s.edges), pendingPin: null };
    }),

  clearPending: () => set({ pendingPin: null }),
}));

/**
 * Union-find over edges to assign every (node, port) a net name, then build
 * the CircuitGraph payload. A port's 1-based index into its node's `ports`
 * list is its pin_number (matching cosim.graph_netlist PORT_ORDER ordering).
 */
export function serializeGraph(
  nodes: Node<PartNodeData>[],
  edges: Edge[],
  title: string,
): CircuitGraphPayload {
  const portKey = (nodeId: string, port: string) => `${nodeId}::${port}`;

  // union-find
  const parent = new Map<string, string>();
  const find = (x: string): string => {
    if (!parent.has(x)) parent.set(x, x);
    let root = x;
    while (parent.get(root) !== root) root = parent.get(root)!;
    parent.set(x, root);
    return root;
  };
  const union = (a: string, b: string) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  };

  // Register every pin
  for (const n of nodes) {
    for (const p of n.data.ports) find(portKey(n.id, p.name));
  }
  // Union wired pins
  for (const e of edges) {
    if (e.sourceHandle && e.targetHandle) {
      union(portKey(e.source, e.sourceHandle), portKey(e.target, e.targetHandle));
    }
  }

  // Group pins by net root, name nets net0, net1, ... (ground/supply kept by role below)
  const rootToPins = new Map<string, { nodeId: string; port: string; pinNo: number }[]>();
  for (const n of nodes) {
    n.data.ports.forEach((p, i) => {
      const root = find(portKey(n.id, p.name));
      const arr = rootToPins.get(root) ?? [];
      arr.push({ nodeId: n.id, port: p.name, pinNo: i + 1 });
      rootToPins.set(root, arr);
    });
  }

  // Net naming: if any pin in the group is a supply/ground/ref/optical role,
  // use a canonical name; otherwise auto-number.
  const ROLE_NET: Record<string, string> = {
    supply_pos: "vcc",
    supply_neg: "vee",
    ref: "vref",
    optical_in: "optical_power",
    output: "dout", // the Output terminal marks the net the BER reads as V(dout)
    ground: "0",    // the Ground terminal marks the SPICE reference node
  };
  const roleOf = (nodeId: string, port: string): string | undefined => {
    const nd = nodes.find((n) => n.id === nodeId);
    return nd?.data.ports.find((p) => p.name === port)?.role;
  };

  const nets: CircuitGraphNet[] = [];
  let auto = 0;
  for (const [, pins] of rootToPins) {
    // skip nets with a single unconnected pin? keep them — ERC handles floating.
    let name: string | undefined;
    for (const pin of pins) {
      const role = roleOf(pin.nodeId, pin.port);
      if (role && ROLE_NET[role]) {
        name = ROLE_NET[role];
        break;
      }
    }
    if (!name) name = `net${auto++}`;
    nets.push({
      name,
      pins: pins.map((p) => ({ component_ref: p.nodeId, pin_number: p.pinNo })),
    });
  }

  const components: CircuitGraphComponent[] = nodes.map((n) => {
    const pins: Record<number, string> = {};
    n.data.ports.forEach((p, i) => {
      pins[i + 1] = p.name;
    });
    return {
      ref: n.id,
      component_type: n.data.ctype,
      value: n.data.value ?? "",
      pins,
    };
  });

  return { title, components, nets };
}

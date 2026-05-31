/**
 * Canvas presets — pre-built, pre-wired circuits the user can drop onto the
 * schematic editor in one click. Each preset fetches the real SPICE ports for
 * its library parts (so pin handles match), then returns React Flow nodes +
 * edges ready for schematicStore.loadGraph().
 *
 * The breadboard PoC mirrors presets/lifi_poc_breadboard.json's amp_slicer
 * topology: PV cell -> R_sense -> instrumentation amp -> comparator slicer,
 * single-supply, no BPF. Its descriptive parts (generic 5mm LED, TL072, small
 * poly-Si panel) are mapped to the nearest wireable library equivalents.
 */

import type { Edge, Node } from "@xyflow/react";

import { api, type SpicePort } from "@/api/client";
import { symbolFor } from "@/features/schematic/symbols";
import type { PartNodeData } from "@/store/schematicStore";

export interface CanvasPreset {
  key: string;
  label: string;
  description: string;
  build: () => Promise<{ nodes: Node<PartNodeData>[]; edges: Edge[] }>;
}

async function ports(part: string): Promise<SpicePort[]> {
  const detail = await api.getComponent(part);
  return detail.spice_ports ?? [];
}

function node(
  id: string,
  part: string,
  ctype: string,
  label: string,
  partCode: string,
  category: string,
  ps: SpicePort[],
  x: number,
  y: number,
  value?: string,
): Node<PartNodeData> {
  return {
    id,
    type: "part",
    position: { x, y },
    data: { part, ctype, label, partCode, symbol: symbolFor(category, ctype), ports: ps, value },
  };
}

// Connect (sourceNode.sourcePort) -> (targetNode.targetPort) as a React Flow edge.
function wire(src: string, sp: string, tgt: string, tp: string): Edge {
  return {
    id: `${src}.${sp}-${tgt}.${tp}`,
    source: src,
    sourceHandle: sp,
    target: tgt,
    targetHandle: tp,
    type: "smoothstep",
    style: { stroke: "#94a3b8", strokeWidth: 1.6 },
  };
}

const BREADBOARD: CanvasPreset = {
  key: "breadboard_poc",
  label: "Breadboard PoC",
  description: "PV → sense R → instrumentation amp → comparator (amp-slicer RX).",
  build: async () => {
    // Library equivalents of the breadboard parts:
    //   small poly-Si panel  -> SM141K (silicon solar cell)
    //   TL072 gain stage     -> INA322 (single-supply, REF-referenced)
    //   data slicer          -> TLV7011 comparator
    const [pvP, ampP, cmpP] = await Promise.all([
      ports("SM141K"),
      ports("INA322"),
      ports("TLV7011"),
    ]);

    const nodes: Node<PartNodeData>[] = [
      node("Xsc", "SM141K", "SM141K", "Solar Cell", "SM141K", "Solar Cell", pvP, 40, 140),
      node("Rsense", "R", "R", "Resistor", "R", "", [
        { name: "1", role: "signal" },
        { name: "2", role: "signal_out" },
      ], 230, 200, "10k"),
      node("Vgnd_ref", "V", "V", "Ground ref", "V", "", [
        { name: "1", role: "signal" },
        { name: "2", role: "signal_out" },
      ], 230, 300, "0"),
      node("Xina", "INA322", "INA322", "Instrumentation Amp", "INA322", "Amplifier", ampP, 420, 120),
      node("Xcmp", "TLV7011", "TLV7011", "Comparator", "TLV7011", "Comparator", cmpP, 640, 120),
      node("GND1", "GND", "GND", "Ground", "0", "", [{ name: "gnd", role: "ground" }], 230, 380),
      node("OUT1", "OUT", "OUT", "Output", "dout", "", [{ name: "out", role: "output" }], 840, 140),
    ];

    const edges: Edge[] = [
      // PV cathode -> sense R and amp inverting input
      wire("Xsc", "cathode", "Rsense", "1"),
      wire("Rsense", "2", "Vgnd_ref", "1"),
      // ground reference source returns to ground net 0
      wire("Vgnd_ref", "2", "GND1", "gnd"),
      // amp: non-inverting from sense_lo, inverting from PV cathode
      wire("Rsense", "2", "Xina", "INP"),
      wire("Xsc", "cathode", "Xina", "INN"),
      // amp out -> comparator input
      wire("Xina", "OUT", "Xcmp", "INP"),
      // shared rails: tie comparator supplies to the amp's (one vcc/vee net each)
      wire("Xina", "VCC", "Xcmp", "VCC"),
      wire("Xina", "VEE", "Xcmp", "VEE"),
      // comparator threshold (INN) referenced to the amp's REF (vref) rail
      wire("Xina", "REF", "Xcmp", "INN"),
      // comparator out -> output terminal (net 'dout', read as V(dout))
      wire("Xcmp", "OUT", "OUT1", "out"),
    ];

    return { nodes, edges };
  },
};

// Generic marker / primitive ports (not registry parts).
const P = {
  drive: [{ name: "out", role: "signal_out" }] as SpicePort[],
  channel: [
    { name: "tx", role: "optical_in" },
    { name: "rx", role: "optical_out" },
  ] as SpicePort[],
  twoTerm: [
    { name: "1", role: "signal" },
    { name: "2", role: "signal_out" },
  ] as SpicePort[],
  gnd: [{ name: "gnd", role: "ground" }] as SpicePort[],
  out: [{ name: "out", role: "output" }] as SpicePort[],
  vcc: [{ name: "v", role: "supply_pos" }] as SpicePort[],
  vee: [{ name: "v", role: "supply_neg" }] as SpicePort[],
  vref: [{ name: "v", role: "ref" }] as SpicePort[],
};

// Power-rail flag node (VCC / VEE / VREF / GND). Like a real schematic, each
// supply pin gets a local rail tag instead of a wire dragged across the canvas;
// every flag of the same kind resolves to the same SPICE net by role.
function rail(id: string, ctype: "VCC" | "VEE" | "VREF" | "GND", x: number, y: number): Node<PartNodeData> {
  const ps = ctype === "GND" ? P.gnd : ctype === "VCC" ? P.vcc : ctype === "VEE" ? P.vee : P.vref;
  const label = ctype === "GND" ? "Ground" : ctype;
  const code = ctype === "GND" ? "0" : ctype.toLowerCase();
  return node(id, ctype, ctype, label, code, "", ps, x, y);
}

// Full single-canvas OOK link: TX analog (Data Source -> MOSFET -> LED) ->
// optical Channel -> RX analog (PV -> sense R -> instrumentation amp ->
// average-value comparator slicer). This is the milestone-1 reference design
// the two-pass co-simulator closes to BER 0 (drawn TX SPICE -> Python channel
// -> drawn RX SPICE, comparator demodulating in-circuit).
const OOK_LINK: CanvasPreset = {
  key: "ook_link",
  label: "OOK Link (TX → Channel → RX)",
  description: "Full in-circuit OOK link: MOSFET-driven LED → optical channel → PV + amp + comparator slicer.",
  build: async () => {
    const [ledP, mosP, pvP, ampP, cmpP] = await Promise.all([
      ports("LXM5_PD01"), ports("BSD235N"), ports("SM141K"),
      ports("INA322"), ports("TLV7011"),
    ]);

    const nodes: Node<PartNodeData>[] = [
      // --- TX lane ---
      rail("VCC1", "VCC", 210, 20),
      node("Rlimit", "R", "R", "Limit R", "R", "", P.twoTerm, 190, 90, "150"),
      node("Xled", "LXM5_PD01", "LXM5_PD01", "LED", "LXM5_PD01", "LED", ledP, 190, 220),
      node("Vdrv", "DRIVE", "DRIVE", "Data Source", "OOK", "", P.drive, 20, 360),
      node("Xmos", "BSD235N", "BSD235N", "Driver MOSFET", "BSD235N", "MOSFET", mosP, 210, 360),
      rail("GND1", "GND", 230, 490),
      // --- channel ---
      node("Ch1", "CHANNEL", "CHANNEL", "Channel", "optical", "", P.channel, 400, 300),
      // --- RX lane ---
      node("Xsc", "SM141K", "SM141K", "Solar Cell", "SM141K", "Solar Cell", pvP, 560, 300),
      rail("GND2", "GND", 570, 450),
      node("Rsense", "R", "R", "Sense R", "R", "", P.twoTerm, 700, 430, "10k"),
      node("Vgnd", "V", "V", "Ground ref", "V", "", P.twoTerm, 700, 540, "0"),
      rail("GND3", "GND", 850, 560),
      rail("VREF1", "VREF", 640, 190),
      rail("VCC2", "VCC", 860, 180),
      node("Xina", "INA322", "INA322", "Instrumentation Amp", "INA322", "Amplifier", ampP, 840, 300),
      rail("VEE1", "VEE", 860, 430),
      node("Ravg", "R", "R", "Avg R", "R", "", P.twoTerm, 1010, 200, "1Meg"),
      node("Cavg", "C", "C", "Avg C", "C", "", P.twoTerm, 1130, 210, "2n"),
      rail("VREF2", "VREF", 1190, 130),
      rail("VCC3", "VCC", 1010, 330),
      node("Xcmp", "TLV7011", "TLV7011", "Comparator", "TLV7011", "Comparator", cmpP, 1000, 410),
      rail("VEE2", "VEE", 1130, 470),
      node("OUT1", "OUT", "OUT", "Output", "dout", "", P.out, 1210, 410),
    ];

    const edges: Edge[] = [
      // --- TX: Data Source -> MOSFET -> LED -> Limit R -> VCC ---
      wire("Vdrv", "out", "Xmos", "gate"),
      wire("Xled", "cathode", "Xmos", "drain"),
      wire("Rlimit", "2", "Xled", "anode"),
      wire("Rlimit", "1", "VCC1", "v"),
      wire("Xmos", "source", "GND1", "gnd"),
      // --- channel: optical link into the PV ---
      wire("Ch1", "rx", "Xsc", "photo_in"),
      // --- RX front end: PV -> sense R -> instrumentation amp ---
      wire("Xsc", "anode", "GND2", "gnd"),
      wire("Xsc", "cathode", "Rsense", "1"),
      wire("Xsc", "cathode", "Xina", "INN"),
      wire("Rsense", "2", "Vgnd", "1"),
      wire("Rsense", "2", "Xina", "INP"),
      wire("Vgnd", "2", "GND3", "gnd"),
      wire("Xina", "VCC", "VCC2", "v"),
      wire("Xina", "VEE", "VEE1", "v"),
      wire("Xina", "REF", "VREF1", "v"),
      // --- average-value slicer: amp out -> RC threshold -> comparator ---
      wire("Xina", "OUT", "Ravg", "1"),
      wire("Xina", "OUT", "Xcmp", "INP"),
      wire("Ravg", "2", "Xcmp", "INN"),
      wire("Cavg", "1", "Xcmp", "INN"),
      wire("Cavg", "2", "VREF2", "v"),
      wire("Xcmp", "VCC", "VCC3", "v"),
      wire("Xcmp", "VEE", "VEE2", "v"),
      wire("Xcmp", "OUT", "OUT1", "out"),
    ];

    return { nodes, edges };
  },
};

// OFDM/BFSK link: a fast photodiode front-end (NOT the energy-harvesting solar
// cell — its 798 nF cap is far too slow). The transmitter is a linear driver
// (modelled in Python) and the receiver is a passive transimpedance node
// (photodiode -> sense R) sampled by the Python DSP. Closes OFDM via a
// circuit-measured per-subcarrier equaliser; pick OFDM or BFSK in Run settings.
const ANALOG_LINK: CanvasPreset = {
  key: "analog_link",
  label: "OFDM / BFSK Link (fast photodiode RX)",
  description: "Linear-driver TX → optical channel → fast photodiode + sense R, DSP-demodulated. Select OFDM or BFSK in Run settings.",
  build: async () => {
    const [ledP, mosP, pdP] = await Promise.all([
      ports("LXM5_PD01"), ports("BSD235N"), ports("BPW34"),
    ]);

    const nodes: Node<PartNodeData>[] = [
      // --- TX lane (linear driver modelled in Python; drawn for the link) ---
      rail("VCC1", "VCC", 210, 20),
      node("Rlimit", "R", "R", "Limit R", "R", "", P.twoTerm, 190, 90, "150"),
      node("Xled", "LXM5_PD01", "LXM5_PD01", "LED", "LXM5_PD01", "LED", ledP, 190, 220),
      node("Vdrv", "DRIVE", "DRIVE", "Data Source", "OFDM", "", P.drive, 20, 360),
      node("Xmos", "BSD235N", "BSD235N", "Driver MOSFET", "BSD235N", "MOSFET", mosP, 210, 360),
      rail("GND1", "GND", 230, 490),
      // --- channel ---
      node("Ch1", "CHANNEL", "CHANNEL", "Channel", "optical", "", P.channel, 400, 300),
      // --- RX lane: fast photodiode -> sense R -> sampled output ---
      node("Xpd", "BPW34", "BPW34", "Photodiode", "BPW34", "Photodiode", pdP, 560, 280),
      node("Rsense", "R", "R", "Sense R", "R", "", P.twoTerm, 720, 360, "22k"),
      node("OUT1", "OUT", "OUT", "Output", "dout", "", P.out, 760, 250),
      // The MCU (ESP32) is the digital baseband: its ADC samples the analog
      // signal and its firmware demodulates (the Python DSP). Its ADC pin marks
      // where the link is read.
      node("Mcu1", "MCU", "MCU", "MCU (ESP32)", "demod", "", [
        { name: "adc", role: "signal_in" },
        { name: "gpio", role: "signal_out" },
      ], 920, 270),
      rail("GND2", "GND", 600, 440),
    ];

    const edges: Edge[] = [
      // TX
      wire("Vdrv", "out", "Xmos", "gate"),
      wire("Xled", "cathode", "Xmos", "drain"),
      wire("Rlimit", "2", "Xled", "anode"),
      wire("Rlimit", "1", "VCC1", "v"),
      wire("Xmos", "source", "GND1", "gnd"),
      // channel -> photodiode
      wire("Ch1", "rx", "Xpd", "photo_in"),
      // RX: photocurrent through the sense R develops the analog signal
      wire("Xpd", "anode", "GND2", "gnd"),
      wire("Xpd", "cathode", "Rsense", "1"),
      wire("Xpd", "cathode", "OUT1", "out"),
      wire("Xpd", "cathode", "Mcu1", "adc"),   // the MCU samples the link here
      wire("Rsense", "2", "GND2", "gnd"),
    ];

    return { nodes, edges };
  },
};

export const CANVAS_PRESETS: CanvasPreset[] = [OOK_LINK, ANALOG_LINK, BREADBOARD];

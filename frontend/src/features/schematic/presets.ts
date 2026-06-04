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
  /** When set, loading the preset also selects this Run-settings modulation. */
  modulation?: string;
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

// Fully-faithful breadboard PoC: the user's exact netlist as a drawn circuit.
// TX: ESP32 GPIO -> R1 -> 2N2222 low-side switch -> R_LED -> 5mm white LED -> +5V.
// RX (split +/-5V via the ICL7660 = the VEE rail): small PV panel -> R3 load ->
// C1 AC-couple -> R4 bias to VMID(=vref) -> TL072 unity buffer -> TL072 G=23
// (R5 to VMID, R6 feedback) -> C2 -> R9 -> BZX84C3V3 ADC clamp -> ESP32 ADC.
// PWM-ASK, demodulated by the MCU's firmware (Python DSP). Closes BER 0 through
// the two-pass SPICE co-sim (run_pwm_ask_link). Note: D2 (Zener) is drawn as a
// correct clamp (cathode->ADC node) -- the original netlist's anode->ADC would
// forward-conduct at the 1.65V ADC bias; VMID is modelled as the stiff vref
// rail (a hardware build wants a VMID bypass cap to actually reach G=23).
const BREADBOARD_POC: CanvasPreset = {
  key: "breadboard_poc_faithful",
  label: "Breadboard PoC (faithful, PWM-ASK)",
  description:
    "Exact netlist: ESP32+2N2222+5mm LED TX, small PV + TL072 x2 (+/-5V) RX, ESP32 ADC. PWM-ASK.",
  modulation: "PWM_ASK",
  build: async () => {
    const [ledP, bjtP, pvP, opP, znP] = await Promise.all([
      ports("LED5MM_WHITE"),
      ports("BJT_2N2222"),
      ports("PV_PANEL_5V1W"),
      ports("TL072"),
      ports("BZX84C3V3"),
    ]);

    // Compact, rails-adjacent layout. Each VCC/VREF flag sits directly above
    // the pin it feeds (its pin points down); each VEE/GND flag sits directly
    // below (pin points up) -> short vertical rail stubs, no long crossings.
    // TX is a vertical common-emitter column; RX flows left->right.
    const nodes: Node<PartNodeData>[] = [
      // --- TX: common-emitter LED driver (VCC -> LED -> R_LED -> BJT -> GND) ---
      rail("VccLED", "VCC", 250, 40),
      node("Xled", "LED5MM_WHITE", "LED5MM_WHITE", "5mm white LED", "LED5MM_WHITE", "LED", ledP, 230, 110),
      node("Rled", "R", "R", "R_LED", "R", "", P.twoTerm, 230, 190, "220"),
      node("Xq1", "BJT_2N2222", "BJT_2N2222", "2N2222", "BJT_2N2222", "BJT", bjtP, 250, 270),
      rail("GndE", "GND", 262, 380),
      node("R1", "R", "R", "R1", "R", "", P.twoTerm, 110, 285, "1k"),
      node("Vdrv", "DRIVE", "DRIVE", "ESP32 GPIO25", "PWM_ASK", "", P.drive, 0, 280),
      // --- RX front end: PV -> R3 load + C1 couple -> R4 bias ---
      node("Xpv", "PV_PANEL_5V1W", "PV_PANEL_5V1W", "PV panel", "PV_PANEL_5V1W", "Solar Cell", pvP, 430, 270),
      node("R3", "R", "R", "R3 load", "R", "", P.twoTerm, 430, 380, "1k"),
      rail("GndPV", "GND", 442, 470),
      node("C1", "C", "C", "C1 couple", "C", "", P.twoTerm, 560, 280),
      node("R4", "R", "R", "R4 bias", "R", "", P.twoTerm, 560, 185, "1Meg"),
      rail("VrefR4", "VREF", 622, 110),
      // --- TL072 U1a: unity buffer (VCC/VEE rails aligned to the pin x: opX+10) ---
      node("Xu1a", "TL072", "TL072", "TL072 U1a (buffer)", "TL072", "Amplifier", opP, 690, 270),
      rail("VccA", "VCC", 700, 185),
      rail("VeeA", "VEE", 700, 370),
      // --- TL072 U1b: non-inverting G=23 (R5 to VMID, R6 feedback) ---
      node("Xu1b", "TL072", "TL072", "TL072 U1b (G=23)", "TL072", "Amplifier", opP, 890, 270),
      rail("VccB", "VCC", 900, 185),
      rail("VeeB", "VEE", 900, 370),
      node("R6", "R", "R", "R6 fb", "R", "", P.twoTerm, 880, 175, "22k"),
      node("R5", "R", "R", "R5", "R", "", P.twoTerm, 790, 110, "1k"),
      rail("VrefR5", "VREF", 852, 40),
      // --- output coupling + VMID re-bias + ADC protection ---
      node("C2", "C", "C", "C2 couple", "C", "", P.twoTerm, 1030, 280),
      node("Rbias", "R", "R", "R_VMID", "R", "", P.twoTerm, 1030, 185, "100k"),
      rail("VrefRb", "VREF", 1092, 110),
      node("R9", "R", "R", "R9", "R", "", P.twoTerm, 1150, 280, "1k"),
      node("Xd2", "BZX84C3V3", "BZX84C3V3", "BZX84C3V3 clamp", "BZX84C3V3", "Zener", znP, 1150, 380),
      rail("GndZ", "GND", 1162, 470),
      node("Mcu", "MCU", "MCU", "ESP32 ADC (GPIO34)", "demod", "", [
        { name: "adc", role: "signal_in" },
      ], 1280, 270),
    ];

    const edges: Edge[] = [
      // --- TX ---
      wire("Vdrv", "out", "R1", "1"),
      wire("R1", "2", "Xq1", "base"),
      wire("Xq1", "collector", "Rled", "2"),
      wire("Rled", "1", "Xled", "cathode"),
      wire("Xled", "anode", "VccLED", "v"),
      wire("Xq1", "emitter", "GndE", "gnd"),
      // --- RX front end: PV(+)=anode -> R3 load + C1 couple; cathode to GND ---
      wire("Xpv", "anode", "R3", "1"),
      wire("Xpv", "anode", "C1", "1"),
      wire("Xpv", "cathode", "GndPV", "gnd"),
      wire("R3", "2", "GndPV", "gnd"),
      wire("C1", "2", "R4", "1"),
      wire("C1", "2", "Xu1a", "INP"),
      wire("R4", "2", "VrefR4", "v"),
      // --- stage 1 unity buffer (INN tied to OUT) ---
      wire("Xu1a", "OUT", "Xu1a", "INN"),
      wire("Xu1a", "OUT", "Xu1b", "INP"),
      wire("Xu1a", "VCC", "VccA", "v"),
      wire("Xu1a", "VEE", "VeeA", "v"),
      // --- stage 2 non-inverting G=23 (R5 to VMID, R6 feedback) ---
      wire("Xu1b", "INN", "R5", "1"),
      wire("Xu1b", "INN", "R6", "1"),
      wire("R5", "2", "VrefR5", "v"),
      wire("Xu1b", "OUT", "R6", "2"),
      wire("Xu1b", "OUT", "C2", "1"),
      wire("Xu1b", "VCC", "VccB", "v"),
      wire("Xu1b", "VEE", "VeeB", "v"),
      // --- output coupling + VMID re-bias + ADC protection ---
      wire("C2", "2", "Rbias", "1"),
      wire("C2", "2", "R9", "1"),
      wire("Rbias", "2", "VrefRb", "v"),
      wire("R9", "2", "Xd2", "cathode"),   // flipped clamp: cathode -> ADC node
      wire("R9", "2", "Mcu", "adc"),
      wire("Xd2", "anode", "GndZ", "gnd"),
    ];

    return { nodes, edges };
  },
};

export const CANVAS_PRESETS: CanvasPreset[] = [
  OOK_LINK,
  ANALOG_LINK,
  BREADBOARD,
  BREADBOARD_POC,
];

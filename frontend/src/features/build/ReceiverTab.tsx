/**
 * ReceiverTab — PV cell, sense, amplifier, BPF, comparator, DC-DC. The
 * `rx_topology` field drives which sections appear:
 *   - ina_bpf_comp: full chain (PV → R_sense → INA → BPF → Comparator → DC-DC)
 *   - amp_slicer: PV → R_sense → simple gain → slicer
 *   - direct: PV → R_sense → output (no analog chain)
 *
 * Mirrors gui/tab_system_setup.py "Receiver" QGroupBox plus advanced
 * settings sourced from cosim/system_config.py RXConfig.
 */

import { AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";

import { api, type ComponentSummary } from "@/api/client";
import { NumberField } from "@/primitives/NumberField";
import { Select } from "@/primitives/Select";

import { Section } from "@/features/build/Section";
import { useDraftField } from "@/features/build/useDraftField";

const TOPOLOGIES = [
  { value: "ina_bpf_comp", label: "INA + BPF + Comparator (full chain)" },
  { value: "amp_slicer", label: "Amp + Slicer" },
  { value: "direct", label: "Direct (no analog chain)" },
  { value: "auto", label: "Auto-detect from other fields" },
];

export function ReceiverTab() {
  const [comps, setComps] = useState<ComponentSummary[]>([]);
  useEffect(() => {
    api.listComponents().then(setComps).catch(() => setComps([]));
  }, []);

  const topology = useDraftField<string>("rx_topology");
  const pvPart = useDraftField<string>("pv_part");
  const scArea = useDraftField<number>("sc_area_cm2");
  const scResp = useDraftField<number>("sc_responsivity");
  const rSense = useDraftField<number>("r_sense_ohm");
  const vcc = useDraftField<number>("vcc_volts");

  const inaPart = useDraftField<string>("ina_part");
  const inaGain = useDraftField<number>("ina_gain_dB");
  const inaGbw = useDraftField<number>("ina_gbw_kHz");

  const bpfStages = useDraftField<number>("bpf_stages");
  const bpfLow = useDraftField<number>("bpf_f_low_Hz");
  const bpfHigh = useDraftField<number>("bpf_f_high_Hz");

  const compPart = useDraftField<string>("comparator_part");

  const dcdcEnable = useDraftField<boolean>("dcdc_enable");
  const dcdcFsw = useDraftField<number>("dcdc_fsw_kHz");
  const dcdcL = useDraftField<number>("dcdc_l_uH");
  const rLoad = useDraftField<number>("r_load_ohm");

  const pvOptions = comps.filter((c) => c.config_field === "pv_part").map((c) => c.part);
  const inaOptions = comps.filter((c) => c.config_field === "ina_part").map((c) => c.part);
  const compOptions = comps.filter((c) => c.config_field === "comparator_part").map((c) => c.part);

  if (pvPart.value && !pvOptions.includes(pvPart.value)) pvOptions.unshift(pvPart.value);
  if (inaPart.value && !inaOptions.includes(inaPart.value)) inaOptions.unshift(inaPart.value);
  if (compPart.value && !compOptions.includes(compPart.value)) compOptions.unshift(compPart.value);

  const showAnalog = topology.value === "ina_bpf_comp" || topology.value === "amp_slicer" || topology.value === "auto";
  const showBpf = topology.value === "ina_bpf_comp" || topology.value === "auto";
  const showComparator = topology.value === "ina_bpf_comp" || topology.value === "amp_slicer" || topology.value === "auto";

  return (
    <div className="space-y-5">
      <Section title="Topology" description="Which receiver chain to simulate. Choose 'Auto' to derive from the other fields.">
        <Select label="RX topology"
          value={topology.value}
          onChange={topology.set}
          options={TOPOLOGIES} />
        <NumberField label="V_cc supply" unit="V" step={0.1} min={0.5}
          value={vcc.value} onChange={vcc.set} />
      </Section>

      <Section title="Photovoltaic cell">
        <Select label="PV part"
          value={pvPart.value}
          onChange={pvPart.set}
          options={pvOptions.length ? pvOptions : [pvPart.value ?? ""]} />
        <NumberField label="Active area" unit="cm²" step={0.5} min={0.01}
          value={scArea.value} onChange={scArea.set} />
        <NumberField label="Responsivity" unit="A/W" step={0.01} min={0}
          value={scResp.value} onChange={scResp.set} />
        <NumberField label="R_sense" unit="Ω" step={0.1} min={0.01}
          value={rSense.value} onChange={rSense.set}
          hint="Current-sense resistor" />
      </Section>

      <AnimatePresence>
        {showAnalog && (
          <Section title="Amplifier">
            <Select label="INA / op-amp part"
              value={inaPart.value}
              onChange={inaPart.set}
              options={inaOptions.length ? inaOptions : [inaPart.value ?? ""]} />
            <NumberField label="Gain" unit="dB" step={1}
              value={inaGain.value} onChange={inaGain.set} />
            <NumberField label="GBW" unit="kHz" step={10} min={0}
              value={inaGbw.value} onChange={inaGbw.set} />
          </Section>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showBpf && (
          <Section title="Band-pass filter" description="One stage per cascaded BPF section. Set to 0 to bypass.">
            <NumberField label="# stages" step={1} min={0} max={4}
              value={bpfStages.value} onChange={bpfStages.set} />
            <NumberField label="f_low" unit="Hz" step={50} min={0}
              value={bpfLow.value} onChange={bpfLow.set}
              disabled={!bpfStages.value} />
            <NumberField label="f_high" unit="Hz" step={500} min={0}
              value={bpfHigh.value} onChange={bpfHigh.set}
              disabled={!bpfStages.value} />
          </Section>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showComparator && (
          <Section title="Comparator / slicer">
            <Select label="Comparator part"
              value={compPart.value}
              onChange={compPart.set}
              options={compOptions.length ? compOptions : [compPart.value ?? ""]} />
          </Section>
        )}
      </AnimatePresence>

      <Section title="DC-DC converter (energy harvest)" description="Boost converter on the PV anode side. Disable for comms-only systems.">
        <div className="sm:col-span-3">
          <label className="mb-2 inline-flex items-center gap-2 text-sm text-slate-200">
            <input
              type="checkbox"
              checked={dcdcEnable.value ?? false}
              onChange={(e) => dcdcEnable.set(e.target.checked)}
              className="h-4 w-4 rounded border-white/20 bg-white/[0.06]"
            />
            DC-DC enabled
          </label>
        </div>
        <NumberField label="Switching freq" unit="kHz" step={10} min={0}
          disabled={!dcdcEnable.value}
          value={dcdcFsw.value} onChange={dcdcFsw.set} />
        <NumberField label="Inductor L" unit="µH" step={1} min={0}
          disabled={!dcdcEnable.value}
          value={dcdcL.value} onChange={dcdcL.set} />
        <NumberField label="Load R" unit="Ω" step={1000} min={0}
          value={rLoad.value} onChange={rLoad.set} />
      </Section>
    </div>
  );
}

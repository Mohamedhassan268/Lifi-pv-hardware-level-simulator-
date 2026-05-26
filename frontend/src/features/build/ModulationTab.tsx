/**
 * ModulationTab — modulation scheme + simulation timing + per-scheme params.
 * Scheme-specific sub-section appears only when its modulation is selected.
 */

import { AnimatePresence } from "framer-motion";

import { NumberField } from "@/primitives/NumberField";
import { Select } from "@/primitives/Select";

import { Section } from "@/features/build/Section";
import { useDraftField } from "@/features/build/useDraftField";

const MODULATIONS = ["OOK", "OOK_Manchester", "OFDM", "BFSK", "PWM_ASK"];
const ENGINES = [
  { value: "python", label: "Python (fast, no SPICE)" },
  { value: "spice", label: "SPICE (LTspice / ngspice — falls back to Python if unavailable)" },
];

export function ModulationTab() {
  const modulation = useDraftField<string>("modulation");
  const dataRate = useDraftField<number>("data_rate_bps");
  const nBits = useDraftField<number>("n_bits");
  const tStop = useDraftField<number>("t_stop_s");
  const prbs = useDraftField<number>("prbs_order");
  const engine = useDraftField<string>("simulation_engine");
  const seed = useDraftField<number | null>("random_seed");

  // OFDM
  const ofdmNfft = useDraftField<number>("ofdm_nfft");
  const ofdmQam = useDraftField<number>("ofdm_qam_order");
  const ofdmSub = useDraftField<number>("ofdm_n_subcarriers");
  const ofdmCp = useDraftField<number>("ofdm_cp_len");
  // BFSK
  const bfskF0 = useDraftField<number>("bfsk_f0_hz");
  const bfskF1 = useDraftField<number>("bfsk_f1_hz");
  // PWM-ASK
  const pwmFreq = useDraftField<number>("pwm_freq_hz");
  const carrierFreq = useDraftField<number>("carrier_freq_hz");

  return (
    <div className="space-y-5">
      <Section title="Modulation">
        <Select label="Scheme"
          value={modulation.value}
          onChange={modulation.set}
          options={MODULATIONS} />
        <NumberField label="Data rate" unit="bps" step={100} min={1}
          value={dataRate.value} onChange={dataRate.set} />
        <NumberField label="# bits to transmit" step={10} min={4}
          value={nBits.value} onChange={nBits.set}
          hint="More = better BER statistics, slower sim" />
        <NumberField label="t_stop" unit="s" step={1e-4} min={1e-6}
          value={tStop.value} onChange={tStop.set} />
        <NumberField label="PRBS order" step={1} min={3} max={15}
          value={prbs.value} onChange={prbs.set} />
      </Section>

      <Section title="Engine & RNG">
        <Select label="Simulation engine"
          value={engine.value}
          onChange={engine.set}
          options={ENGINES} />
        <NumberField label="Random seed" step={1}
          value={seed.value ?? undefined}
          onChange={(v) => seed.set(v)}
          hint="Leave blank for non-deterministic" />
      </Section>

      <AnimatePresence>
        {modulation.value === "OFDM" && (
          <Section title="OFDM parameters">
            <NumberField label="N_FFT" step={64} min={64}
              value={ofdmNfft.value} onChange={ofdmNfft.set} />
            <NumberField label="QAM order" step={1} min={2}
              value={ofdmQam.value} onChange={ofdmQam.set}
              hint="4 / 16 / 64" />
            <NumberField label="# subcarriers" step={4} min={4}
              value={ofdmSub.value} onChange={ofdmSub.set} />
            <NumberField label="CP length" step={4} min={0}
              value={ofdmCp.value} onChange={ofdmCp.set} />
          </Section>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {modulation.value === "BFSK" && (
          <Section title="BFSK parameters">
            <NumberField label="f_0" unit="Hz" step={100}
              value={bfskF0.value} onChange={bfskF0.set} />
            <NumberField label="f_1" unit="Hz" step={100}
              value={bfskF1.value} onChange={bfskF1.set} />
          </Section>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {modulation.value === "PWM_ASK" && (
          <Section title="PWM-ASK parameters">
            <NumberField label="PWM frequency" unit="Hz" step={1}
              value={pwmFreq.value} onChange={pwmFreq.set} />
            <NumberField label="Carrier frequency" unit="Hz" step={100}
              value={carrierFreq.value} onChange={carrierFreq.set} />
          </Section>
        )}
      </AnimatePresence>
    </div>
  );
}

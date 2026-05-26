/**
 * NoiseInspector — worked-example inspector content for the 6-source physical
 * noise model. Structured one accordion per source, matching cosim/noise.py's
 * NoiseModel ordering. Field names are the verbatim JSON keys from
 * cosim/system_config.py's NoiseConfig + adjacent RX/Channel/Sim configs so
 * engineers can cross-reference against the Python source.
 *
 * Source 1 (Shot)        pv_dark_current_A, pv_dark_current_ref_T_K, dark_current_doubling_dT_K
 * Source 2 (Thermal)     r_sense_ohm, temperature_K
 * Source 3 (Ambient)     ambient_illuminance_lux
 * Source 4 (Amplifier)   ina_noise_nV_rtHz, ina_noise_current_pA_rtHz
 * Source 5 (ADC)         adc_bits, adc_vref
 * Source 6 (Processing)  comparator_offset_mV, comparator_jitter_ns
 */

import { InspectorSection } from "@/features/builder/inspectors/InspectorSection";
import { useDraftField } from "@/features/build/useDraftField";
import { NumberField } from "@/primitives/NumberField";
import { Toggle } from "@/primitives/Toggle";

export function NoiseInspector() {
  const master = useDraftField<boolean>("noise_enable");
  const shot = useDraftField<boolean>("noise_shot_enable");
  const thermal = useDraftField<boolean>("noise_thermal_enable");
  const ambient = useDraftField<boolean>("noise_ambient_enable");
  const amp = useDraftField<boolean>("noise_amplifier_enable");
  const adc = useDraftField<boolean>("noise_adc_enable");
  const processing = useDraftField<boolean>("noise_processing_enable");

  const temperatureK = useDraftField<number>("temperature_K");
  const dataRate = useDraftField<number>("data_rate_bps");
  const pvDarkCurrent = useDraftField<number>("pv_dark_current_A");
  const pvDarkRefT = useDraftField<number>("pv_dark_current_ref_T_K");
  const darkDoubling = useDraftField<number>("dark_current_doubling_dT_K");
  const rSense = useDraftField<number>("r_sense_ohm");
  const ambientLux = useDraftField<number>("ambient_illuminance_lux");
  const inaNoiseV = useDraftField<number>("ina_noise_nV_rtHz");
  const inaNoiseI = useDraftField<number>("ina_noise_current_pA_rtHz");
  const adcBits = useDraftField<number>("adc_bits");
  const adcVref = useDraftField<number>("adc_vref");
  const compOffset = useDraftField<number>("comparator_offset_mV");
  const compJitter = useDraftField<number>("comparator_jitter_ns");

  const off = !master.value;

  return (
    <div className="space-y-2">
      {/* Global controls — not collapsible, always visible at the top. */}
      <div className="border-l-2 border-l-harvest-400/60 bg-slate-900/40">
        <header className="border-b border-hair px-2 py-1.5">
          <span className="label-inst">Global</span>
        </header>
        <div className="space-y-2 px-2 py-2">
          <Toggle
            label="noise_enable"
            value={master.value ?? false}
            onChange={master.set}
            hint="Master switch for the 6-source noise model"
          />
          <NumberField
            label="temperature_K"
            unit="K"
            step={1}
            min={273}
            max={373}
            value={temperatureK.value}
            onChange={temperatureK.set}
            hint="Affects shot & thermal terms"
          />
          <NumberField
            label="data_rate_bps"
            unit="bps"
            step={100}
            min={1}
            value={dataRate.value}
            onChange={dataRate.set}
            hint="Bandwidth = data_rate / 2"
          />
        </div>
      </div>

      <InspectorSection
        index="①"
        title="Shot noise"
        formula="2q·(I_ph + I_dark)·B_n"
        enabled={shot.value ?? true}
        onEnabledChange={shot.set}
        defaultOpen
      >
        <NumberField
          label="pv_dark_current_A"
          unit="A"
          step={1e-11}
          min={0}
          value={pvDarkCurrent.value}
          onChange={pvDarkCurrent.set}
          hint="Si PV typical 1e-10 A @ 300 K"
          disabled={off}
        />
        <NumberField
          label="pv_dark_current_ref_T_K"
          unit="K"
          step={1}
          min={0}
          value={pvDarkRefT.value}
          onChange={pvDarkRefT.set}
          disabled={off}
        />
        <NumberField
          label="dark_current_doubling_dT_K"
          unit="K"
          step={1}
          min={1}
          value={darkDoubling.value}
          onChange={darkDoubling.set}
          hint="Si = 10, GaAs ≈ 7"
          disabled={off}
        />
      </InspectorSection>

      <InspectorSection
        index="②"
        title="Thermal noise"
        formula="4kT·B_n / R_load"
        enabled={thermal.value ?? true}
        onEnabledChange={thermal.set}
      >
        <NumberField
          label="r_sense_ohm"
          unit="Ω"
          step={1}
          min={0.1}
          value={rSense.value}
          onChange={rSense.set}
          hint="Transimpedance load (1–1000 Ω typical)"
          disabled={off}
        />
      </InspectorSection>

      <InspectorSection
        index="③"
        title="Ambient light"
        formula="2q·I_ambient·B_n"
        enabled={ambient.value ?? false}
        onEnabledChange={ambient.set}
      >
        <NumberField
          label="ambient_illuminance_lux"
          unit="lux"
          step={50}
          min={0}
          value={ambientLux.value}
          onChange={ambientLux.set}
          hint="Indoor 100–500 · Outdoor 1e4+"
          disabled={off}
        />
      </InspectorSection>

      <InspectorSection
        index="④"
        title="Amplifier (INA322)"
        formula="e_n² + (i_n·Z_in)²"
        enabled={amp.value ?? true}
        onEnabledChange={amp.set}
      >
        <NumberField
          label="ina_noise_nV_rtHz"
          unit="nV/√Hz"
          step={1}
          min={0}
          value={inaNoiseV.value}
          onChange={inaNoiseV.set}
          hint="INA322 ≈ 45 · TL072 ≈ 15"
          disabled={off}
        />
        <NumberField
          label="ina_noise_current_pA_rtHz"
          unit="pA/√Hz"
          step={0.01}
          min={0}
          value={inaNoiseI.value}
          onChange={inaNoiseI.set}
          disabled={off}
        />
      </InspectorSection>

      <InspectorSection
        index="⑤"
        title="ADC quantization"
        formula="V_LSB² / 12"
        enabled={adc.value ?? false}
        onEnabledChange={adc.set}
      >
        <NumberField
          label="adc_bits"
          step={1}
          min={4}
          max={24}
          value={adcBits.value}
          onChange={adcBits.set}
          disabled={off}
        />
        <NumberField
          label="adc_vref"
          unit="V"
          step={0.1}
          min={0}
          value={adcVref.value}
          onChange={adcVref.set}
          disabled={off}
        />
      </InspectorSection>

      <InspectorSection
        index="⑥"
        title="Processing / threshold"
        formula="σ_offset² + σ_jitter²"
        enabled={processing.value ?? false}
        onEnabledChange={processing.set}
      >
        <NumberField
          label="comparator_offset_mV"
          unit="mV"
          step={0.1}
          min={0}
          value={compOffset.value}
          onChange={compOffset.set}
          hint="TLV7011 ≈ 1 mV"
          disabled={off}
        />
        <NumberField
          label="comparator_jitter_ns"
          unit="ns"
          step={0.5}
          min={0}
          value={compJitter.value}
          onChange={compJitter.set}
          hint="Comparator propagation jitter (1σ)"
          disabled={off}
        />
      </InspectorSection>
    </div>
  );
}

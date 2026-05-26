/**
 * NoiseTab — the 6-source noise model. All sub-fields disabled until the
 * master `noise_enable` toggle is on.
 */

import { NumberField } from "@/primitives/NumberField";
import { Toggle } from "@/primitives/Toggle";

import { Section } from "@/features/build/Section";
import { useDraftField } from "@/features/build/useDraftField";

export function NoiseTab() {
  const master = useDraftField<boolean>("noise_enable");
  const shot = useDraftField<boolean>("noise_shot_enable");
  const thermal = useDraftField<boolean>("noise_thermal_enable");
  const ambient = useDraftField<boolean>("noise_ambient_enable");
  const amp = useDraftField<boolean>("noise_amplifier_enable");
  const adc = useDraftField<boolean>("noise_adc_enable");
  const processing = useDraftField<boolean>("noise_processing_enable");

  const inaNoiseV = useDraftField<number>("ina_noise_nV_rtHz");
  const inaNoiseI = useDraftField<number>("ina_noise_current_pA_rtHz");
  const ambientLux = useDraftField<number>("ambient_illuminance_lux");
  const compOffset = useDraftField<number>("comparator_offset_mV");
  const compJitter = useDraftField<number>("comparator_jitter_ns");
  const adcBits = useDraftField<number>("adc_bits");
  const adcVref = useDraftField<number>("adc_vref");

  const off = !master.value;

  return (
    <div className="space-y-5">
      <Section title="Master switch" description="Turn the noise model on or off. Disabled = ideal noiseless simulation.">
        <div className="sm:col-span-3">
          <Toggle label="Noise model enabled"
            value={master.value ?? false}
            onChange={master.set}
            hint="6-source physical model (shot, thermal, ambient, amp, ADC, processing)" />
        </div>
      </Section>

      <Section title="Sources" description="Individually toggle the 6 noise sources.">
        <Toggle label="Shot noise"
          value={shot.value ?? false} onChange={shot.set} disabled={off}
          hint="Photocurrent statistics: 2q·I·Bn" />
        <Toggle label="Thermal noise"
          value={thermal.value ?? false} onChange={thermal.set} disabled={off}
          hint="Johnson-Nyquist: 4kT·Bn/R" />
        <Toggle label="Ambient light"
          value={ambient.value ?? false} onChange={ambient.set} disabled={off} />
        <Toggle label="Amplifier noise"
          value={amp.value ?? false} onChange={amp.set} disabled={off}
          hint="e_n² + (i_n·Z_in)²" />
        <Toggle label="ADC quantization"
          value={adc.value ?? false} onChange={adc.set} disabled={off}
          hint="V_LSB²/12" />
        <Toggle label="Processing / threshold"
          value={processing.value ?? false} onChange={processing.set} disabled={off}
          hint="Comparator offset + jitter" />
      </Section>

      <Section title="Amplifier noise spec" description="From the INA datasheet (e.g. INA322).">
        <NumberField label="e_n" unit="nV/√Hz" step={1} min={0}
          disabled={off || !amp.value}
          value={inaNoiseV.value} onChange={inaNoiseV.set} />
        <NumberField label="i_n" unit="pA/√Hz" step={0.01} min={0}
          disabled={off || !amp.value}
          value={inaNoiseI.value} onChange={inaNoiseI.set} />
      </Section>

      <Section title="Environment & ADC">
        <NumberField label="Ambient illuminance" unit="lux" step={50} min={0}
          disabled={off || !ambient.value}
          value={ambientLux.value} onChange={ambientLux.set} />
        <NumberField label="ADC bits" step={1} min={4} max={24}
          disabled={off || !adc.value}
          value={adcBits.value} onChange={adcBits.set} />
        <NumberField label="ADC V_ref" unit="V" step={0.1} min={0}
          disabled={off || !adc.value}
          value={adcVref.value} onChange={adcVref.set} />
      </Section>

      <Section title="Comparator">
        <NumberField label="Offset" unit="mV" step={0.1} min={0}
          disabled={off || !processing.value}
          value={compOffset.value} onChange={compOffset.set} />
        <NumberField label="Jitter" unit="ns" step={1} min={0}
          disabled={off || !processing.value}
          value={compJitter.value} onChange={compJitter.set} />
      </Section>
    </div>
  );
}

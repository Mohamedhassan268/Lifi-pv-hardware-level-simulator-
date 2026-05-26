/**
 * TransmitterTab — LED, driver, bias. Fields mirror the "Transmitter"
 * QGroupBox in gui/tab_system_setup.py (lines 138–184).
 *
 * LED/driver dropdowns hit /api/components filtered by config_field.
 * Picking a part writes only the part-name string into the draft —
 * same contract as today's Workbench. Datasheet auto-fill is deferred.
 */

import { useEffect, useState } from "react";

import { api, type ComponentSummary } from "@/api/client";
import { NumberField } from "@/primitives/NumberField";
import { Select } from "@/primitives/Select";

import { Section } from "@/features/build/Section";
import { useDraftField } from "@/features/build/useDraftField";

export function TransmitterTab() {
  const [allComponents, setAllComponents] = useState<ComponentSummary[]>([]);

  useEffect(() => {
    api.listComponents().then(setAllComponents).catch(() => setAllComponents([]));
  }, []);

  const ledPart = useDraftField<string>("led_part");
  const driverPart = useDraftField<string>("driver_part");
  const bias = useDraftField<number>("bias_current_A");
  const modDepth = useDraftField<number>("modulation_depth");
  const ledPower = useDraftField<number>("led_radiated_power_mW");
  const lensT = useDraftField<number>("lens_transmittance");

  const ledOptions = allComponents
    .filter((c) => c.config_field === "led_part")
    .map((c) => c.part);
  const driverOptions = allComponents
    .filter((c) => c.config_field === "driver_part")
    .map((c) => c.part);

  // Always include the current draft value as a fallback option so the user
  // can see what's selected even if the registry hasn't loaded yet.
  if (ledPart.value && !ledOptions.includes(ledPart.value))
    ledOptions.unshift(ledPart.value);
  if (driverPart.value && !driverOptions.includes(driverPart.value))
    driverOptions.unshift(driverPart.value);

  return (
    <div className="space-y-5">
      <Section title="LED" description="Pick a part from the component library; only the name is written. Tune electrical parameters below for custom parts.">
        <Select label="LED part"
          value={ledPart.value}
          onChange={ledPart.set}
          options={ledOptions.length ? ledOptions : [ledPart.value ?? ""]} />
        <NumberField label="Radiated power" unit="mW" step={0.1} min={0}
          value={ledPower.value} onChange={ledPower.set} />
        <NumberField label="Lens transmittance" step={0.05} min={0} max={1}
          value={lensT.value} onChange={lensT.set}
          hint="Fraen / similar lens" />
      </Section>

      <Section title="Driver & modulation depth">
        <Select label="Driver part"
          value={driverPart.value}
          onChange={driverPart.set}
          options={driverOptions.length ? driverOptions : [driverPart.value ?? ""]} />
        <NumberField label="Bias current" unit="A" step={0.01} min={0}
          value={bias.value} onChange={bias.set} />
        <NumberField label="Modulation depth" step={0.05} min={0} max={1}
          value={modDepth.value} onChange={modDepth.set}
          hint="AC swing / DC bias" />
      </Section>
    </div>
  );
}

/**
 * ChannelTab — geometry, FOV, environment, multipath. Field set mirrors the
 * "Channel" QGroupBox in gui/tab_system_setup.py (lines 186–225) plus the
 * less-commonly-tuned environment fields from cosim/system_config.py
 * ChannelConfig.
 */

import { NumberField } from "@/primitives/NumberField";
import { Toggle } from "@/primitives/Toggle";

import { Section } from "@/features/build/Section";
import { useDraftField } from "@/features/build/useDraftField";

export function ChannelTab() {
  const distance = useDraftField<number>("distance_m");
  const txAngle = useDraftField<number>("tx_angle_deg");
  const rxTilt = useDraftField<number>("rx_tilt_deg");
  const fov = useDraftField<number>("fov_half_angle_deg");
  const ledHalf = useDraftField<number>("led_half_angle_deg");
  const temperatureK = useDraftField<number>("temperature_K");
  const beerLambert = useDraftField<boolean>("beer_lambert_enabled");
  const humidity = useDraftField<number | null>("humidity_rh");
  const nReflections = useDraftField<number>("n_reflections");
  const roomL = useDraftField<number>("room_length_m");
  const roomW = useDraftField<number>("room_width_m");
  const roomH = useDraftField<number>("room_height_m");
  const wallR = useDraftField<number>("wall_reflectivity");

  return (
    <div className="space-y-5">
      <Section title="Geometry" description="TX→RX positioning. All angles in degrees, distances in metres.">
        <NumberField label="Distance" unit="m" step={0.01} min={0.01}
          value={distance.value} onChange={distance.set} />
        <NumberField label="TX angle" unit="°" step={1}
          value={txAngle.value} onChange={txAngle.set} />
        <NumberField label="RX tilt" unit="°" step={1}
          value={rxTilt.value} onChange={rxTilt.set} />
        <NumberField label="RX FOV half-angle" unit="°" step={1} min={1} max={90}
          value={fov.value} onChange={fov.set}
          hint="90° = hemispherical" />
        <NumberField label="LED half-angle" unit="°" step={1} min={1}
          value={ledHalf.value} onChange={ledHalf.set}
          hint="Lambertian order m = −ln2 / ln cos α" />
        <NumberField label="Ambient temperature" unit="K" step={1}
          value={temperatureK.value} onChange={temperatureK.set} />
      </Section>

      <Section title="Atmosphere" description="Optional Beer–Lambert attenuation. Leave disabled for dry indoor links.">
        <Toggle label="Beer–Lambert enabled"
          value={beerLambert.value ?? false}
          onChange={beerLambert.set} />
        <NumberField label="Humidity" unit="0–1 RH" step={0.05} min={0} max={1}
          disabled={!beerLambert.value}
          value={humidity.value ?? undefined}
          onChange={(v) => humidity.set(v)} />
      </Section>

      <Section title="Multipath" description="First-order diffuse wall reflections inside a rectangular room.">
        <NumberField label="# reflections" step={1} min={0} max={4}
          value={nReflections.value} onChange={nReflections.set}
          hint="0 = LOS only" />
        <NumberField label="Room length" unit="m" step={0.1} min={0.5}
          value={roomL.value} onChange={roomL.set} />
        <NumberField label="Room width" unit="m" step={0.1} min={0.5}
          value={roomW.value} onChange={roomW.set} />
        <NumberField label="Room height" unit="m" step={0.1} min={0.5}
          value={roomH.value} onChange={roomH.set} />
        <NumberField label="Wall reflectivity" step={0.05} min={0} max={1}
          value={wallR.value} onChange={wallR.set} />
      </Section>
    </div>
  );
}

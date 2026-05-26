/**
 * GeometrySliders — four sliders matching gui/channel_canvas.py controls.
 * Each slider writes to the central configStore; ChannelCanvas and
 * LinkBudgetTable subscribe via selectors.
 */

import { Slider } from "@/primitives/Slider";
import { useConfigStore } from "@/store/configStore";

export function GeometrySliders() {
  const distance = useConfigStore((s) => (s.config.distance_m as number) ?? 0.325);
  const txAngle = useConfigStore((s) => (s.config.tx_angle_deg as number) ?? 0);
  const rxTilt = useConfigStore((s) => (s.config.rx_tilt_deg as number) ?? 0);
  const pvArea = useConfigStore((s) => (s.config.sc_area_cm2 as number) ?? 9);
  const patch = useConfigStore((s) => s.patch);

  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
      <Slider
        label="Distance"
        unit="cm"
        min={0.05}
        max={2}
        step={0.005}
        value={distance}
        onChange={(v) => patch({ distance_m: v })}
        format={(v) => (v * 100).toFixed(1)}
      />
      <Slider
        label="TX angle"
        unit="°"
        min={-60}
        max={60}
        step={1}
        value={txAngle}
        onChange={(v) => patch({ tx_angle_deg: v })}
      />
      <Slider
        label="RX tilt"
        unit="°"
        min={-60}
        max={60}
        step={1}
        value={rxTilt}
        onChange={(v) => patch({ rx_tilt_deg: v })}
      />
      <Slider
        label="PV area"
        unit="cm²"
        min={1}
        max={36}
        step={0.5}
        value={pvArea}
        onChange={(v) => patch({ sc_area_cm2: v })}
        format={(v) => v.toFixed(1)}
      />
    </div>
  );
}

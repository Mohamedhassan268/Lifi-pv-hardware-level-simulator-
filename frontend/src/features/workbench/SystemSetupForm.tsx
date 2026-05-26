/**
 * SystemSetupForm — at-a-glance view of the currently selected TX / RX / Power
 * parts. Mirrors gui/tab_system_setup.py. Each row is read-only here — you
 * change parts by selecting one in the ComponentLibrary and clicking "Use as…".
 */

import { Card } from "@/primitives/Card";
import { useConfigStore } from "@/store/configStore";

const SLOTS: Array<{ field: string; label: string; group: string }> = [
  { field: "led_part", label: "LED", group: "Transmitter" },
  { field: "driver_part", label: "LED driver", group: "Transmitter" },
  { field: "pv_part", label: "Photodetector", group: "Receiver" },
  { field: "ina_part", label: "Amplifier", group: "Receiver" },
  { field: "comparator_part", label: "Comparator", group: "Receiver" },
];

export function SystemSetupForm() {
  const config = useConfigStore((s) => s.config);
  const presetName = useConfigStore((s) => s.presetName);

  const grouped: Record<string, typeof SLOTS> = {};
  for (const slot of SLOTS) {
    (grouped[slot.group] ??= []).push(slot);
  }

  return (
    <Card>
      <header className="mb-4 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Current System
        </h3>
        {presetName && (
          <p className="text-xs text-slate-500">
            based on <span className="font-mono text-slate-400">{presetName}</span>
          </p>
        )}
      </header>

      <div className="grid grid-cols-1 gap-5">
        {Object.entries(grouped).map(([group, slots]) => (
          <section key={group}>
            <h4 className="mb-2 text-[10px] uppercase tracking-[0.2em] text-beam-300/70">
              {group}
            </h4>
            <dl className="grid grid-cols-1 gap-2">
              {slots.map((slot) => {
                const value = config[slot.field] as string | undefined;
                return (
                  <div
                    key={slot.field}
                    className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2"
                  >
                    <span className="text-sm text-slate-400">{slot.label}</span>
                    <span className="font-mono text-sm text-slate-100">
                      {value ?? "—"}
                    </span>
                  </div>
                );
              })}
            </dl>
          </section>
        ))}
      </div>
    </Card>
  );
}

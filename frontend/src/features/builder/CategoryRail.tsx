/**
 * CategoryRail — left panel of the BuilderWorkspace. Four category buttons
 * (Transmitter / Receiver / Geometry / Noise) and a Simulate button.
 *
 * The preset-vs-own-design choice happens at the Landing page; once here,
 * Simulate runs straight through — no gate, no prompt.
 *
 * Each category button:
 *   - shows a 1-line summary derived from the live config
 *   - shows a checkmark when builderUIStore.configured[cat] === true
 *   - on click, opens the matching modal
 */

import { useState } from "react";

import { StandardsModal } from "@/features/standards/StandardsModal";
import { useDuplexRun } from "@/hooks/useDuplexRun";
import { useSimulationSocket } from "@/hooks/useSimulationSocket";
import { Button } from "@/primitives/Button";
import { useDuplexStore } from "@/store/duplexStore";
import {
  useActiveConfigured,
  useBuilderUIStore,
  type BuilderCategory,
} from "@/store/builderUIStore";
import { useConfigStore, useTwoSystem } from "@/store/configStore";
import { usePipelineStore } from "@/store/pipelineStore";
import { useStandardsStore } from "@/store/standardsStore";

type CategoryDef = {
  key: BuilderCategory;
  label: string;
  hint: string;
  summary: (cfg: Record<string, unknown>, configured: boolean) => string;
};

const CATEGORIES: CategoryDef[] = [
  {
    key: "transmitter",
    label: "Transmitter",
    hint: "LED · driver · modulation",
    summary: (c, ok) =>
      ok
        ? `${c.led_part ?? "—"} · ${formatA(c.bias_current_A)} · ${c.modulation ?? "OOK"}`
        : "Not configured",
  },
  {
    key: "receiver",
    label: "Receiver",
    hint: "PV cell · amplifier · ADC",
    summary: (c, ok) =>
      ok
        ? `${c.pv_part ?? "—"} · INA ${c.ina_gain_dB ?? 40}dB · ${c.adc_bits ?? 12}-bit ADC`
        : "Not configured",
  },
  {
    key: "geometry",
    label: "Geometry",
    hint: "Distance · angles · room",
    summary: (c, ok) =>
      ok
        ? `${formatM(c.distance_m)} · TX ${c.tx_angle_deg ?? 0}° · RX ${c.rx_tilt_deg ?? 0}°`
        : "Not configured",
  },
  {
    key: "noise",
    label: "Noise",
    hint: "6 physical sources",
    summary: (c, ok) =>
      ok ? (c.noise_enable ? "enabled (6 sources)" : "disabled") : "Not configured",
  },
  {
    key: "mcu",
    label: "Controller",
    hint: "ESP32 · Arduino · clock",
    summary: (c, ok) =>
      ok ? `${c.mcu_board ?? "MCU"} · ${c.mcu_clock_MHz ?? "—"} MHz` : "Not configured",
  },
];

export function CategoryRail() {
  const config = useConfigStore((s) => s.config);
  const systems = useConfigStore((s) => s.systems);
  const twoSystem = useTwoSystem();
  const selectEntity = useBuilderUIStore((s) => s.selectEntity);
  const selectedEntity = useBuilderUIStore((s) => s.selectedEntity);
  const configured = useActiveConfigured();
  const running = usePipelineStore((s) => s.running);
  const duplexRunning = useDuplexStore((s) => s.running);
  const { runSimulation, cancel } = useSimulationSocket();
  const { runDuplex } = useDuplexRun();

  const isRunning = twoSystem ? duplexRunning : running;

  const configuredCount = (Object.values(configured) as boolean[]).filter(Boolean).length;
  const activeProfileLabel = useStandardsStore((s) => s.activeProfileLabel);
  const [standardsOpen, setStandardsOpen] = useState(false);

  const onSimulate = () => {
    if (twoSystem && systems.B) {
      runDuplex(systems.A, systems.B);
    } else {
      runSimulation({ ...config, simulation_engine: "python" });
    }
  };

  const onPrecisionRun = () => {
    // 100k bits = measurable BER floor ~1e-5 with 95% Wilson interval.
    if (twoSystem && systems.B) {
      runDuplex({ ...systems.A, n_bits: 100000 }, { ...systems.B, n_bits: 100000 });
    } else {
      runSimulation({ ...config, simulation_engine: "python", n_bits: 100000 });
    }
  };

  return (
    <aside className="flex h-full flex-col border-r border-hair bg-slate-950">
      <header className="border-b border-hair px-3 py-2">
        <p className="label-inst">System Builder</p>
        <p className="readout mt-0.5 text-[10px] text-slate-500">
          {configuredCount}/{CATEGORIES.length} entities configured
        </p>
      </header>

      <SystemSwitcher />

      <div className="flex flex-col">
        {CATEGORIES.map((cat) => (
          <CategoryButton
            key={cat.key}
            def={cat}
            summary={cat.summary(config, configured[cat.key])}
            configured={configured[cat.key]}
            active={selectedEntity === cat.key}
            onOpen={() => selectEntity(cat.key)}
          />
        ))}
      </div>

      <div className="mt-auto flex flex-col gap-1 border-t border-hair p-2">
        <button
          type="button"
          onClick={() => setStandardsOpen(true)}
          className="flex items-center justify-between border border-hair bg-white/[0.02] px-2 py-1 text-[11px] text-slate-300 hover:bg-white/[0.05]"
        >
          <span className="label-inst">Standards</span>
          <span className="readout text-[10px] text-slate-500">
            {activeProfileLabel ?? "—"}
          </span>
        </button>

        {isRunning ? (
          twoSystem ? (
            <Button disabled className="w-full">
              Running A → B…
            </Button>
          ) : (
            <Button variant="ghost" className="w-full" onClick={cancel}>
              Cancel
            </Button>
          )
        ) : (
          <>
            <Button className="w-full" onClick={onSimulate}>
              ▶ Simulate{twoSystem ? " · A + B" : " · 10k bits"}
            </Button>
            <Button variant="ghost" className="w-full" onClick={onPrecisionRun}>
              ▶ Precision · 100k bits
            </Button>
          </>
        )}
      </div>

      <StandardsModal open={standardsOpen} onClose={() => setStandardsOpen(false)} />
    </aside>
  );
}

function CategoryButton({
  def,
  summary,
  configured,
  active,
  onOpen,
}: {
  def: CategoryDef;
  summary: string;
  configured: boolean;
  active: boolean;
  onOpen: () => void;
}) {
  const borderClass = active
    ? "border-l-beam-400 bg-beam-400/[0.05]"
    : configured
      ? "border-l-emerald-400/60 hover:bg-white/[0.03]"
      : "border-l-transparent hover:bg-white/[0.03]";

  return (
    <button
      type="button"
      onClick={onOpen}
      className={`flex flex-col gap-0.5 border-b border-hair border-l-2 px-2 py-1.5 text-left transition-colors ${borderClass}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="readout text-[11px] font-semibold text-slate-100">
          {def.label}
          {configured && (
            <span className="ml-1 text-emerald-400" aria-label="configured">
              ✓
            </span>
          )}
        </span>
        <span className="label-inst text-[9px] text-slate-600">{def.hint}</span>
      </div>
      <span
        className={`readout truncate text-[10px] ${
          configured ? "text-slate-400" : "text-slate-600"
        }`}
      >
        {summary}
      </span>
    </button>
  );
}

function formatA(v: unknown): string {
  if (typeof v !== "number") return "—";
  return v < 1 ? `${(v * 1000).toFixed(0)}mA` : `${v.toFixed(2)}A`;
}

function formatM(v: unknown): string {
  if (typeof v !== "number") return "—";
  return v < 1 ? `${(v * 100).toFixed(0)}cm` : `${v.toFixed(2)}m`;
}

// A/B system switcher. Single-system mode shows just "+ Add system B"; once B
// exists, shows the A|B tabs, a remove button, and the duplex-tie toggle.
function SystemSwitcher() {
  const twoSystem = useTwoSystem();
  const activeSystem = useConfigStore((s) => s.activeSystem);
  const setActiveSystem = useConfigStore((s) => s.setActiveSystem);
  const addSystemB = useConfigStore((s) => s.addSystemB);
  const removeSystemB = useConfigStore((s) => s.removeSystemB);
  const coupling = useConfigStore((s) => s.coupling);
  const setCoupling = useConfigStore((s) => s.setCoupling);

  if (!twoSystem) {
    return (
      <div className="border-b border-hair px-2 py-1.5">
        <button
          type="button"
          onClick={addSystemB}
          className="w-full border border-dashed border-hair px-2 py-1 text-[11px] text-slate-300 hover:bg-white/[0.04]"
        >
          + Add system B
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 border-b border-hair px-2 py-1.5">
      <div className="flex items-center gap-1">
        {(["A", "B"] as const).map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setActiveSystem(s)}
            className={
              "flex-1 border px-2 py-1 text-[11px] uppercase tracking-wider " +
              (activeSystem === s
                ? "border-beam-400/60 bg-beam-400/[0.08] text-beam-200"
                : "border-hair text-slate-400 hover:bg-white/[0.04]")
            }
          >
            System {s}
          </button>
        ))}
        <button
          type="button"
          onClick={removeSystemB}
          title="Remove system B"
          className="px-1.5 py-1 text-[11px] text-slate-500 hover:text-rose-300"
        >
          ✕
        </button>
      </div>
      <label className="flex items-center justify-between px-1 text-[10px] text-slate-400">
        <span>Link A↔B</span>
        <input
          type="checkbox"
          checked={coupling !== "none"}
          onChange={(e) => setCoupling(e.target.checked ? "duplex" : "none")}
          className="accent-beam-400"
        />
      </label>
      {coupling !== "none" && (
        <div className="flex gap-1 px-1">
          {(["duplex", "shared"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setCoupling(m)}
              className={
                "flex-1 border px-1.5 py-0.5 text-[10px] " +
                (coupling === m
                  ? "border-beam-400/60 bg-beam-400/[0.08] text-beam-200"
                  : "border-hair text-slate-400 hover:bg-white/[0.04]")
              }
            >
              {m === "duplex" ? "Duplex" : "Shared ch."}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

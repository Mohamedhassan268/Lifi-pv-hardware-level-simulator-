/**
 * Palette — left rail of placeable parts, grouped into collapsible sections so
 * the rail isn't a crowded button cloud:
 *   - Components: passives, system blocks, and the library active parts
 *     (fetched from /api/components, which carry their SPICE ports).
 *   - Instruments: probes/meters (measurement-only, no SPICE element).
 *   - Power: sources, ground, and the supply rails.
 *
 * Clicking a part drops it at a default position on the canvas; the user then
 * drags it. (Click-to-add is simpler and touch-friendly vs HTML5 DnD.)
 */

import { useEffect, useMemo, useState, type ReactNode } from "react";

import { api, type ComponentSummary, type SpicePort } from "@/api/client";
import { Card } from "@/primitives/Card";
import { symbolFor } from "@/features/schematic/symbols";
import { useSchematicStore, type PartNodeData } from "@/store/schematicStore";

type Group = "component" | "instrument" | "power";

// Primitives the editor can place that aren't registry components.
const BUILTINS: { label: string; group: Group; data: PartNodeData }[] = [
  // ---- Components ----
  {
    label: "Resistor",
    group: "component",
    data: {
      part: "R", ctype: "R", label: "Resistor", partCode: "R", symbol: "resistor", value: "1k",
      ports: [
        { name: "1", role: "signal" },
        { name: "2", role: "signal_out" },
      ],
    },
  },
  {
    label: "Capacitor",
    group: "component",
    data: {
      part: "C", ctype: "C", label: "Capacitor", partCode: "C", symbol: "capacitor", value: "1u",
      ports: [
        { name: "1", role: "signal" },
        { name: "2", role: "signal_out" },
      ],
    },
  },
  {
    label: "Data Source",
    group: "component",
    data: {
      part: "DRIVE", ctype: "DRIVE", label: "Data Source", partCode: "OOK", symbol: "drive",
      ports: [{ name: "out", role: "signal_out" }],
    },
  },
  {
    label: "Channel",
    group: "component",
    data: {
      part: "CHANNEL", ctype: "CHANNEL", label: "Channel", partCode: "optical", symbol: "channel",
      ports: [
        { name: "tx", role: "optical_in" },
        { name: "rx", role: "optical_out" },
      ],
    },
  },
  {
    label: "MCU (ESP32)",
    group: "component",
    data: {
      part: "MCU", ctype: "MCU", label: "MCU (ESP32)", partCode: "demod", symbol: "mcu",
      ports: [
        { name: "adc", role: "signal_in" },
        { name: "gpio", role: "signal_out" },
      ],
    },
  },
  // ---- Instruments ----
  {
    label: "Output",
    group: "instrument",
    data: {
      part: "OUT", ctype: "OUT", label: "Output", partCode: "dout", symbol: "output",
      ports: [{ name: "out", role: "output" }],
    },
  },
  {
    label: "Multimeter",
    group: "instrument",
    data: {
      part: "DMM", ctype: "PROBE_DMM", label: "Multimeter", partCode: "dmm", symbol: "multimeter",
      ports: [{ name: "probe", role: "signal_in" }],
    },
  },
  {
    label: "Scope",
    group: "instrument",
    data: {
      part: "SCOPE", ctype: "PROBE_SCOPE", label: "Scope", partCode: "scope", symbol: "scope",
      ports: [{ name: "probe", role: "signal_in" }],
    },
  },
  // ---- Power ----
  {
    label: "DC source",
    group: "power",
    data: {
      part: "V", ctype: "V", label: "DC source", partCode: "V", symbol: "source", value: "0",
      ports: [
        { name: "1", role: "signal" },
        { name: "2", role: "signal_out" },
      ],
    },
  },
  {
    label: "Ground",
    group: "power",
    data: {
      part: "GND", ctype: "GND", label: "Ground", partCode: "0", symbol: "ground",
      ports: [{ name: "gnd", role: "ground" }],
    },
  },
  {
    label: "VCC",
    group: "power",
    data: {
      part: "VCC", ctype: "VCC", label: "VCC", partCode: "vcc", symbol: "vcc",
      ports: [{ name: "v", role: "supply_pos" }],
    },
  },
  {
    label: "VEE",
    group: "power",
    data: {
      part: "VEE", ctype: "VEE", label: "VEE", partCode: "vee", symbol: "vee",
      ports: [{ name: "v", role: "supply_neg" }],
    },
  },
  {
    label: "VREF",
    group: "power",
    data: {
      part: "VREF", ctype: "VREF", label: "VREF", partCode: "vref", symbol: "vref",
      ports: [{ name: "v", role: "ref" }],
    },
  },
];

let dropIndex = 0;

/** Collapsible labelled section. */
function Section({
  title,
  count,
  children,
}: {
  title: string;
  count?: number;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className="border-b border-white/5 pb-2 last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between py-1.5 text-left"
      >
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
          {title}
          {count != null && <span className="ml-1 text-slate-600">({count})</span>}
        </span>
        <span className="text-slate-500">{open ? "▾" : "▸"}</span>
      </button>
      {open && <div className="pb-1">{children}</div>}
    </div>
  );
}

function PartButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1.5 text-xs
                 text-slate-200 hover:bg-white/[0.08]"
    >
      {label}
    </button>
  );
}

export function Palette() {
  const [rows, setRows] = useState<ComponentSummary[]>([]);
  const [portCache, setPortCache] = useState<Record<string, SpicePort[]>>({});
  const [error, setError] = useState<string | null>(null);
  const addPart = useSchematicStore((s) => s.addPart);

  useEffect(() => {
    api
      .listComponents()
      .then(setRows)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  // Parts that expose a wireable SPICE subcircuit + ports.
  const wireable = useMemo(
    () =>
      rows.filter((r) =>
        ["Amplifier", "Comparator", "Photodiode", "Solar Cell", "LED", "MOSFET"].includes(
          r.category,
        ),
      ),
    [rows],
  );

  const builtinsBy = (g: Group) => BUILTINS.filter((b) => b.group === g);

  const place = (data: PartNodeData) => {
    const x = 120 + (dropIndex % 4) * 60;
    const y = 80 + Math.floor(dropIndex / 4) * 60;
    dropIndex += 1;
    addPart(data, { x, y });
  };

  const placeLibraryPart = async (row: ComponentSummary) => {
    const part = row.part;
    let ports = portCache[part];
    if (!ports) {
      const detail = await api.getComponent(part);
      ports = detail.spice_ports ?? [];
      setPortCache((c) => ({ ...c, [part]: ports! }));
    }
    if (ports.length === 0) {
      setError(`${part} has no SPICE ports — cannot place.`);
      return;
    }
    place({
      part,
      ctype: part,
      label: row.display_name ?? row.category,
      partCode: part,
      symbol: symbolFor(row.category, part),
      ports,
    });
  };

  return (
    <Card>
      <header className="mb-2">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Parts</h3>
        <p className="mt-1 text-xs text-slate-500">Click to place on the canvas</p>
      </header>

      {error && <p className="mb-2 text-xs text-rose-400">{error}</p>}

      <Section title="Components" count={builtinsBy("component").length + wireable.length}>
        <div className="mb-2 flex flex-wrap gap-1.5">
          {builtinsBy("component").map((b) => (
            <PartButton key={b.label} label={b.label} onClick={() => place(b.data)} />
          ))}
        </div>
        <ul className="max-h-[16rem] divide-y divide-white/5 overflow-y-auto rounded-lg border border-white/10">
          {wireable.map((r) => (
            <li key={r.class}>
              <button
                type="button"
                onClick={() => placeLibraryPart(r)}
                className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-white/[0.04]"
              >
                <span className="flex flex-col">
                  <span className="text-xs text-slate-100">{r.display_name ?? r.category}</span>
                  <span className="font-mono text-[10px] text-slate-500">{r.part}</span>
                </span>
                <span className="text-[10px] uppercase tracking-wider text-slate-400">
                  {r.category}
                </span>
              </button>
            </li>
          ))}
          {wireable.length === 0 && !error && (
            <li className="px-3 py-4 text-center text-xs text-slate-500">Loading…</li>
          )}
        </ul>
      </Section>

      <Section title="Instruments" count={builtinsBy("instrument").length}>
        <div className="flex flex-wrap gap-1.5">
          {builtinsBy("instrument").map((b) => (
            <PartButton key={b.label} label={b.label} onClick={() => place(b.data)} />
          ))}
        </div>
      </Section>

      <Section title="Power" count={builtinsBy("power").length}>
        <div className="flex flex-wrap gap-1.5">
          {builtinsBy("power").map((b) => (
            <PartButton key={b.label} label={b.label} onClick={() => place(b.data)} />
          ))}
        </div>
      </Section>
    </Card>
  );
}

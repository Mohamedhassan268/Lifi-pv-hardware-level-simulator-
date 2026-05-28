/**
 * SchematicsTab — renders the SVG returned by /api/schematic/{preset}/{idx}.
 *
 * The dropdowns are seeded from /api/schematic/index. If the currently loaded
 * preset (from configStore) has a registry entry, that's selected by default;
 * otherwise the first preset wins. The SVG is fetched directly by the browser
 * via an <img src=…>, so we don't need to plumb it through state.
 *
 * The KiCad download buttons live below — they target the KiCad export route
 * already wired by KicadExportCard, so users can hand the .kicad_sch off to
 * a real CAD tool when they want PCB layout / DRC.
 */

import { useEffect, useMemo, useState } from "react";

import {
  api,
  type SchematicIndex,
  type SchematicPresetEntry,
} from "@/api/client";
import { Card } from "@/primitives/Card";
import { useConfigStore } from "@/store/configStore";

export function SchematicsTab() {
  const presetName = useConfigStore((s) => s.presetName);
  const [index, setIndex] = useState<SchematicIndex | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [selectedDrawing, setSelectedDrawing] = useState<number>(0);
  const [svgErr, setSvgErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listSchematics()
      .then((r) => {
        if (cancelled) return;
        setIndex(r);
        setLoadError(null);
      })
      .catch((e) => {
        if (!cancelled) {
          setLoadError(e instanceof Error ? e.message : String(e));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Default preset selection: current preset if it has a registry entry,
  // else first available.
  useEffect(() => {
    if (!index || index.presets.length === 0) return;
    if (selectedPreset && index.presets.some((p) => p.key === selectedPreset)) {
      return;
    }
    const match = presetName
      ? index.presets.find((p) => p.key === presetName)
      : null;
    setSelectedPreset(match?.key ?? index.presets[0].key);
    setSelectedDrawing(0);
  }, [index, presetName, selectedPreset]);

  // Reset drawing index when switching preset so an out-of-range pick can't
  // linger and 404.
  useEffect(() => {
    setSelectedDrawing(0);
    setSvgErr(null);
  }, [selectedPreset]);

  const currentEntry: SchematicPresetEntry | null = useMemo(() => {
    if (!index || !selectedPreset) return null;
    return index.presets.find((p) => p.key === selectedPreset) ?? null;
  }, [index, selectedPreset]);

  if (loadError) {
    return (
      <Card>
        <p className="text-sm text-rose-300">
          Could not load schematic registry: {loadError}
        </p>
      </Card>
    );
  }

  if (!index) {
    return (
      <Card>
        <p className="text-sm text-slate-400">Loading schematic registry…</p>
      </Card>
    );
  }

  if (index.presets.length === 0) {
    return (
      <Card>
        <p className="text-sm text-slate-400">
          No schematic drawings registered. The backend reports an empty
          registry (schemdraw may not be installed in the sidecar).
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-[2fr_3fr]">
          <label className="flex flex-col gap-1">
            <span className="text-[11px] uppercase tracking-wider text-slate-400">
              Paper
            </span>
            <select
              value={selectedPreset ?? ""}
              onChange={(e) => setSelectedPreset(e.target.value)}
              className="rounded-lg border border-slate-700 bg-slate-800/40 px-2 py-1.5 text-sm text-slate-100"
            >
              {index.presets.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] uppercase tracking-wider text-slate-400">
              Schematic
            </span>
            <select
              value={selectedDrawing}
              onChange={(e) => setSelectedDrawing(Number(e.target.value))}
              disabled={!currentEntry}
              className="rounded-lg border border-slate-700 bg-slate-800/40 px-2 py-1.5 text-sm text-slate-100 disabled:opacity-50"
            >
              {currentEntry?.drawings.map((d) => (
                <option key={d.index} value={d.index}>
                  {d.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </Card>

      <Card>
        {currentEntry ? (
          <div className="overflow-auto rounded-lg bg-white p-4">
            <img
              key={`${currentEntry.key}-${selectedDrawing}`}
              src={api.schematicSvgUrl(currentEntry.key, selectedDrawing)}
              alt={`${currentEntry.label} — ${
                currentEntry.drawings[selectedDrawing]?.name ?? "schematic"
              }`}
              onError={() =>
                setSvgErr(
                  "Render failed (HTTP error). Check the backend log — schemdraw may have raised on this drawing.",
                )
              }
              onLoad={() => setSvgErr(null)}
              className="mx-auto block max-w-full"
            />
            {svgErr ? (
              <p className="mt-3 text-xs text-rose-600">{svgErr}</p>
            ) : null}
          </div>
        ) : null}
      </Card>

      {presetName && currentEntry && presetName === currentEntry.key ? (
        <Card>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            CAD handoff
          </h4>
          <p className="mb-3 text-xs text-slate-400">
            Open the same circuit in KiCad for PCB layout / DRC. Run the
            export from the Builder rail first if these 404.
          </p>
          <div className="flex flex-wrap gap-2">
            <a
              href={api.kicadDownloadUrl(presetName, "sch")}
              download
              className="rounded-lg border border-sky-500/40 bg-sky-500/10 px-3 py-2 text-xs font-medium text-sky-200 hover:bg-sky-500/20"
            >
              ⬇ {presetName}.kicad_sch
            </a>
            <a
              href={api.kicadDownloadUrl(presetName, "bom")}
              download
              className="rounded-lg border border-slate-700 bg-slate-800/40 px-3 py-2 text-xs font-mono text-slate-200 hover:border-sky-500/50"
            >
              ⬇ {presetName}_bom.csv
            </a>
          </div>
        </Card>
      ) : null}
    </div>
  );
}

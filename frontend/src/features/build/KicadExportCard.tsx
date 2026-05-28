/**
 * KicadExportCard — Builder-rail card that triggers a KiCad schematic + BOM
 * export for the currently loaded preset. Disabled when the preset has no
 * graph builder registered on the backend (today: only `kadirvelu2021`).
 *
 * On success the card switches to a result view with file counts, warnings,
 * and download links. Files live on disk in `workspace/kicad/` and are
 * served by GET /api/kicad/download/{preset}/{kind}.
 */

import { useEffect, useState } from "react";

import { api, type KicadExportResult } from "@/api/client";
import { Card } from "@/primitives/Card";
import { useConfigStore } from "@/store/configStore";

type Status =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "ok"; result: KicadExportResult }
  | { kind: "err"; message: string };

export function KicadExportCard() {
  const presetName = useConfigStore((s) => s.presetName);
  const [available, setAvailable] = useState<string[] | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  useEffect(() => {
    let cancelled = false;
    api
      .listKicadPresets()
      .then((r) => {
        if (!cancelled) setAvailable(r.presets);
      })
      .catch(() => {
        if (!cancelled) setAvailable([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Reset the result when the user loads a different preset.
  useEffect(() => {
    setStatus({ kind: "idle" });
  }, [presetName]);

  const supported = presetName !== null && (available?.includes(presetName) ?? false);

  async function runExport() {
    if (!presetName) return;
    setStatus({ kind: "running" });
    try {
      const result = await api.exportKicad(presetName);
      setStatus({ kind: "ok", result });
    } catch (e) {
      setStatus({ kind: "err", message: e instanceof Error ? e.message : String(e) });
    }
  }

  return (
    <Card>
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">
        KiCad export
      </h3>

      {!presetName ? (
        <p className="text-xs text-slate-400">
          Load a preset to enable KiCad export.
        </p>
      ) : !supported ? (
        <p className="text-xs text-slate-400">
          No KiCad graph builder for{" "}
          <span className="font-mono text-slate-300">{presetName}</span> yet.
          {available && available.length > 0 ? (
            <>
              {" "}Available:{" "}
              <span className="font-mono text-slate-300">
                {available.join(", ")}
              </span>
              .
            </>
          ) : null}
        </p>
      ) : status.kind === "ok" ? (
        <ExportResultView
          result={status.result}
          onReset={() => setStatus({ kind: "idle" })}
        />
      ) : (
        <div className="space-y-2">
          <p className="text-xs text-slate-400">
            Generate <span className="font-mono text-slate-300">.kicad_sch</span>{" "}
            and <span className="font-mono text-slate-300">_bom.csv</span> for{" "}
            <span className="font-mono text-slate-300">{presetName}</span>.
          </p>
          <button
            type="button"
            onClick={runExport}
            disabled={status.kind === "running"}
            className="w-full rounded-lg border border-sky-500/40 bg-sky-500/10 px-3 py-2 text-sm font-medium text-sky-200 transition hover:bg-sky-500/20 disabled:cursor-wait disabled:opacity-60"
          >
            {status.kind === "running" ? "Exporting…" : "Export schematic + BOM"}
          </button>
          {status.kind === "err" ? (
            <p className="rounded-lg border border-rose-500/30 bg-rose-500/5 px-3 py-2 font-mono text-[11px] text-rose-300">
              {status.message}
            </p>
          ) : null}
        </div>
      )}
    </Card>
  );
}

function ExportResultView({
  result,
  onReset,
}: {
  result: KicadExportResult;
  onReset: () => void;
}) {
  return (
    <div className="space-y-3 text-xs text-slate-300">
      <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-2 text-emerald-200">
        ✓ Exported {result.component_count} components · {result.net_count} nets
      </div>

      <div className="flex flex-col gap-2">
        <a
          href={api.kicadDownloadUrl(result.preset, "sch")}
          download
          className="rounded-lg border border-slate-700 bg-slate-800/40 px-3 py-2 font-mono text-[11px] hover:border-sky-500/50 hover:text-sky-200"
        >
          ⬇ {result.preset}.kicad_sch
        </a>
        <a
          href={api.kicadDownloadUrl(result.preset, "bom")}
          download
          className="rounded-lg border border-slate-700 bg-slate-800/40 px-3 py-2 font-mono text-[11px] hover:border-sky-500/50 hover:text-sky-200"
        >
          ⬇ {result.preset}_bom.csv
        </a>
      </div>

      {result.warnings.length > 0 ? (
        <details className="text-[11px]">
          <summary className="cursor-pointer text-amber-200">
            {result.warnings.length} warning
            {result.warnings.length === 1 ? "" : "s"}
          </summary>
          <ul className="mt-2 space-y-1 pl-4">
            {result.warnings.map((w, i) => (
              <li key={i} className="text-amber-100/80">
                {w}
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      <button
        type="button"
        onClick={onReset}
        className="text-[11px] text-slate-400 underline-offset-2 hover:text-slate-200 hover:underline"
      >
        Run again
      </button>
    </div>
  );
}

/**
 * ExportButton — opens a native folder picker (Tauri dialog) and writes
 * config.json + one PNG per Plotly panel to the picked folder.
 *
 * Outside Tauri (e.g. `npm run dev` in a browser) the button is rendered
 * disabled with a tooltip indicating the desktop build is required.
 *
 * The panel order is hardcoded to match ResultsPanel.tsx's mount order:
 *   1. WaveformsPanel       → waveforms.png
 *   2. EyeDiagramPanel      → eye.png
 *   3. ConstellationPanel   → decision.png
 *   4. SpectrumPanel        → spectrum.png
 *   5. BitsComparePanel     → bits.png
 *
 * If a panel rendered an empty state (no Plotly element), it is skipped.
 */

import { useState } from "react";

import { exportResults } from "@/lib/exporter";
import { isTauri, pickFolder } from "@/lib/tauriBridge";
import { Button } from "@/primitives/Button";
import { useConfigStore } from "@/store/configStore";

const FILE_ORDER = [
  "waveforms.png",
  "eye.png",
  "decision.png",
  "spectrum.png",
  "bits.png",
];

type Status =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "saved"; folder: string; count: number; errors: number }
  | { kind: "error"; message: string };

export function ExportButton() {
  const config = useConfigStore((s) => s.config);
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  const onClick = async () => {
    if (!isTauri || status.kind === "saving") return;
    try {
      const folder = await pickFolder();
      if (!folder) return;
      setStatus({ kind: "saving" });

      // Collect Plotly DOM nodes in panel order.
      const nodes = Array.from(
        document.querySelectorAll<HTMLElement>(".js-plotly-plot"),
      );
      const plots = nodes
        .slice(0, FILE_ORDER.length)
        .map((el, i) => ({ filename: FILE_ORDER[i], el }));

      const result = await exportResults(folder, { config, plots });
      setStatus({
        kind: "saved",
        folder,
        count: result.written.length,
        errors: result.errors.length,
      });
    } catch (e) {
      setStatus({
        kind: "error",
        message: e instanceof Error ? e.message : String(e),
      });
    }
  };

  const label =
    status.kind === "saving"
      ? "Saving…"
      : status.kind === "saved"
        ? `Saved ${status.count} file${status.count === 1 ? "" : "s"}`
        : "Export to folder";

  return (
    <div className="flex flex-col items-end gap-1">
      <Button
        variant="ghost"
        onClick={onClick}
        disabled={!isTauri || status.kind === "saving"}
        title={
          !isTauri
            ? "Available in the desktop build"
            : status.kind === "saved"
              ? status.folder
              : undefined
        }
      >
        {label}
      </Button>
      {status.kind === "saved" && status.errors > 0 && (
        <p className="text-[11px] text-rose-400">
          {status.errors} file{status.errors === 1 ? "" : "s"} failed
        </p>
      )}
      {status.kind === "error" && (
        <p className="text-[11px] text-rose-400">{status.message}</p>
      )}
    </div>
  );
}

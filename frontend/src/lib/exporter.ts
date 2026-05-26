/**
 * exporter — orchestrates the "Save to folder" flow:
 *
 *   1. The user picks a folder via Tauri's native dialog (in tauriBridge.ts).
 *   2. exportResults() writes:
 *        - config.json   (snapshot of the live SystemConfig)
 *        - <name>.png    (one per Plotly panel element passed in)
 *
 * Plotly figures: we use the global Plotly object's `toImage(el, opts)` to
 * rasterize each panel to PNG (1200×800). The element is the DOM node react-
 * plotly.js mounts under the hood — every PlotCanvas wraps exactly one, and
 * Plotly tags it with the class `.js-plotly-plot`.
 */

// @ts-expect-error — plotly.js-dist-min ships without TS types
import Plotly from "plotly.js-dist-min";

import type { ConfigDict } from "@/api/client";
import { joinPath, writeBinary, writeText } from "@/lib/tauriBridge";

export interface PlotSpec {
  filename: string; // e.g. "waveforms.png"
  el: HTMLElement;
}

export interface ExportInput {
  config: ConfigDict;
  plots: PlotSpec[];
}

export interface ExportResult {
  written: string[];
  errors: Array<{ filename: string; message: string }>;
}

const PNG_WIDTH = 1200;
const PNG_HEIGHT = 800;

export async function exportResults(
  folder: string,
  input: ExportInput,
): Promise<ExportResult> {
  const written: string[] = [];
  const errors: ExportResult["errors"] = [];

  // 1. Config snapshot
  try {
    const path = joinPath(folder, "config.json");
    await writeText(path, JSON.stringify(input.config, null, 2));
    written.push(path);
  } catch (e) {
    errors.push({
      filename: "config.json",
      message: e instanceof Error ? e.message : String(e),
    });
  }

  // 2. PNGs of each plot
  for (const { filename, el } of input.plots) {
    try {
      const dataUrl: string = await Plotly.toImage(el, {
        format: "png",
        width: PNG_WIDTH,
        height: PNG_HEIGHT,
      });
      const bytes = dataUrlToBytes(dataUrl);
      const path = joinPath(folder, filename);
      await writeBinary(path, bytes);
      written.push(path);
    } catch (e) {
      errors.push({
        filename,
        message: e instanceof Error ? e.message : String(e),
      });
    }
  }

  return { written, errors };
}

function dataUrlToBytes(dataUrl: string): Uint8Array {
  const commaIdx = dataUrl.indexOf(",");
  const base64 = commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : dataUrl;
  const binary = atob(base64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

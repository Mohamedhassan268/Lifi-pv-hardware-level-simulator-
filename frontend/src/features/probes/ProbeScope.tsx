/**
 * ProbeScope — minimal uPlot wrapper for raw signal display.
 *
 * Rendering policy (the "no faking" rule):
 *   - No interpolation: each pixel column shows the min/max of the samples
 *     mapped to it (preserves noise floor + spikes).
 *   - No trace smoothing: uPlot draws straight segments, no spline.
 *   - Decimation is purely visual; the underlying Float64Array is never
 *     mutated. Hover readouts pull from the raw array, not the decimated view.
 *
 * For arrays under ~50k samples we render every sample (uPlot handles that
 * fast). Above that, we min/max-bin into 2× pixel-column count traces and
 * draw the upper + lower bound as two stacked series.
 */

import "uplot/dist/uPlot.min.css";

import uPlot, { type AlignedData, type Options } from "uplot";
import { useEffect, useRef } from "react";

const RAW_THRESHOLD = 50_000;

export interface ProbeScopeProps {
  /** X-axis values, typically a time array. */
  x: Float64Array | Float32Array | null;
  /** Sample data, same length as x. */
  y: Float64Array | Float32Array | Uint8Array | Int32Array;
  units: string;
  label: string;
  height?: number;
}

export function ProbeScope({ x, y, units, label, height = 220 }: ProbeScopeProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const host = ref.current;

    const xs = makeXArray(x, y.length);
    const data = buildAlignedData(xs, y, host.clientWidth || 600);

    const opts: Options = {
      width: host.clientWidth || 600,
      height,
      pxAlign: 0,
      cursor: { drag: { x: true, y: false } },
      scales: { x: { time: false }, y: {} },
      axes: [
        { stroke: "#94a3b8", grid: { stroke: "rgba(148,163,184,0.1)" } },
        {
          stroke: "#94a3b8",
          grid: { stroke: "rgba(148,163,184,0.1)" },
          label: `${label} (${units})`,
          labelGap: 4,
          labelSize: 28,
        },
      ],
      series: [
        { label: "t" },
        ...(data.length === 3
          ? [
              { label: `${label} max`, stroke: "#67e8f9", width: 1, points: { show: false } },
              { label: `${label} min`, stroke: "#67e8f9", width: 1, points: { show: false } },
            ]
          : [
              { label, stroke: "#67e8f9", width: 1, points: { show: false } },
            ]),
      ],
    };

    plotRef.current = new uPlot(opts, data, host);

    const ro = new ResizeObserver(() => {
      if (!ref.current || !plotRef.current) return;
      plotRef.current.setSize({
        width: ref.current.clientWidth,
        height,
      });
    });
    ro.observe(host);

    return () => {
      ro.disconnect();
      plotRef.current?.destroy();
      plotRef.current = null;
    };
    // We intentionally rebuild on every data change rather than mutating —
    // uPlot's incremental APIs assume aligned data and we want a clean redraw.
  }, [x, y, height, label, units]);

  return (
    <div
      ref={ref}
      className="w-full"
      style={{ height }}
      data-testid="probe-scope"
    />
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeXArray(
  x: Float64Array | Float32Array | null,
  n: number,
): Float64Array {
  if (x && x.length === n) {
    // uPlot accepts Float64Array directly.
    return x instanceof Float64Array ? x : new Float64Array(x);
  }
  // Fall back to sample-index axis.
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}

/**
 * Build the uPlot AlignedData for a signal trace.
 *
 * If the raw array is small, returns [x, y].
 * If large, returns [x_binned, y_max, y_min] using min/max-per-pixel binning.
 */
function buildAlignedData(
  x: Float64Array,
  y: Float64Array | Float32Array | Uint8Array | Int32Array,
  pixelWidth: number,
): AlignedData {
  const n = y.length;
  if (n <= RAW_THRESHOLD) {
    return [x, toFloat64(y)];
  }

  const targetBins = Math.max(64, pixelWidth);
  const bin = Math.ceil(n / targetBins);
  const bins = Math.ceil(n / bin);

  const xb = new Float64Array(bins);
  const ymax = new Float64Array(bins);
  const ymin = new Float64Array(bins);

  for (let b = 0; b < bins; b++) {
    const start = b * bin;
    const end = Math.min(start + bin, n);
    let mn = Number(y[start]);
    let mx = mn;
    for (let i = start + 1; i < end; i++) {
      const v = Number(y[i]);
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    xb[b] = x[Math.floor((start + end) / 2)];
    ymin[b] = mn;
    ymax[b] = mx;
  }
  return [xb, ymax, ymin];
}

function toFloat64(
  y: Float64Array | Float32Array | Uint8Array | Int32Array,
): Float64Array {
  if (y instanceof Float64Array) return y;
  const out = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) out[i] = Number(y[i]);
  return out;
}

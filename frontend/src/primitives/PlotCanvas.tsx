/**
 * PlotCanvas — Plotly wrapper.
 *
 * Hard rules (per the migration plan §5):
 *   1. Do NOT wrap this in a `motion.div` with `layout` — Plotly's internal
 *      ResizeObserver would fire every animation frame and reflow a 100k-point
 *      trace into oblivion.
 *   2. The Plotly node is `absolute inset-0` inside a `relative` aspect-locked
 *      wrapper, so the chart fills its parent without ever changing the
 *      surrounding layout.
 *
 * Implementation notes:
 *   - We bind react-plotly.js to `plotly.js-dist-min` via createPlotlyComponent
 *     so we only ship one Plotly bundle (the minified one we installed) and
 *     avoid importing the much larger `plotly.js` by mistake.
 *   - `data` and `layout` are accepted as already-memoized props from the
 *     caller — this component is intentionally dumb so Plotly only re-renders
 *     when its inputs change identity.
 */

import createPlotlyComponent from "react-plotly.js/factory";
// @ts-expect-error — plotly.js-dist-min ships without TS types
import Plotly from "plotly.js-dist-min";
import type { Data, Layout, Config } from "plotly.js";

const Plot = createPlotlyComponent(Plotly);

interface PlotCanvasProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  aspect?: string;
  className?: string;
}

const DARK_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "rgb(203, 213, 225)", family: "ui-sans-serif, system-ui" },
  margin: { l: 56, r: 16, t: 24, b: 40 },
  xaxis: { gridcolor: "rgba(255,255,255,0.06)", zerolinecolor: "rgba(255,255,255,0.12)" },
  yaxis: { gridcolor: "rgba(255,255,255,0.06)", zerolinecolor: "rgba(255,255,255,0.12)" },
  showlegend: true,
  legend: { bgcolor: "rgba(0,0,0,0)", orientation: "h", y: 1.15 },
};

const DEFAULT_CONFIG: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
};

export function PlotCanvas({
  data,
  layout,
  config,
  aspect = "16/9",
  className = "",
}: PlotCanvasProps) {
  const merged: Partial<Layout> = {
    ...DARK_LAYOUT,
    ...layout,
    xaxis: { ...DARK_LAYOUT.xaxis, ...(layout?.xaxis ?? {}) },
    yaxis: { ...DARK_LAYOUT.yaxis, ...(layout?.yaxis ?? {}) },
  };
  return (
    <div
      className={`relative w-full ${className}`}
      style={{ aspectRatio: aspect }}
    >
      <Plot
        data={data}
        layout={merged}
        config={{ ...DEFAULT_CONFIG, ...config }}
        useResizeHandler
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      />
    </div>
  );
}

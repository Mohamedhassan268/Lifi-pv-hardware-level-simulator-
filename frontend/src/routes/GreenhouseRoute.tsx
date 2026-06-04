/**
 * GreenhouseRoute — interactive multi-node coverage canvas (Path C).
 *
 * Place ceiling luminaires on the floor (click to add, click a marker to remove)
 * and watch the link coverage, PV harvest, and self-powered "deployable" region
 * update live. Calls the fast analytical /api/coverage endpoint, so it's
 * interactive; the transient scripts remain the high-fidelity reference.
 */

import { useCallback, useEffect, useState } from "react";

import { api, type CoverageResponse } from "@/api/client";

const SIZE = 460; // px, square canvas

const LAYOUTS: Record<string, [number, number][]> = {
  "1": [[0, 0]],
  "4": [[-1, -1], [1, -1], [-1, 1], [1, 1]],
  "9": [
    [-1.2, -1.2], [0, -1.2], [1.2, -1.2],
    [-1.2, 0], [0, 0], [1.2, 0],
    [-1.2, 1.2], [0, 1.2], [1.2, 1.2],
  ],
};

type Mode = "deployable" | "harvest" | "snr";

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * Math.max(0, Math.min(1, t));
}

function harvestColor(v: number, lo: number, hi: number): string {
  const t = hi > lo ? (v - lo) / (hi - lo) : 0;
  // black -> orange -> yellow
  const r = Math.round(lerp(20, 255, Math.min(t * 2, 1)));
  const g = Math.round(lerp(10, 230, t));
  const b = Math.round(lerp(30, 60, t));
  return `rgb(${r},${g},${b})`;
}

function snrColor(db: number): string {
  const t = (db + 20) / 60; // -20..40 dB
  const r = Math.round(lerp(180, 26, t));
  const g = Math.round(lerp(40, 152, t));
  const b = Math.round(lerp(60, 80, t));
  return `rgb(${r},${g},${b})`;
}

export function GreenhouseRoute() {
  const [totalPowerW, setTotalPowerW] = useState(8);
  const [heightM, setHeightM] = useState(2.5);
  const [halfExtentM, setHalfExtentM] = useState(2.0);
  const [grid, setGrid] = useState(15);
  const [halfAngleDeg, setHalfAngleDeg] = useState(40);
  const [rSenseOhm, setRSenseOhm] = useState(500);
  const [nodeBudgetUW, setNodeBudgetUW] = useState(10);
  const [lumins, setLumins] = useState<[number, number][]>(LAYOUTS["1"]);
  const [mode, setMode] = useState<Mode>("deployable");
  const [res, setRes] = useState<CoverageResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = useCallback(async () => {
    setBusy(true);
    setErr(null);
    try {
      const r = await api.coverage({
        luminaires: lumins.map(([x, y]) => ({ x, y })),
        total_power_W: totalPowerW,
        height_m: heightM,
        half_extent_m: halfExtentM,
        grid,
        half_angle_deg: halfAngleDeg,
        r_sense_ohm: rSenseOhm,
        node_budget_uW: nodeBudgetUW,
      });
      setRes(r);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [lumins, totalPowerW, heightM, halfExtentM, grid, halfAngleDeg,
      rSenseOhm, nodeBudgetUW]);

  // Auto-run (debounced) whenever an input changes — endpoint is ~0.5 s.
  useEffect(() => {
    const t = setTimeout(run, 250);
    return () => clearTimeout(t);
  }, [run]);

  const px = (m: number) => ((m + halfExtentM) / (2 * halfExtentM)) * SIZE;
  const py = (m: number) => SIZE - ((m + halfExtentM) / (2 * halfExtentM)) * SIZE;

  const onCanvasClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * 2 * halfExtentM - halfExtentM;
    const my = halfExtentM - ((e.clientY - rect.top) / rect.height) * 2 * halfExtentM;
    setLumins((ls) => [...ls, [Number(mx.toFixed(2)), Number(my.toFixed(2))]]);
  };

  const removeLumin = (i: number) =>
    setLumins((ls) => ls.filter((_, j) => j !== i));

  const cells: React.ReactNode[] = [];
  if (res) {
    const n = res.xs.length;
    const cw = SIZE / n;
    for (let iy = 0; iy < n; iy++) {
      for (let ix = 0; ix < n; ix++) {
        let fill: string;
        if (mode === "deployable") {
          const closes = res.BER[iy][ix] <= 0.01;
          fill = res.deployable[iy][ix] ? "#1a9850" : closes ? "#f0a73a" : "#3b0f3f";
        } else if (mode === "harvest") {
          fill = harvestColor(res.harvest_uW[iy][ix], res.harvest_min_uW, res.harvest_max_uW);
        } else {
          fill = snrColor(res.snr_dB[iy][ix]);
        }
        cells.push(
          <rect key={`${iy}-${ix}`} x={ix * cw} y={(n - 1 - iy) * cw}
            width={cw + 0.6} height={cw + 0.6} fill={fill} />,
        );
      }
    }
  }

  return (
    <section className="mx-auto max-w-6xl px-6 py-6">
      <header className="mb-4">
        <h1 className="text-lg font-semibold">Greenhouse coverage canvas</h1>
        <p className="text-xs text-slate-400">
          Click the floor to add a ceiling luminaire; click a marker to remove it.
          The {totalPowerW} W total optical power is split across all luminaires.
          Maps update live (analytical fast path).
        </p>
      </header>

      <div className="flex flex-wrap gap-6">
        {/* Canvas */}
        <div className="flex flex-col gap-2">
          <div className="flex gap-1">
            {(["deployable", "harvest", "snr"] as Mode[]).map((m) => (
              <button key={m} type="button" onClick={() => setMode(m)}
                className={"px-2 py-1 text-[11px] uppercase tracking-wider border-b-2 " +
                  (mode === m ? "border-beam-400 text-beam-300"
                    : "border-transparent text-slate-400 hover:text-slate-200")}>
                {m}
              </button>
            ))}
            {busy && <span className="ml-2 self-center text-[11px] text-slate-500">computing…</span>}
          </div>
          <svg width={SIZE} height={SIZE} onClick={onCanvasClick}
            className="cursor-crosshair rounded-lg border border-hair bg-slate-900">
            {cells}
            {lumins.map(([lx, ly], i) => (
              <g key={i} onClick={(e) => { e.stopPropagation(); removeLumin(i); }}
                className="cursor-pointer">
                <circle cx={px(lx)} cy={py(ly)} r={8} fill="#fff" stroke="#000" strokeWidth={1.5} />
                <circle cx={px(lx)} cy={py(ly)} r={3} fill="#222" />
              </g>
            ))}
          </svg>
          {err && <p className="max-w-[460px] text-[11px] text-rose-400">{err}</p>}
        </div>

        {/* Controls + summary */}
        <div className="flex min-w-[260px] flex-1 flex-col gap-3">
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Data coverage" value={res ? `${res.data_cov_pct.toFixed(0)}%` : "—"} />
            <Stat label="Deployable" value={res ? `${res.deployable_pct.toFixed(0)}%` : "—"}
              accent />
            <Stat label="Harvest range"
              value={res ? `${res.harvest_min_uW.toFixed(1)}–${res.harvest_max_uW.toFixed(0)} µW` : "—"} />
            <Stat label="Per luminaire"
              value={`${(totalPowerW / Math.max(lumins.length, 1)).toFixed(2)} W ×${lumins.length}`} />
          </div>

          <div className="flex gap-1">
            {Object.keys(LAYOUTS).map((k) => (
              <button key={k} type="button" onClick={() => setLumins(LAYOUTS[k])}
                className="rounded border border-hair px-2 py-1 text-[11px] text-slate-300 hover:bg-white/5">
                {k} luminaire{k === "1" ? "" : "s"}
              </button>
            ))}
            <button type="button" onClick={() => setLumins([])}
              className="rounded border border-hair px-2 py-1 text-[11px] text-slate-400 hover:bg-white/5">
              clear
            </button>
          </div>

          <Slider label="Total optical power" unit="W" min={1} max={32} step={1}
            value={totalPowerW} onChange={setTotalPowerW} />
          <Slider label="Node power budget" unit="µW" min={1} max={50} step={1}
            value={nodeBudgetUW} onChange={setNodeBudgetUW} />
          <Slider label="Ceiling height" unit="m" min={1} max={5} step={0.1}
            value={heightM} onChange={setHeightM} />
          <Slider label="Floor half-size" unit="m" min={1} max={4} step={0.25}
            value={halfExtentM} onChange={setHalfExtentM} />
          <Slider label="LED half-angle" unit="°" min={10} max={80} step={5}
            value={halfAngleDeg} onChange={setHalfAngleDeg} />
          <Slider label="PV load (op. point)" unit="Ω" min={100} max={5000} step={100}
            value={rSenseOhm} onChange={setRSenseOhm} />
          <Slider label="Grid resolution" unit="" min={7} max={31} step={2}
            value={grid} onChange={setGrid} />

          <p className="text-[11px] leading-relaxed text-slate-500">
            <b>deployable</b> = link closes <i>and</i> harvest ≥ node budget (green);
            <span className="text-amber-400"> amber</span> = linked but energy-starved.
            Distributing the same power across more luminaires lowers peak harvest —
            watch the deployable area shrink.
          </p>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-lg border border-hair bg-white/[0.02] px-3 py-2">
      <div className="label-inst">{label}</div>
      <div className={"readout text-sm " + (accent ? "text-emerald-300" : "text-slate-200")}>
        {value}
      </div>
    </div>
  );
}

function Slider({ label, unit, min, max, step, value, onChange }: {
  label: string; unit: string; min: number; max: number; step: number;
  value: number; onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex justify-between text-[11px] text-slate-400">
        <span>{label}</span>
        <span className="readout text-slate-200">{value}{unit}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="accent-beam-400" />
    </label>
  );
}

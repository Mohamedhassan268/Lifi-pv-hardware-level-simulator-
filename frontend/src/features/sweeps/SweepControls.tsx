/**
 * SweepControls — kicks off thermal, Monte Carlo, or one of the two new
 * BER sweep types (ber_snr, ber_distance).
 */

import { useState } from "react";

import { useSweepSocket } from "@/hooks/useSweepSocket";
import { Button } from "@/primitives/Button";
import { Card } from "@/primitives/Card";
import { useConfigStore } from "@/store/configStore";
import { useSweepStore, type SweepKind } from "@/store/sweepStore";

const DEFAULT_TEMPS = "-10, 0, 25, 45, 65, 85";

const KIND_LABEL: Record<SweepKind, string> = {
  thermal: "Thermal",
  montecarlo: "Monte Carlo",
  ber_snr: "BER vs SNR",
  ber_distance: "BER vs distance",
};

export function SweepControls() {
  const presetName = useConfigStore((s) => s.presetName);
  const running = useSweepStore((s) => s.running);
  const reset = useSweepStore((s) => s.reset);
  const { runSweep, cancel } = useSweepSocket();

  const [kind, setKind] = useState<SweepKind>("thermal");

  // Thermal
  const [tempsStr, setTempsStr] = useState(DEFAULT_TEMPS);
  // Monte Carlo
  const [nRuns, setNRuns] = useState(50);
  const [tolPct, setTolPct] = useState(5);
  const [autoTolerance, setAutoTolerance] = useState(true);
  // BER sweeps
  const [nPoints, setNPoints] = useState(12);
  const [luxLo, setLuxLo] = useState(1);
  const [luxHi, setLuxHi] = useState(10000);
  const [distLo, setDistLo] = useState(0.1);
  const [distHi, setDistHi] = useState(10);

  const start = () => {
    if (!presetName) return;
    switch (kind) {
      case "thermal": {
        const temps = tempsStr
          .split(/[, ]+/)
          .map((s) => parseFloat(s.trim()))
          .filter((n) => Number.isFinite(n));
        runSweep("thermal", presetName, { temps_C: temps });
        break;
      }
      case "montecarlo":
        runSweep("montecarlo", presetName, autoTolerance
          ? { n_runs: nRuns, auto_tolerance: true, seed: 1 }
          : { n_runs: nRuns, tolerance_spec: { sc_responsivity: tolPct / 100 }, seed: 1 });
        break;
      case "ber_snr":
        runSweep("ber_snr", presetName, {
          n_points: nPoints,
          ambient_lux_lo: luxLo,
          ambient_lux_hi: luxHi,
        });
        break;
      case "ber_distance":
        runSweep("ber_distance", presetName, {
          n_points: nPoints,
          distance_lo_m: distLo,
          distance_hi_m: distHi,
        });
        break;
    }
  };

  const kinds: SweepKind[] = ["thermal", "montecarlo", "ber_snr", "ber_distance"];

  return (
    <Card>
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-300">
        Sweep Controls
      </h3>

      <div className="mb-4 grid grid-cols-2 gap-1 rounded-xl border border-white/10 bg-white/[0.02] p-1 text-xs">
        {kinds.map((k) => (
          <button
            key={k}
            type="button"
            disabled={running}
            onClick={() => setKind(k)}
            className={
              "rounded-lg px-2 py-1.5 transition-colors " +
              (kind === k
                ? "bg-beam-400/[0.15] text-beam-200"
                : "text-slate-400 hover:text-slate-200")
            }
          >
            {KIND_LABEL[k]}
          </button>
        ))}
      </div>

      {kind === "thermal" && (
        <ThermalInputs tempsStr={tempsStr} setTempsStr={setTempsStr} running={running} />
      )}
      {kind === "montecarlo" && (
        <MonteCarloInputs
          nRuns={nRuns}
          tolPct={tolPct}
          setNRuns={setNRuns}
          setTolPct={setTolPct}
          autoTolerance={autoTolerance}
          setAutoTolerance={setAutoTolerance}
          running={running}
        />
      )}
      {(kind === "ber_snr" || kind === "ber_distance") && (
        <BerInputs
          kind={kind}
          nPoints={nPoints}
          setNPoints={setNPoints}
          luxLo={luxLo}
          luxHi={luxHi}
          setLuxLo={setLuxLo}
          setLuxHi={setLuxHi}
          distLo={distLo}
          distHi={distHi}
          setDistLo={setDistLo}
          setDistHi={setDistHi}
          running={running}
        />
      )}

      <div className="mt-5 flex items-center gap-2">
        {running ? (
          <Button variant="ghost" onClick={cancel}>
            Cancel
          </Button>
        ) : (
          <>
            <Button onClick={start} disabled={!presetName}>
              Run {KIND_LABEL[kind]}
            </Button>
            <Button variant="ghost" onClick={reset}>
              Clear
            </Button>
          </>
        )}
      </div>
    </Card>
  );
}

function ThermalInputs({
  tempsStr,
  setTempsStr,
  running,
}: {
  tempsStr: string;
  setTempsStr: (v: string) => void;
  running: boolean;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs uppercase tracking-wider text-slate-400">
        Temperatures (°C)
      </span>
      <input
        type="text"
        value={tempsStr}
        onChange={(e) => setTempsStr(e.target.value)}
        disabled={running}
        className="w-full rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2
                   text-sm text-slate-100 placeholder:text-slate-500
                   focus:outline-none focus:ring-2 focus:ring-beam-400/40
                   disabled:opacity-50"
      />
      <span className="mt-1 block text-xs text-slate-500">
        Comma-separated. The sweep runs the preset at each temperature.
      </span>
    </label>
  );
}

function MonteCarloInputs({
  nRuns,
  tolPct,
  setNRuns,
  setTolPct,
  autoTolerance,
  setAutoTolerance,
  running,
}: {
  nRuns: number;
  tolPct: number;
  setNRuns: (n: number) => void;
  setTolPct: (n: number) => void;
  autoTolerance: boolean;
  setAutoTolerance: (v: boolean) => void;
  running: boolean;
}) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <NumberField label="# runs" value={nRuns} onChange={setNRuns} min={5} max={500} disabled={running} />
      <div />
      <div className="col-span-2">
        <span className="mb-1 block text-xs uppercase tracking-wider text-slate-400">
          Tolerance source
        </span>
        <div className="flex rounded-lg border border-white/10 bg-white/[0.02] p-1 text-xs">
          {[
            { v: true, label: "Component datasheets" },
            { v: false, label: "Custom (manual)" },
          ].map((opt) => (
            <button
              key={String(opt.v)}
              type="button"
              disabled={running}
              onClick={() => setAutoTolerance(opt.v)}
              className={
                "flex-1 rounded-md px-2 py-1.5 transition-colors " +
                (autoTolerance === opt.v
                  ? "bg-beam-400/[0.15] text-beam-200"
                  : "text-slate-400 hover:text-slate-200")
              }
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
      {!autoTolerance && (
        <>
          <NumberField label="R tol (%)" value={tolPct} onChange={setTolPct} min={0} max={50} step={0.5} disabled={running} />
          <p className="col-span-2 text-xs text-slate-500">
            Each trial samples sc_responsivity uniformly within ±{tolPct}% of nominal.
          </p>
        </>
      )}
      {autoTolerance && (
        <p className="col-span-2 text-xs text-slate-500">
          Tolerances are derived from the components named in the live config (PV, LED, INA).
          The resolved table appears below once the run starts.
        </p>
      )}
    </div>
  );
}

function BerInputs({
  kind,
  nPoints,
  setNPoints,
  luxLo,
  luxHi,
  setLuxLo,
  setLuxHi,
  distLo,
  distHi,
  setDistLo,
  setDistHi,
  running,
}: {
  kind: "ber_snr" | "ber_distance";
  nPoints: number;
  setNPoints: (n: number) => void;
  luxLo: number;
  luxHi: number;
  setLuxLo: (n: number) => void;
  setLuxHi: (n: number) => void;
  distLo: number;
  distHi: number;
  setDistLo: (n: number) => void;
  setDistHi: (n: number) => void;
  running: boolean;
}) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <NumberField label="# points" value={nPoints} onChange={setNPoints} min={4} max={40} disabled={running} />
      <div />
      {kind === "ber_snr" ? (
        <>
          <NumberField label="lux lo" value={luxLo} onChange={setLuxLo} min={0.01} disabled={running} />
          <NumberField label="lux hi" value={luxHi} onChange={setLuxHi} min={1} disabled={running} />
          <p className="col-span-2 text-xs text-slate-500">
            Ambient illuminance is the SNR knob. Log-spaced between the two endpoints. Higher lux → more shot noise → lower SNR.
          </p>
        </>
      ) : (
        <>
          <NumberField label="d lo (m)" value={distLo} onChange={setDistLo} min={0.01} step={0.01} disabled={running} />
          <NumberField label="d hi (m)" value={distHi} onChange={setDistHi} min={0.1} step={0.1} disabled={running} />
          <p className="col-span-2 text-xs text-slate-500">
            Log-spaced distances. Channel gain falls as 1/r^(m+3), so BER rises with distance.
          </p>
        </>
      )}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs uppercase tracking-wider text-slate-400">
        {label}
      </span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value || "0") || 0)}
        disabled={disabled}
        className="w-full rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2
                   font-mono text-sm text-slate-100
                   focus:outline-none focus:ring-2 focus:ring-beam-400/40
                   disabled:opacity-50"
      />
    </label>
  );
}

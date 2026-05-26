/**
 * AcRoute — small-signal AC analysis page. A single Run button kicks off
 * `useAcSocket.runAc(config, params)`; the result lands in acStore and
 * renders in BodePanel.
 */

import { motion } from "framer-motion";
import { useState } from "react";

import { BodePanel } from "@/features/results/BodePanel";
import { useAcSocket } from "@/hooks/useAcSocket";
import { DUR, EASE } from "@/lib/motion";
import { Button } from "@/primitives/Button";
import { Card } from "@/primitives/Card";
import { useAcStore } from "@/store/acStore";
import { useConfigStore } from "@/store/configStore";
import { useUIStore } from "@/store/uiStore";

export function AcRoute() {
  const config = useConfigStore((s) => s.config);
  const running = useAcStore((s) => s.running);
  const setRoute = useUIStore((s) => s.setRoute);
  const { runAc } = useAcSocket();

  const [fStart, setFStart] = useState(1);
  const [fStop, setFStop] = useState(1e7);
  const [nPoints, setNPoints] = useState(400);

  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: DUR.base, ease: EASE }}
      className="mx-auto max-w-7xl px-8 py-10"
    >
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">AC analysis</h1>
          <p className="text-sm text-slate-400">
            Small-signal frequency response of the RX chain
            (sense resistor → INA → BPF). Gain and phase margins are
            extracted from unity-gain and 180°-phase crossings.
          </p>
        </div>
        <Button variant="ghost" onClick={() => setRoute("builder")}>
          ← Back to Builder
        </Button>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <Card>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-300">
              Sweep
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <Field label="f_start (Hz)" value={fStart} onChange={setFStart} step={0.1} />
              <Field label="f_stop (Hz)" value={fStop} onChange={setFStop} step={1} />
              <Field label="# points" value={nPoints} onChange={setNPoints} step={50} />
            </div>
            <div className="mt-4">
              <Button
                className="w-full"
                disabled={running}
                onClick={() =>
                  runAc(config, {
                    f_start_hz: fStart,
                    f_stop_hz: fStop,
                    n_points: nPoints,
                  })
                }
              >
                {running ? "Running…" : "Run AC sweep"}
              </Button>
            </div>
            <p className="mt-3 text-[11px] text-slate-500">
              Closed-form Python (scipy.signal-style cascade); no SPICE round-trip.
              Single-pole INA model; first-order HP+LP for each BPF stage.
            </p>
          </Card>
        </div>
        <div className="lg:col-span-2">
          <BodePanel />
        </div>
      </div>
    </motion.section>
  );
}

function Field({
  label,
  value,
  onChange,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  step?: number;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs uppercase tracking-wider text-slate-400">
        {label}
      </span>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value || "0") || 0)}
        className="w-full rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2
                   font-mono text-sm text-slate-100
                   focus:outline-none focus:ring-2 focus:ring-beam-400/40"
      />
    </label>
  );
}

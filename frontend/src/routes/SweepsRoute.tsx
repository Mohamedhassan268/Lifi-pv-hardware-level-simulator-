/**
 * SweepsRoute — page that hosts the thermal + Monte Carlo sweep controls and
 * their live chart. Replaces gui/tab_results.py "Energy Harvest"-style sub-tab
 * for sweep visualisation.
 */

import { motion } from "framer-motion";

import { SweepChart } from "@/features/sweeps/SweepChart";
import { SweepControls } from "@/features/sweeps/SweepControls";
import { SweepProgress } from "@/features/sweeps/SweepProgress";
import { ToleranceTable } from "@/features/sweeps/ToleranceTable";
import { DUR, EASE } from "@/lib/motion";
import { Button } from "@/primitives/Button";
import { useUIStore } from "@/store/uiStore";

export function SweepsRoute() {
  const setRoute = useUIStore((s) => s.setRoute);

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
          <h1 className="text-2xl font-semibold">Parameter Sweeps</h1>
          <p className="text-sm text-slate-400">
            Thermal robustness and Monte Carlo yield. Points stream in live as
            each trial completes.
          </p>
        </div>
        <Button variant="ghost" onClick={() => setRoute("setup")}>
          ← Back to Setup
        </Button>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1 space-y-6">
          <SweepControls />
          <SweepProgress />
          <ToleranceTable />
        </div>
        <div className="lg:col-span-2">
          <SweepChart />
        </div>
      </div>
    </motion.section>
  );
}

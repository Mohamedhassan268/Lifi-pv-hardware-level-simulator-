/**
 * EngineRoute — pipeline cards on top, metrics + results below. Mirrors
 * gui/view_engine.py.
 */

import { motion } from "framer-motion";

import { MetricsCard } from "@/features/engine/MetricsCard";
import { PipelineCards } from "@/features/engine/PipelineCards";
import { ExportButton } from "@/features/results/ExportButton";
import { ResultsPanel } from "@/features/results/ResultsPanel";
import { ComplianceBanner } from "@/features/standards/ComplianceBanner";
import { DUR, EASE } from "@/lib/motion";
import { Button } from "@/primitives/Button";
import { usePipelineStore } from "@/store/pipelineStore";
import { useUIStore } from "@/store/uiStore";

export function EngineRoute() {
  const error = usePipelineStore((s) => s.error);
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
          <h1 className="text-2xl font-semibold">Simulation Engine</h1>
          <p className="text-sm text-slate-400">
            TX → Channel → RX. Updates stream live.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <ExportButton />
          <Button variant="ghost" onClick={() => setRoute("builder")}>
            ← Back to Builder
          </Button>
        </div>
      </header>

      <PipelineCards />

      <div className="mt-6">
        <ComplianceBanner />
      </div>

      {error && (
        <div className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300">
          {error}
        </div>
      )}

      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <MetricsCard />
        </div>
        <div className="lg:col-span-2">
          <ResultsPanel />
        </div>
      </div>
    </motion.section>
  );
}

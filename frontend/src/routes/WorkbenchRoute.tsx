/**
 * WorkbenchRoute — replaces gui/workbench_window.py. Three columns:
 *   - Current system (read-only, mirrors what's in configStore)
 *   - Component library (filter + search)
 *   - Component parameter table (apply button)
 */

import { motion } from "framer-motion";
import { useState } from "react";

import { ComponentLibrary } from "@/features/workbench/ComponentLibrary";
import { ComponentParamTable } from "@/features/workbench/ComponentParamTable";
import { SystemSetupForm } from "@/features/workbench/SystemSetupForm";
import { DUR, EASE } from "@/lib/motion";
import { Button } from "@/primitives/Button";
import { useUIStore } from "@/store/uiStore";

export function WorkbenchRoute() {
  const [selected, setSelected] = useState<string | null>(null);
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
          <h1 className="text-2xl font-semibold">System Workbench</h1>
          <p className="text-sm text-slate-400">
            Browse the component library and swap parts into the current preset.
          </p>
        </div>
        <Button variant="ghost" onClick={() => setRoute("setup")}>
          ← Done
        </Button>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="lg:col-span-3">
          <SystemSetupForm />
        </div>
        <div className="lg:col-span-4">
          <ComponentLibrary selected={selected} onSelect={setSelected} />
        </div>
        <div className="lg:col-span-5">
          <ComponentParamTable part={selected} />
        </div>
      </div>
    </motion.section>
  );
}

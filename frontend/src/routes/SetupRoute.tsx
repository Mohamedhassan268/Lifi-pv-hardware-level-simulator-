/**
 * SetupRoute — landing page. Hero on top, then the channel canvas + sliders +
 * link budget. Route-level animations are owned by App.tsx (LayoutGroup +
 * AnimatePresence) so this component stays focused on layout.
 */

import { motion } from "framer-motion";

import { Hero } from "@/components/Hero";
import { ChannelCanvas } from "@/features/setup/ChannelCanvas";
import { GeometrySliders } from "@/features/setup/GeometrySliders";
import { LinkBudgetTable } from "@/features/setup/LinkBudgetTable";
import { DUR, EASE } from "@/lib/motion";
import { Card } from "@/primitives/Card";
import { useUIStore } from "@/store/uiStore";

export function SetupRoute() {
  const setRoute = useUIStore((s) => s.setRoute);

  const openBuilder = () => setRoute("builder");

  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: DUR.base, ease: EASE }}
      className="min-h-full"
    >
      <Hero />

      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-8 py-12 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <div className="mb-4 flex items-baseline justify-between gap-4">
            <div>
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                Optical Channel
              </h2>
              <p className="mt-1 text-sm text-slate-400">
                Adjust geometry to explore the link budget. Updates stream live
                from the simulator.
              </p>
            </div>
            <button
              type="button"
              onClick={openBuilder}
              className="shrink-0 rounded-lg border border-beam-400/40 bg-beam-400/[0.08] px-3 py-1.5
                         text-xs font-medium text-beam-200 transition-colors
                         hover:border-beam-400/60 hover:bg-beam-400/[0.14]"
            >
              Build my own system →
            </button>
          </div>
          <ChannelCanvas />
          <div className="mt-6">
            <GeometrySliders />
          </div>
        </Card>

        <LinkBudgetTable />
      </div>
    </motion.section>
  );
}

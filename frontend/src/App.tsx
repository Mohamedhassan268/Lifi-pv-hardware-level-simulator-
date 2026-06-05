/**
 * App shell. Landing → Setup / Builder / Engine / Sweeps route swap via
 * LayoutGroup + AnimatePresence — all transitions are transform/opacity
 * only.
 *
 * Boot sequence:
 *   1. waitForBackend() — inside Tauri, wait for window.__BACKEND_PORT and
 *      a successful /health probe. Outside Tauri this returns immediately.
 *
 * The TopBar is hidden on the landing route so the splash is full-bleed.
 * The user advances into the builder via the LaunchChoiceModal triggered
 * by the Landing page's Builder CTA.
 */

import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import { useEffect, useState } from "react";

import { waitForBackend } from "@/api/backend";
import { FormToggle } from "@/features/topbar/FormToggle";
import { TopBar } from "@/features/topbar/TopBar";
import { DUR, EASE } from "@/lib/motion";
import { applyWorkspaceSpec, parseWorkspaceHash } from "@/lib/workspace";
import { AcRoute } from "@/routes/AcRoute";
import { BuilderRoute } from "@/routes/BuilderRoute";
import { EngineRoute } from "@/routes/EngineRoute";
import { SchematicRoute } from "@/routes/SchematicRoute";
import { InspectorRoute } from "@/routes/InspectorRoute";
import { GreenhouseRoute } from "@/routes/GreenhouseRoute";
import { MeasuredRoute } from "@/routes/MeasuredRoute";
import { LandingRoute } from "@/routes/LandingRoute";
import { SetupRoute } from "@/routes/SetupRoute";
import { SweepsRoute } from "@/routes/SweepsRoute";
import { useUIStore } from "@/store/uiStore";

export function App() {
  const route = useUIStore((s) => s.route);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      await waitForBackend();
      if (cancelled) return;
      // A dedicated workspace window is told what to be via the URL hash.
      // The launcher window has no hash and stays on the landing page.
      const spec = parseWorkspaceHash(window.location.hash);
      if (spec) await applyWorkspaceSpec(spec);
      if (cancelled) return;
      setReady(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (!ready) return <BootSplash />;

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      {route !== "landing" && <TopBar />}
      <main className="relative flex-1">
        <FormToggle />
        <LayoutGroup>
          <AnimatePresence mode="wait">
            {route === "landing" && <LandingRoute key="landing" />}
            {route === "setup" && <SetupRoute key="setup" />}
            {route === "engine" && <EngineRoute key="engine" />}
            {route === "sweeps" && <SweepsRoute key="sweeps" />}
            {route === "builder" && <BuilderRoute key="builder" />}
            {route === "schematic" && <SchematicRoute key="schematic" />}
            {route === "ac" && <AcRoute key="ac" />}
            {route === "inspector" && <InspectorRoute key="inspector" />}
            {route === "greenhouse" && <GreenhouseRoute key="greenhouse" />}
            {route === "measured" && <MeasuredRoute key="measured" />}
          </AnimatePresence>
        </LayoutGroup>
      </main>
    </div>
  );
}

function BootSplash() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950 text-slate-100">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: DUR.base, ease: EASE }}
        className="flex flex-col items-center gap-4"
      >
        <motion.div
          className="h-3 w-3 rounded-full bg-beam-300 shadow-[0_0_28px_rgba(103,232,249,0.7)] will-change-transform"
          animate={{ scale: [1, 1.4, 1], opacity: [0.7, 1, 0.7] }}
          transition={{ duration: 1.4, ease: EASE, repeat: Infinity }}
        />
        <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
          Starting simulator…
        </p>
      </motion.div>
    </div>
  );
}

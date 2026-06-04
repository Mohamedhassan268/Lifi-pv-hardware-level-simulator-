/**
 * LandingRoute — first-paint splash. Single Builder CTA opens the
 * LaunchChoiceModal (preset vs build-your-own). No top bar is rendered
 * on this route (App.tsx handles that).
 */

import { motion } from "framer-motion";
import { useState } from "react";

import { LaunchChoiceModal } from "@/features/landing/LaunchChoiceModal";
import { DUR, EASE } from "@/lib/motion";
import { useUIStore } from "@/store/uiStore";

export function LandingRoute() {
  const [choiceOpen, setChoiceOpen] = useState(false);
  const setRoute = useUIStore((s) => s.setRoute);

  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: DUR.base, ease: EASE }}
      className="relative flex min-h-screen flex-col items-center justify-center
                 overflow-hidden bg-slate-950 px-8 text-slate-100"
    >
      {/* Animated background grid */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.06]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(103,232,249,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(103,232,249,0.5) 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />
      {/* Radial glow */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 60% 50% at 50% 40%, rgba(34,211,238,0.10), transparent 70%)",
        }}
      />

      <BeamArt />

      <motion.div
        initial={{ y: 12, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.15, duration: DUR.slow, ease: EASE }}
        className="relative z-10 mt-10 max-w-2xl text-center"
      >
        <p className="text-[11px] uppercase tracking-[0.3em] text-beam-400/70">
          Optical wireless · LiFi · PV
        </p>
        <h1 className="mt-4 text-5xl font-semibold leading-tight tracking-tight">
          Design and simulate a hardware-faithful{" "}
          <span className="text-beam-300">optical communication</span> link.
        </h1>
        <p className="mt-5 text-sm leading-relaxed text-slate-400">
          Configure the transmitter, channel, receiver, and noise model.
          Run a full TX → Channel → RX simulation with real component
          datasheets behind it.
        </p>

        <motion.button
          type="button"
          onClick={() => setChoiceOpen(true)}
          whileHover={{ y: -2, scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          transition={{ duration: DUR.fast, ease: EASE }}
          className="mt-10 inline-flex items-center gap-2 rounded-xl
                     bg-beam-400 px-6 py-3 text-sm font-semibold text-slate-950
                     shadow-[0_0_36px_-8px_rgba(34,211,238,0.7)]
                     hover:bg-beam-300"
        >
          Builder
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14M13 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </motion.button>

        <p className="mt-3 text-[11px] text-slate-500">
          Choose a preset to start fast, or build a system from scratch.
        </p>

        <button
          type="button"
          onClick={() => setRoute("greenhouse")}
          className="mt-6 text-[12px] text-slate-400 underline-offset-4 hover:text-beam-300 hover:underline"
        >
          → Greenhouse coverage canvas
        </button>
      </motion.div>

      <LaunchChoiceModal open={choiceOpen} onClose={() => setChoiceOpen(false)} />
    </motion.section>
  );
}

function BeamArt() {
  return (
    <motion.svg
      width="320"
      height="120"
      viewBox="0 0 320 120"
      className="relative z-10"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.05, duration: DUR.slow, ease: EASE }}
    >
      <defs>
        <linearGradient id="beam" x1="0" y1="0.5" x2="1" y2="0.5">
          <stop offset="0%" stopColor="rgba(34,211,238,0.0)" />
          <stop offset="50%" stopColor="rgba(34,211,238,0.7)" />
          <stop offset="100%" stopColor="rgba(251,191,36,0.0)" />
        </linearGradient>
      </defs>
      {/* TX dot */}
      <motion.circle
        cx={40}
        cy={60}
        r={6}
        fill="#22d3ee"
        animate={{ opacity: [0.6, 1, 0.6], scale: [1, 1.15, 1] }}
        transition={{ duration: 1.6, ease: EASE, repeat: Infinity }}
      />
      {/* Beam line */}
      <line x1={50} y1={60} x2={270} y2={60} stroke="url(#beam)" strokeWidth={2} />
      {/* Moving photons */}
      {[0, 0.33, 0.66].map((d, i) => (
        <motion.circle
          key={i}
          r={2.5}
          fill="#67e8f9"
          initial={{ cx: 50, cy: 60, opacity: 0 }}
          animate={{ cx: [50, 270], opacity: [0, 1, 0] }}
          transition={{ duration: 1.8, ease: EASE, repeat: Infinity, delay: d * 1.8 }}
        />
      ))}
      {/* RX dot */}
      <motion.circle
        cx={280}
        cy={60}
        r={6}
        fill="#fbbf24"
        animate={{ opacity: [0.6, 1, 0.6], scale: [1, 1.15, 1] }}
        transition={{ duration: 1.6, ease: EASE, repeat: Infinity, delay: 0.8 }}
      />
    </motion.svg>
  );
}

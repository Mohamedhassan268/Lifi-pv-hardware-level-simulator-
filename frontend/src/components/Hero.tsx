/**
 * Hero — top-of-app banner. Showcases the LiFi/PV story (TX LED → light cone
 * → PV cell) with stagger-revealed copy and a looping beam.
 *
 * Animation rules (all enforced here):
 *   - transform/opacity only (no width/height/top/left)
 *   - fixed aspect-[5/3] stage so the viewport never reflows
 *   - useReducedMotion disables the continuous loops
 *   - all transitions use the shared EASE token
 */

import { motion, useReducedMotion, type Variants } from "framer-motion";

import { EASE } from "@/lib/motion";

const container: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08, delayChildren: 0.15 } },
};

const lineUp: Variants = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: EASE } },
};

export function Hero() {
  const reduce = useReducedMotion();
  const beamAnim = reduce
    ? { opacity: 0.85 }
    : { opacity: [0.55, 0.95, 0.55], scaleX: [0.985, 1.015, 0.985] };

  return (
    <section className="relative isolate overflow-hidden bg-slate-950 text-slate-100">
      {/* Subtle radial glow */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 [background:radial-gradient(60%_50%_at_50%_0%,rgba(34,211,238,0.10),transparent_70%)]"
      />
      <div className="mx-auto grid min-h-[78vh] max-w-7xl grid-cols-12 items-center gap-8 px-8 py-24">
        {/* Copy */}
        <motion.div
          className="col-span-12 md:col-span-6"
          variants={container}
          initial="hidden"
          animate="show"
        >
          <motion.p
            variants={lineUp}
            className="mb-4 text-sm uppercase tracking-[0.2em] text-beam-300/80"
          >
            Hardware-faithful LiFi / PV co-simulation
          </motion.p>
          <motion.h1
            variants={lineUp}
            className="text-5xl font-semibold leading-tight md:text-6xl"
          >
            Light becomes <span className="text-beam-300">data</span>
            <br />
            and <span className="text-harvest-300">power</span>.
          </motion.h1>
          <motion.p
            variants={lineUp}
            className="mt-6 max-w-lg text-slate-300"
          >
            Validate your TX → Channel → RX chain against six peer-reviewed
            papers, with SPICE-grade circuits and Python-fast sweeps.
          </motion.p>
        </motion.div>

        {/* Stage: TX LED → cone → PV cell. Fixed aspect — no layout shift. */}
        <div className="col-span-12 md:col-span-6">
          <div className="relative mx-auto aspect-[5/3] w-full max-w-xl">
            {/* TX LED */}
            <motion.div
              className="absolute left-[6%] top-1/2 h-10 w-10 -translate-y-1/2 rounded-full
                         bg-beam-300 shadow-[0_0_60px_18px_rgba(103,232,249,0.55)] will-change-transform"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.6, ease: EASE }}
            />
            {/* Light cone — only transform/opacity animate */}
            <motion.svg
              viewBox="0 0 500 300"
              className="absolute inset-0 h-full w-full"
              initial={{ opacity: 0 }}
              animate={beamAnim}
              transition={{
                duration: 2.8,
                ease: EASE,
                repeat: Infinity,
                repeatType: "mirror",
              }}
              style={{ originX: 0.06, originY: 0.5, willChange: "transform, opacity" }}
            >
              <defs>
                <linearGradient id="beam" x1="0" x2="1">
                  <stop offset="0%" stopColor="rgb(103,232,249)" stopOpacity="0.85" />
                  <stop offset="100%" stopColor="rgb(252,211,77)" stopOpacity="0.25" />
                </linearGradient>
              </defs>
              <polygon points="40,150 460,90 460,210" fill="url(#beam)" />
            </motion.svg>
            {/* PV cell */}
            <motion.div
              className="absolute right-[4%] top-1/2 h-24 w-16 -translate-y-1/2 rounded-md
                         bg-gradient-to-br from-harvest-300 to-harvest-500
                         shadow-[0_0_40px_4px_rgba(252,211,77,0.35)] will-change-transform"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.6, ease: EASE, delay: 0.2 }}
            />
            {/* Photons — transform-only loop, no width/height */}
            {!reduce &&
              Array.from({ length: 5 }).map((_, i) => (
                <motion.span
                  key={i}
                  className="absolute left-[10%] top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-white/80 will-change-transform"
                  initial={{ x: 0, opacity: 0 }}
                  animate={{ x: "78%", opacity: [0, 1, 0] }}
                  transition={{
                    duration: 1.6,
                    ease: EASE,
                    repeat: Infinity,
                    delay: i * 0.32,
                  }}
                />
              ))}
          </div>
        </div>
      </div>
    </section>
  );
}

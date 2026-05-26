/**
 * NoiseOverlay — visual interference layer rendered on top of the
 * ChannelNode's beam envelope. Visible only when noise has been applied
 * AND `noise_enable` is true.
 *
 * Two layers:
 *   1. Grain — ~36 small circles with phase-offset opacity flicker
 *   2. Beam distortion — a sine-wobble path that drifts horizontally
 *
 * pointer-events-none so it never intercepts node clicks.
 */

import { AnimatePresence, motion } from "framer-motion";

// Pre-computed pseudo-random positions for the grain layer (deterministic
// so React can reconcile across renders).
const GRAIN_POINTS: Array<{ x: number; y: number; seed: number }> = (() => {
  const pts: Array<{ x: number; y: number; seed: number }> = [];
  let s = 1;
  // simple LCG so positions are stable
  const rand = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  for (let i = 0; i < 36; i++) {
    pts.push({
      x: 12 + rand() * 176,
      y: 16 + rand() * 32,
      seed: i,
    });
  }
  return pts;
})();

// Two beam-distortion paths Framer Motion alternates between.
const WAVE_A = "M 8 32 Q 50 22 100 32 T 192 32";
const WAVE_B = "M 8 32 Q 50 42 100 32 T 192 32";

export function NoiseOverlay() {
  return (
    <AnimatePresence>
      <motion.g
        key="noise"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.25 }}
        className="pointer-events-none"
        aria-hidden
      >
        {/* Grain layer */}
        {GRAIN_POINTS.map((p) => (
          <motion.circle
            key={p.seed}
            cx={p.x}
            cy={p.y}
            r={1.5}
            fill="#fda4af"
            animate={{ opacity: [0.05, 0.45, 0.05] }}
            transition={{
              duration: 0.4 + (p.seed % 7) * 0.07,
              repeat: Infinity,
              delay: (p.seed * 0.013) % 0.5,
              ease: "easeInOut",
            }}
          />
        ))}

        {/* Beam distortion wobble */}
        <motion.path
          d={WAVE_A}
          fill="none"
          stroke="rgba(244,63,94,0.55)"
          strokeWidth={1.1}
          animate={{ d: [WAVE_A, WAVE_B, WAVE_A] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.g>
    </AnimatePresence>
  );
}

/**
 * PipelineCards — the 3-step orchestration view (TX / Channel / RX). Mirrors
 * gui/tab_simulation_engine.py.
 */

import { motion } from "framer-motion";

import { StepCard } from "@/features/engine/StepCard";
import { DUR, EASE } from "@/lib/motion";
import { usePipelineStore, type StepName } from "@/store/pipelineStore";

const ORDER: StepName[] = ["TX", "Channel", "RX"];

export function PipelineCards() {
  const steps = usePipelineStore((s) => s.steps);
  const running = usePipelineStore((s) => s.running);

  const activeName: StepName | null =
    running ? (ORDER.find((n) => steps[n].status === "running") ?? null) : null;

  return (
    <motion.ul
      layout
      transition={{ duration: DUR.base, ease: EASE }}
      className="grid grid-cols-1 gap-4 md:grid-cols-3"
    >
      {ORDER.map((name) => {
        const step = steps[name];
        return (
          <StepCard
            key={name}
            name={name}
            status={step.status}
            message={step.message}
            duration_s={step.duration_s}
            active={activeName === name}
          />
        );
      })}
    </motion.ul>
  );
}

/**
 * Section — visual grouping for related Builder fields. Animates in via
 * opacity/y when conditionally rendered (e.g. BPF block only shows when
 * topology=ina_bpf_comp).
 */

import { motion } from "framer-motion";
import { type ReactNode } from "react";

import { DUR, EASE } from "@/lib/motion";

interface SectionProps {
  title: string;
  description?: string;
  children: ReactNode;
}

export function Section({ title, description, children }: SectionProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: DUR.fast, ease: EASE }}
      className="rounded-2xl border border-white/10 bg-white/[0.02] p-5"
    >
      <header className="mb-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          {title}
        </h3>
        {description && (
          <p className="mt-1 text-xs text-slate-500">{description}</p>
        )}
      </header>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {children}
      </div>
    </motion.section>
  );
}

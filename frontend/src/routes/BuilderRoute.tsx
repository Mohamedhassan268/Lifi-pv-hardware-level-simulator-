/**
 * BuilderRoute — replaces the old BuildRoute + WorkbenchRoute.
 * Thin wrapper that renders BuilderWorkspace with the route-level enter
 * animation expected by App.tsx's AnimatePresence.
 */

import { motion } from "framer-motion";

import { BuilderWorkspace } from "@/features/builder/BuilderWorkspace";
import { DUR, EASE } from "@/lib/motion";

export function BuilderRoute() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: DUR.base, ease: EASE }}
    >
      <BuilderWorkspace />
    </motion.div>
  );
}

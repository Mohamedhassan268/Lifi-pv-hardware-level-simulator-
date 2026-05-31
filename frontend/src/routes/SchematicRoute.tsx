/**
 * SchematicRoute — route wrapper for the drag-and-drop circuit editor,
 * matching the enter/exit animation other routes use under App.tsx's
 * AnimatePresence.
 */

import { motion } from "framer-motion";

import { SchematicWorkspace } from "@/features/schematic/SchematicWorkspace";
import { DUR, EASE } from "@/lib/motion";

export function SchematicRoute() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: DUR.base, ease: EASE }}
    >
      <SchematicWorkspace />
    </motion.div>
  );
}

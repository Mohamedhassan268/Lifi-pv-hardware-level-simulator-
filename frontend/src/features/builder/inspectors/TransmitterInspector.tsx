/**
 * TransmitterInspector — reuses the existing TransmitterTab + ModulationTab
 * form bodies from features/build. Modulation lives here per the original
 * design choice (no separate Modulation entity).
 */

import { ModulationTab } from "@/features/build/ModulationTab";
import { TransmitterTab } from "@/features/build/TransmitterTab";

export function TransmitterInspector() {
  return (
    <div className="space-y-6">
      <TransmitterTab />
      <div className="border-t border-white/10 pt-6">
        <ModulationTab />
      </div>
    </div>
  );
}

/**
 * TransmitterModal — TX category, including Modulation per the design choice
 * to fold modulation into the Transmitter modal rather than have a 5th
 * category button.
 *
 * Reuses the existing TransmitterTab + ModulationTab form bodies — both
 * already write to the draft slice via useDraftField, so the ModalShell's
 * Apply button picks up their edits transparently.
 */

import { ModulationTab } from "@/features/build/ModulationTab";
import { TransmitterTab } from "@/features/build/TransmitterTab";
import { ModalShell } from "@/features/builder/ModalShell";

export function TransmitterModal() {
  return (
    <ModalShell
      category="transmitter"
      title="Transmitter"
      subtitle="LED + driver + modulation. Apply to push changes into the schematic."
    >
      <div className="space-y-6">
        <TransmitterTab />
        <div className="border-t border-white/10 pt-6">
          <ModulationTab />
        </div>
      </div>
    </ModalShell>
  );
}

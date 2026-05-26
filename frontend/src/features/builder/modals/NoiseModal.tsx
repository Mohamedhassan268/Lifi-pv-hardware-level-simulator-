import { NoiseTab } from "@/features/build/NoiseTab";
import { ModalShell } from "@/features/builder/ModalShell";

export function NoiseModal() {
  return (
    <ModalShell
      category="noise"
      title="Noise"
      subtitle="6-source physical noise model: shot, thermal, ambient, amplifier, ADC, processing."
    >
      <NoiseTab />
    </ModalShell>
  );
}

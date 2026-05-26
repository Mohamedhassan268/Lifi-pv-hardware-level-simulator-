import { ReceiverTab } from "@/features/build/ReceiverTab";
import { ModalShell } from "@/features/builder/ModalShell";

export function ReceiverModal() {
  return (
    <ModalShell
      category="receiver"
      title="Receiver"
      subtitle="PV cell, amplifier chain, comparator, ADC."
    >
      <ReceiverTab />
    </ModalShell>
  );
}

import { ChannelTab } from "@/features/build/ChannelTab";
import { ModalShell } from "@/features/builder/ModalShell";

export function GeometryModal() {
  return (
    <ModalShell
      category="geometry"
      title="Geometry"
      subtitle="TX/RX distance, angles, room dimensions and reflections."
    >
      <ChannelTab />
    </ModalShell>
  );
}

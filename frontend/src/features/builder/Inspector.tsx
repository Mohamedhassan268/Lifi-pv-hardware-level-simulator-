/**
 * Inspector — docked right-side panel that replaces the slide-out ModalShell.
 *
 * Driven entirely by `builderUIStore.selectedEntity`. The same draft-validate-
 * commit lifecycle ModalShell ran on open/close is now triggered by selection
 * change: on mount the inspector seeds a draft; on Apply it validates, commits,
 * bumps the schematic, and marks the entity configured. Revert just discards.
 *
 * The inspector never overlays the canvas — it sits beside it. Selection
 * swaps content in place; no exit animation, no scrim.
 */

import { useEffect, useState } from "react";

import { api } from "@/api/client";
import { NoiseInspector } from "@/features/builder/inspectors/NoiseInspector";
import { TransmitterInspector } from "@/features/builder/inspectors/TransmitterInspector";
import { ReceiverInspector } from "@/features/builder/inspectors/ReceiverInspector";
import { GeometryInspector } from "@/features/builder/inspectors/GeometryInspector";
import { Button } from "@/primitives/Button";
import { useBuilderUIStore, type BuilderCategory } from "@/store/builderUIStore";
import { useConfigStore } from "@/store/configStore";

const TITLES: Record<BuilderCategory, { title: string; subtitle: string }> = {
  transmitter: {
    title: "Transmitter",
    subtitle: "LED · driver · modulation",
  },
  receiver: {
    title: "Receiver",
    subtitle: "PV cell · amplifier · ADC",
  },
  geometry: {
    title: "Channel · Geometry",
    subtitle: "Distance · angles · room",
  },
  noise: {
    title: "Noise",
    subtitle: "6-source physical model",
  },
};

export function Inspector() {
  const selected = useBuilderUIStore((s) => s.selectedEntity);
  const bumpSchematic = useBuilderUIStore((s) => s.bumpSchematic);
  const markConfigured = useBuilderUIStore((s) => s.markConfigured);

  const draft = useConfigStore((s) => s.draft);
  const config = useConfigStore((s) => s.config);
  const startDraft = useConfigStore((s) => s.startDraft);
  const commitDraft = useConfigStore((s) => s.commitDraft);
  const discardDraft = useConfigStore((s) => s.discardDraft);

  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ensure a draft exists whenever the inspector has a selection.
  useEffect(() => {
    if (selected && !draft) startDraft();
  }, [selected, draft, startDraft]);

  // Discard error when selection changes.
  useEffect(() => {
    setError(null);
  }, [selected]);

  if (!selected) {
    return (
      <aside className="flex h-full flex-col border-l border-hair bg-slate-950">
        <header className="border-b border-hair px-3 py-2">
          <p className="label-inst">Inspector</p>
        </header>
        <div className="flex-1 px-3 py-3 text-[11px] text-slate-500">
          No selection. Click a block in the schematic.
        </div>
      </aside>
    );
  }

  const meta = TITLES[selected];
  const changedCount = countChanges(config, draft);

  const handleApply = async () => {
    if (!draft || applying) return;
    setApplying(true);
    setError(null);
    try {
      const r = await api.validateConfig(draft);
      if (!r.valid) {
        setError(
          r.errors
            .filter((e) => e.trim() && e !== "Invalid SystemConfig:")
            .join("; ") || "Validation failed",
        );
        setApplying(false);
        return;
      }
      commitDraft();
      markConfigured(selected);
      bumpSchematic();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplying(false);
    }
  };

  const handleRevert = () => {
    discardDraft();
    setError(null);
  };

  return (
    <aside
      key={selected}
      className="flex h-full flex-col border-l border-hair bg-slate-950"
    >
      <header className="flex items-center justify-between gap-2 border-b border-hair px-3 py-2">
        <div className="min-w-0">
          <p className="label-inst truncate">Inspector · {meta.title}</p>
          <p className="mt-0.5 truncate text-[10px] text-slate-500">{meta.subtitle}</p>
        </div>
        {changedCount > 0 && (
          <span className="shrink-0 border border-harvest-400/40 bg-harvest-400/10 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-harvest-300 readout">
            {changedCount} unsaved
          </span>
        )}
      </header>

      <div className="flex-1 overflow-y-auto px-3 py-3">
        {selected === "noise" && <NoiseInspector />}
        {selected === "transmitter" && <TransmitterInspector />}
        {selected === "receiver" && <ReceiverInspector />}
        {selected === "geometry" && <GeometryInspector />}
      </div>

      <footer className="flex flex-col gap-1 border-t border-hair px-3 py-2">
        {error && <p className="readout text-[11px] text-rose-400">{error}</p>}
        <div className="flex items-center justify-between gap-2">
          <span className="label-inst">
            {applying ? "validating…" : changedCount === 0 ? "in sync" : "modified"}
          </span>
          <div className="flex gap-1">
            <Button
              variant="ghost"
              onClick={handleRevert}
              disabled={applying || changedCount === 0}
            >
              Revert
            </Button>
            <Button onClick={handleApply} disabled={applying || changedCount === 0}>
              {applying ? "…" : `Apply${changedCount ? ` ${changedCount}` : ""}`}
            </Button>
          </div>
        </div>
      </footer>
    </aside>
  );
}

function countChanges(
  live: Record<string, unknown>,
  draft: Record<string, unknown> | null,
): number {
  if (!draft) return 0;
  let n = 0;
  for (const k of Object.keys(draft)) {
    if (live[k] !== draft[k]) n++;
  }
  return n;
}

/**
 * BuilderWorkspace — 3-column workspace per the paradigm-B redesign.
 *
 *   ┌────────────┬──────────────────────────────────┬─────────────┐
 *   │ Category   │  Live schematic (react-flow)     │  Inspector  │
 *   │  rail      │  + Results mini-panel            │  (docked,   │
 *   │            │                                  │   per-entity│
 *   │ [Simulate] │                                  │   sections) │
 *   └────────────┴──────────────────────────────────┴─────────────┘
 *
 * The Inspector reads builderUIStore.selectedEntity and renders the matching
 * panel. It owns the draft → validate → commit lifecycle previously held by
 * ModalShell. The slide-out modals remain mounted during the transitional
 * period so the legacy CategoryRail buttons still work; they will be removed
 * once the ProjectTree replaces CategoryRail.
 */

import { CategoryRail } from "@/features/builder/CategoryRail";
import { DiagnosticsBar } from "@/features/builder/DiagnosticsBar";
import { Inspector } from "@/features/builder/Inspector";
import { ResultsMiniPanel } from "@/features/builder/ResultsMiniPanel";
import { SchematicCanvas } from "@/features/builder/SchematicCanvas";
import { PipelineTransport } from "@/features/probes/PipelineTransport";
import { ProbeDock } from "@/features/probes/ProbeDock";

export function BuilderWorkspace() {
  // Heights: TopBar ≈ 36px, DiagnosticsBar ≈ 34px → leaves ~calc(100vh - 70px)
  return (
    <div className="flex h-[calc(100vh-2.25rem)] flex-col">
      <div className="grid min-h-0 flex-1 grid-cols-[220px_1fr_340px]">
        <CategoryRail />
        <section className="relative flex min-w-0 flex-col">
          <div className="border-b border-slate-800/80 bg-slate-950/40 p-2">
            <PipelineTransport />
          </div>
          <div className="relative min-h-0 flex-1">
            <SchematicCanvas />
            <ResultsMiniPanel />
          </div>
          <ProbeDock />
        </section>
        <Inspector />
      </div>
      <DiagnosticsBar />
    </div>
  );
}

/**
 * InspectorRoute — full-screen Inspector with four tabs:
 *   - Schematics  (KiCad download for now; in-page render coming next)
 *   - Validation  (full structured view of cosim.validation issues)
 *   - Messages    (live WS event log)
 *   - Probes      (signal stats + trace from the last simulation run)
 *
 * Mounted via uiStore route === "inspector"; reached from the top bar.
 */

import { useState } from "react";

import {
  InspectorTabs,
  type InspectorTabId,
} from "@/features/inspector/InspectorTabs";
import { MessagesTab } from "@/features/inspector/MessagesTab";
import { ProbesTab } from "@/features/inspector/ProbesTab";
import { SchematicsTab } from "@/features/inspector/SchematicsTab";
import { ValidationTab } from "@/features/inspector/ValidationTab";
import { Button } from "@/primitives/Button";
import { useMessagesStore } from "@/store/messagesStore";
import { useResultsStore } from "@/store/resultsStore";
import { useUIStore } from "@/store/uiStore";

export function InspectorRoute() {
  const [tab, setTab] = useState<InspectorTabId>("validation");
  const setRoute = useUIStore((s) => s.setRoute);

  const messagesCount = useMessagesStore((s) => s.entries.length);
  const hasWaveforms = useResultsStore((s) => s.waveforms !== null);

  return (
    <div className="mx-auto w-full max-w-6xl space-y-4 px-6 py-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-100">Inspector</h1>
          <p className="text-sm text-slate-400">
            Cross-cutting view of the current run: schematic, validation,
            event log, signal probes.
          </p>
        </div>
        <Button variant="ghost" onClick={() => setRoute("builder")}>
          ← Back to Builder
        </Button>
      </div>

      <InspectorTabs
        active={tab}
        onSelect={setTab}
        tabs={[
          { id: "schematics", label: "Schematics" },
          { id: "validation", label: "Validation" },
          {
            id: "messages",
            label: "Messages",
            badge: messagesCount > 0 ? messagesCount : undefined,
          },
          {
            id: "probes",
            label: "Probes",
            badge: hasWaveforms ? undefined : "—",
          },
        ]}
      />

      <div className="pt-2">
        {tab === "schematics" && <SchematicsTab />}
        {tab === "validation" && <ValidationTab />}
        {tab === "messages" && <MessagesTab />}
        {tab === "probes" && <ProbesTab />}
      </div>
    </div>
  );
}

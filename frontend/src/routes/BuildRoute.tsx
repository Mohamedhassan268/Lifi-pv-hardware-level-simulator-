/**
 * BuildRoute — the "Build my own system" tabbed editor. Five tabs
 * (Channel / Transmitter / Receiver / Modulation & Sim / Noise) plus a
 * persistent right rail with live link budget and validation. Edits go
 * to the draft slice in configStore; Apply commits, Cancel discards.
 *
 * Lifecycle:
 *   - On mount, if no draft exists, seed one from the live config.
 *   - On unmount via Cancel, discardDraft().
 *   - On Apply, commitDraft() then route to /setup.
 */

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

import { api } from "@/api/client";
import { BuilderRail } from "@/features/build/BuilderRail";
import { BuilderTabs, type BuilderTabKey } from "@/features/build/BuilderTabs";
import { ChannelTab } from "@/features/build/ChannelTab";
import { ModulationTab } from "@/features/build/ModulationTab";
import { NoiseTab } from "@/features/build/NoiseTab";
import { ReceiverTab } from "@/features/build/ReceiverTab";
import { TransmitterTab } from "@/features/build/TransmitterTab";
import { DUR, EASE } from "@/lib/motion";
import { Button } from "@/primitives/Button";
import { useConfigStore } from "@/store/configStore";
import { useUIStore } from "@/store/uiStore";

export function BuildRoute() {
  const [tab, setTab] = useState<BuilderTabKey>("channel");
  const [applyError, setApplyError] = useState<string | null>(null);
  const [applying, setApplying] = useState(false);
  const setRoute = useUIStore((s) => s.setRoute);
  const draft = useConfigStore((s) => s.draft);
  const startDraft = useConfigStore((s) => s.startDraft);
  const commitDraft = useConfigStore((s) => s.commitDraft);
  const discardDraft = useConfigStore((s) => s.discardDraft);

  // Seed a draft on first mount if none exists.
  useEffect(() => {
    if (!draft) startDraft();
  }, [draft, startDraft]);

  const cancel = () => {
    discardDraft();
    setRoute("setup");
  };
  const apply = async () => {
    if (!draft || applying) return;
    setApplying(true);
    setApplyError(null);
    // Do a fresh validation right now so we don't rely on the debounced
    // rail state. If valid, commit and route; if not, surface the errors.
    try {
      const r = await api.validateConfig(draft);
      if (!r.valid) {
        setApplyError(
          r.errors.filter((e) => e.trim() && e !== "Invalid SystemConfig:").join("; "),
        );
        setApplying(false);
        return;
      }
      commitDraft();
      setRoute("setup");
    } catch (e) {
      setApplyError(e instanceof Error ? e.message : String(e));
      setApplying(false);
    }
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: DUR.base, ease: EASE }}
      className="mx-auto max-w-7xl px-8 py-10"
    >
      <header className="mb-6 flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Build new system</h1>
          <p className="text-sm text-slate-400">
            Pick a topology + modulation and tune the circuit. Changes preview
            live on the right; nothing is committed until you Apply.
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex gap-2">
            <Button variant="ghost" onClick={cancel}>Cancel</Button>
            <Button onClick={apply} disabled={applying}>
              {applying ? "Applying…" : "Apply & continue"}
            </Button>
          </div>
          {applyError && (
            <p className="max-w-sm text-right text-xs text-rose-400">
              {applyError}
            </p>
          )}
        </div>
      </header>

      <BuilderTabs active={tab} onChange={setTab} />

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="lg:col-span-8">
          <AnimatePresence mode="wait">
            {tab === "channel" && (
              <motion.div key="channel"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: DUR.fast, ease: EASE }}
              ><ChannelTab /></motion.div>
            )}
            {tab === "transmitter" && (
              <motion.div key="transmitter"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: DUR.fast, ease: EASE }}
              ><TransmitterTab /></motion.div>
            )}
            {tab === "receiver" && (
              <motion.div key="receiver"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: DUR.fast, ease: EASE }}
              ><ReceiverTab /></motion.div>
            )}
            {tab === "modulation" && (
              <motion.div key="modulation"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: DUR.fast, ease: EASE }}
              ><ModulationTab /></motion.div>
            )}
            {tab === "noise" && (
              <motion.div key="noise"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: DUR.fast, ease: EASE }}
              ><NoiseTab /></motion.div>
            )}
          </AnimatePresence>
        </div>
        <aside className="lg:col-span-4 lg:sticky lg:top-20 lg:self-start">
          <BuilderRail />
        </aside>
      </div>
    </motion.section>
  );
}

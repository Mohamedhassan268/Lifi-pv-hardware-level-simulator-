/**
 * StandardsModal — pick a PHY compliance profile for the next Simulate.
 *
 * Opening the modal:
 *   - GET /api/standards (cached for the session)
 *   - shows each profile with its spec section + notes + overrides table
 *
 * On select:
 *   - configStore.patch(profile.overrides) so the live config reflects the spec
 *   - standardsStore.setActive(id, label)
 *   - close modal
 *
 * "Clear active profile" returns to free design (no overrides reverted —
 * the user can do that manually if needed).
 */

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

import { api, type StandardsProfileSummary } from "@/api/client";
import { DUR, EASE } from "@/lib/motion";
import { Button } from "@/primitives/Button";
import { useConfigStore } from "@/store/configStore";
import { useStandardsStore } from "@/store/standardsStore";

interface StandardsModalProps {
  open: boolean;
  onClose: () => void;
}

export function StandardsModal({ open, onClose }: StandardsModalProps) {
  const patch = useConfigStore((s) => s.patch);
  const setActive = useStandardsStore((s) => s.setActive);
  const activeId = useStandardsStore((s) => s.activeProfileId);

  const [profiles, setProfiles] = useState<StandardsProfileSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || profiles !== null) return;
    setError(null);
    api
      .listStandards()
      .then(setProfiles)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [open, profiles]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const onPick = (p: StandardsProfileSummary) => {
    patch(p.overrides);
    setActive(p.id, p.label);
    onClose();
  };

  const clearActive = () => {
    setActive(null, null);
    onClose();
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="scrim"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: DUR.fast, ease: EASE }}
            onClick={onClose}
            className="fixed inset-0 z-30 bg-slate-950/70 backdrop-blur-sm"
          />
          <motion.div
            key="panel"
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.97 }}
            transition={{ duration: DUR.base, ease: EASE }}
            className="fixed left-1/2 top-1/2 z-40 w-[min(680px,calc(100vw-2rem))]
                       -translate-x-1/2 -translate-y-1/2 rounded-2xl border
                       border-white/10 bg-slate-950/95 shadow-2xl backdrop-blur"
            role="dialog"
            aria-modal="true"
          >
            <header className="flex items-start justify-between gap-4 border-b border-white/10 p-6">
              <div>
                <h2 className="text-lg font-semibold">Standards compliance</h2>
                <p className="mt-1 text-xs text-slate-400">
                  Pinning a profile patches the live config to the spec values
                  and runs PASS/FAIL checks after each simulation.
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg p-1 text-slate-500 hover:bg-white/[0.06] hover:text-slate-200"
                aria-label="Close"
              >
                ✕
              </button>
            </header>

            <div className="max-h-[60vh] space-y-2 overflow-y-auto p-6 pt-4">
              {error && <p className="text-xs text-rose-400">{error}</p>}
              {!error && profiles === null && (
                <p className="text-xs text-slate-500">Loading profiles…</p>
              )}
              {profiles?.length === 0 && (
                <p className="text-xs text-slate-500">
                  No profiles registered on the backend.
                </p>
              )}
              {profiles?.map((p) => (
                <ProfileCard
                  key={p.id}
                  profile={p}
                  active={p.id === activeId}
                  onPick={() => onPick(p)}
                />
              ))}
            </div>

            <footer className="flex items-center justify-between border-t border-white/10 px-6 py-4">
              <Button variant="ghost" onClick={clearActive} disabled={!activeId}>
                Clear active profile
              </Button>
              <span className="text-[11px] text-slate-500">
                Tip: edit fields inside a category modal to depart from the spec.
              </span>
            </footer>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

function ProfileCard({
  profile,
  active,
  onPick,
}: {
  profile: StandardsProfileSummary;
  active: boolean;
  onPick: () => void;
}) {
  const overrides = Object.entries(profile.overrides);
  return (
    <button
      type="button"
      onClick={onPick}
      className={`flex w-full flex-col items-start gap-2 rounded-xl border px-4 py-3 text-left transition-colors ${
        active
          ? "border-beam-400/60 bg-beam-400/10"
          : "border-white/10 bg-white/[0.02] hover:border-beam-400/30 hover:bg-white/[0.05]"
      }`}
    >
      <div className="flex w-full items-center justify-between gap-2">
        <span className="text-sm font-semibold text-slate-100">{profile.label}</span>
        <span className="text-[10px] uppercase tracking-wider text-slate-500">
          {profile.spec_section}
        </span>
      </div>
      {profile.notes && <p className="text-xs text-slate-400">{profile.notes}</p>}
      {overrides.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {overrides.map(([k, v]) => (
            <span
              key={k}
              className="rounded-md border border-white/10 bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-slate-300"
            >
              {k} = {String(v)}
            </span>
          ))}
        </div>
      )}
      {active && (
        <span className="text-[10px] uppercase tracking-wider text-beam-300">
          ✓ active
        </span>
      )}
    </button>
  );
}

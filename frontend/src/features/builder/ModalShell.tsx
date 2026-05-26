/**
 * ModalShell — slide-out right-side panel used by all 4 category modals.
 *
 * Handles the draft→commit lifecycle so each category modal body only needs
 * to render form fields:
 *
 *   - On open, configStore.startDraft() seeds a draft from the live config.
 *   - Form fields edit the draft via useDraftField (unchanged from before).
 *   - Apply  -> POST /api/config/validate, then commitDraft()
 *              -> builderUIStore.bumpSchematic() so the canvas re-renders
 *              -> close modal.
 *   - Cancel -> discardDraft() -> close modal (live config untouched).
 *   - Escape -> same as Cancel.
 *
 * The Apply button is the single tie between modal state and the schematic:
 * everything else reads from the live config.
 */

import { AnimatePresence, motion } from "framer-motion";
import { type ReactNode, useEffect, useState } from "react";

import { api } from "@/api/client";
import { DUR, EASE } from "@/lib/motion";
import { Button } from "@/primitives/Button";
import { useBuilderUIStore, type BuilderCategory } from "@/store/builderUIStore";
import { useConfigStore } from "@/store/configStore";

interface ModalShellProps {
  category: BuilderCategory;
  title: string;
  subtitle?: string;
  children: ReactNode;
}

export function ModalShell({ category, title, subtitle, children }: ModalShellProps) {
  const openCategory = useBuilderUIStore((s) => s.openCategory);
  const closeModal = useBuilderUIStore((s) => s.closeModal);
  const bumpSchematic = useBuilderUIStore((s) => s.bumpSchematic);
  const markConfigured = useBuilderUIStore((s) => s.markConfigured);

  const draft = useConfigStore((s) => s.draft);
  const startDraft = useConfigStore((s) => s.startDraft);
  const commitDraft = useConfigStore((s) => s.commitDraft);
  const discardDraft = useConfigStore((s) => s.discardDraft);

  const isOpen = openCategory === category;
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Seed a fresh draft each time the modal opens for this category.
  useEffect(() => {
    if (isOpen && !draft) startDraft();
  }, [isOpen, draft, startDraft]);

  // Escape closes (treated as Cancel).
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  const handleCancel = () => {
    discardDraft();
    setError(null);
    closeModal();
  };

  const handleApply = async () => {
    if (!draft || applying) return;
    setApplying(true);
    setError(null);
    try {
      const r = await api.validateConfig(draft);
      if (!r.valid) {
        setError(
          r.errors.filter((e) => e.trim() && e !== "Invalid SystemConfig:").join("; ") ||
            "Validation failed",
        );
        setApplying(false);
        return;
      }
      commitDraft();
      markConfigured(category);
      bumpSchematic();
      closeModal();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplying(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            key="scrim"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: DUR.fast, ease: EASE }}
            onClick={handleCancel}
            className="fixed inset-0 z-30 bg-slate-950/60 backdrop-blur-sm"
          />
          <motion.aside
            key="panel"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ duration: DUR.base, ease: EASE }}
            className="fixed right-0 top-0 z-40 flex h-screen w-full max-w-xl flex-col
                       border-l border-white/10 bg-slate-950/95 shadow-2xl backdrop-blur"
          >
            <header className="flex items-start justify-between gap-4 border-b border-white/10 px-6 py-5">
              <div>
                <h2 className="text-lg font-semibold">{title}</h2>
                {subtitle && (
                  <p className="mt-1 text-xs text-slate-400">{subtitle}</p>
                )}
              </div>
              <button
                type="button"
                onClick={handleCancel}
                className="rounded-lg p-1 text-slate-500 hover:bg-white/[0.06] hover:text-slate-200"
                aria-label="Close"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
                </svg>
              </button>
            </header>

            <div className="flex-1 overflow-y-auto px-6 py-5">{children}</div>

            <footer className="flex flex-col gap-2 border-t border-white/10 px-6 py-4">
              {error && (
                <p className="text-xs text-rose-400">{error}</p>
              )}
              <div className="flex justify-end gap-2">
                <Button variant="ghost" onClick={handleCancel} disabled={applying}>
                  Cancel
                </Button>
                <Button onClick={handleApply} disabled={applying || !draft}>
                  {applying ? "Applying…" : "Apply"}
                </Button>
              </div>
            </footer>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

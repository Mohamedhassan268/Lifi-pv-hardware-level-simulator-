/**
 * LaunchChoiceModal — the ONLY place the preset-vs-own-design choice
 * appears. Triggered exclusively from the LandingRoute's Builder CTA.
 *
 * Flow:
 *   Step 1 (chooser):
 *     [Use a preset]              → advances to Step 2
 *     [Build your own system]     → resets workspace, opens builder
 *
 *   Step 2 (preset picker):
 *     Loads preset → markAllConfigured() (so schematic blocks all light up)
 *     → opens builder.
 *
 *   Closing the modal (scrim / Escape / Back) leaves the user on the
 *   landing page.
 */

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

import { api } from "@/api/client";
import { DUR, EASE } from "@/lib/motion";
import { Button } from "@/primitives/Button";
import { useBuilderUIStore } from "@/store/builderUIStore";
import { useConfigStore } from "@/store/configStore";
import { useUIStore } from "@/store/uiStore";

const PRESET_BLURBS: Record<string, string> = {
  kadirvelu2021: "Kadirvelu et al. (2021) — 5 kbps OOK Manchester PV link",
  correa2025: "Correa et al. (2025) — PWM-ASK greenhouse downlink",
  gonzalez2024: "González et al. (2024) — Manchester OOK lab demo",
  fakidis2020: "Fakidis et al. (2020) — OOK underwater short-range",
  sarwar2017: "Sarwar et al. (2017) — DCO-OFDM indoor LiFi",
  oliveira2024: "Oliveira et al. (2024) — OFDM solar-cell receiver",
  xu2024: "Xu et al. (2024) — BFSK ambient-light backscatter",
  lifi_poc_breadboard: "LiFi PoC breadboard — practical bench setup",
};

type Step = "choose" | "preset";

interface LaunchChoiceModalProps {
  open: boolean;
  onClose: () => void;
}

export function LaunchChoiceModal({ open, onClose }: LaunchChoiceModalProps) {
  const loadPreset = useConfigStore((s) => s.loadPreset);
  const resetWorkspace = useBuilderUIStore((s) => s.resetWorkspace);
  const markAllConfigured = useBuilderUIStore((s) => s.markAllConfigured);
  const setRoute = useUIStore((s) => s.setRoute);

  const [step, setStep] = useState<Step>("choose");
  const [presets, setPresets] = useState<string[] | null>(null);
  const [presetsError, setPresetsError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Reset to step 1 every time the modal opens.
  useEffect(() => {
    if (open) {
      setStep("choose");
      setPresetsError(null);
    }
  }, [open]);

  // Lazy-load preset list on demand.
  useEffect(() => {
    if (step !== "preset" || presets !== null) return;
    setPresetsError(null);
    api
      .listPresets()
      .then((p) => setPresets(p))
      .catch((e) =>
        setPresetsError(e instanceof Error ? e.message : String(e)),
      );
  }, [step, presets]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const onChooseOwn = () => {
    resetWorkspace();
    onClose();
    setRoute("builder");
  };

  const onPickPreset = async (name: string) => {
    if (loading) return;
    setLoading(true);
    try {
      await loadPreset(name);
      markAllConfigured();
      onClose();
      setRoute("builder");
    } finally {
      setLoading(false);
    }
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
            className="fixed left-1/2 top-1/2 z-40 w-[min(560px,calc(100vw-2rem))]
                       -translate-x-1/2 -translate-y-1/2 rounded-2xl border
                       border-white/10 bg-slate-950/95 shadow-2xl backdrop-blur"
            role="dialog"
            aria-modal="true"
            aria-labelledby="launch-choice-title"
          >
            {step === "choose" ? (
              <ChooseStep
                onPreset={() => setStep("preset")}
                onOwn={onChooseOwn}
                onClose={onClose}
              />
            ) : (
              <PresetStep
                presets={presets}
                error={presetsError}
                loading={loading}
                onBack={() => setStep("choose")}
                onPick={onPickPreset}
                onClose={onClose}
              />
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

function ChooseStep({
  onPreset,
  onOwn,
  onClose,
}: {
  onPreset: () => void;
  onOwn: () => void;
  onClose: () => void;
}) {
  return (
    <div className="p-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 id="launch-choice-title" className="text-lg font-semibold">
            How would you like to start?
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Load a published research configuration, or build a system from scratch.
          </p>
        </div>
        <CloseButton onClick={onClose} />
      </header>

      <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <GateCard
          accent="beam"
          title="Use a preset"
          body="Reproduce a published paper's setup. All four categories are populated and lit up immediately."
          cta="Browse presets →"
          onClick={onPreset}
        />
        <GateCard
          accent="harvest"
          title="Build your own system"
          body="Start with an empty workspace and configure Transmitter, Receiver, Geometry, and Noise yourself."
          cta="Open empty builder →"
          onClick={onOwn}
        />
      </div>
    </div>
  );
}

function PresetStep({
  presets,
  error,
  loading,
  onBack,
  onPick,
  onClose,
}: {
  presets: string[] | null;
  error: string | null;
  loading: boolean;
  onBack: () => void;
  onPick: (name: string) => void;
  onClose: () => void;
}) {
  return (
    <div className="p-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 id="launch-choice-title" className="text-lg font-semibold">
            Pick a preset
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Each preset reproduces a published paper's experimental setup.
          </p>
        </div>
        <CloseButton onClick={onClose} />
      </header>

      <div className="mt-5 max-h-[50vh] space-y-2 overflow-y-auto pr-1">
        {error && <p className="text-xs text-rose-400">{error}</p>}
        {!error && presets === null && (
          <p className="text-xs text-slate-500">Loading presets…</p>
        )}
        {presets?.length === 0 && (
          <p className="text-xs text-slate-500">No presets found on the backend.</p>
        )}
        {presets?.map((name) => (
          <button
            key={name}
            type="button"
            disabled={loading}
            onClick={() => onPick(name)}
            className="group flex w-full flex-col items-start gap-1 rounded-xl border
                       border-white/10 bg-white/[0.02] px-4 py-3 text-left
                       hover:border-beam-400/40 hover:bg-white/[0.05]
                       disabled:cursor-wait disabled:opacity-50"
          >
            <span className="text-sm font-semibold text-slate-100">{name}</span>
            <span className="text-xs text-slate-400">
              {PRESET_BLURBS[name] ?? "Published configuration"}
            </span>
          </button>
        ))}
      </div>

      <footer className="mt-4 flex justify-between">
        <Button variant="ghost" onClick={onBack} disabled={loading}>
          ← Back
        </Button>
        <span className="self-center text-[11px] text-slate-500">
          {loading ? "Loading preset…" : "Click a preset to load it into the builder"}
        </span>
      </footer>
    </div>
  );
}

function GateCard({
  accent,
  title,
  body,
  cta,
  onClick,
}: {
  accent: "beam" | "harvest";
  title: string;
  body: string;
  cta: string;
  onClick: () => void;
}) {
  const ring =
    accent === "beam"
      ? "hover:border-beam-400/60 hover:shadow-[0_0_36px_-12px_rgba(34,211,238,0.5)]"
      : "hover:border-harvest-400/60 hover:shadow-[0_0_36px_-12px_rgba(251,191,36,0.5)]";
  const ctaColor = accent === "beam" ? "text-beam-300" : "text-harvest-300";
  return (
    <motion.button
      type="button"
      onClick={onClick}
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.99 }}
      transition={{ duration: DUR.fast, ease: EASE }}
      className={`flex h-full flex-col items-start gap-2 rounded-xl border border-white/10
                  bg-white/[0.02] p-4 text-left transition-colors ${ring}`}
    >
      <span className="text-sm font-semibold text-slate-100">{title}</span>
      <span className="text-xs leading-relaxed text-slate-400">{body}</span>
      <span className={`mt-auto pt-2 text-xs font-medium ${ctaColor}`}>{cta}</span>
    </motion.button>
  );
}

function CloseButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-lg p-1 text-slate-500 hover:bg-white/[0.06] hover:text-slate-200"
      aria-label="Close"
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
      </svg>
    </button>
  );
}

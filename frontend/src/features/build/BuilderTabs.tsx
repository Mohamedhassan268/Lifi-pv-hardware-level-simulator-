/**
 * BuilderTabs — controlled tab strip with a shared `layoutId` indicator so
 * switching tabs animates the underline via transform only (no reflow).
 */

import { motion } from "framer-motion";

import { DUR, EASE } from "@/lib/motion";

export type BuilderTabKey =
  | "channel"
  | "transmitter"
  | "receiver"
  | "modulation"
  | "noise";

const TABS: Array<{ key: BuilderTabKey; label: string }> = [
  { key: "channel", label: "Channel" },
  { key: "transmitter", label: "Transmitter" },
  { key: "receiver", label: "Receiver" },
  { key: "modulation", label: "Modulation & Sim" },
  { key: "noise", label: "Noise" },
];

interface BuilderTabsProps {
  active: BuilderTabKey;
  onChange: (key: BuilderTabKey) => void;
}

export function BuilderTabs({ active, onChange }: BuilderTabsProps) {
  return (
    <div
      role="tablist"
      className="flex flex-wrap gap-1 rounded-2xl border border-white/10 bg-white/[0.03] p-1"
    >
      {TABS.map((t) => {
        const isActive = t.key === active;
        return (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(t.key)}
            className={
              "relative rounded-xl px-4 py-2 text-sm transition-colors " +
              (isActive ? "text-slate-100" : "text-slate-400 hover:text-slate-200")
            }
          >
            {isActive && (
              <motion.span
                layoutId="builder-tab-indicator"
                className="absolute inset-0 rounded-xl bg-beam-400/[0.18]"
                transition={{ duration: DUR.base, ease: EASE }}
              />
            )}
            <span className="relative">{t.label}</span>
          </button>
        );
      })}
    </div>
  );
}

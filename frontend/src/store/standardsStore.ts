/**
 * standardsStore — which PHY profile (if any) is active for the next
 * Simulate, plus the most-recent compliance result.
 *
 * Active profile is sent on the /ws/pipeline `start` payload as
 * `profile_id`; the backend returns a `compliance` message after metrics.
 */

import { create } from "zustand";

export interface CriterionResult {
  label: string;
  ok: boolean;
  actual: number | null;
  bound: number | null;
  detail: string;
}

export interface ComplianceResult {
  profile_id: string;
  profile_label: string;
  spec_section: string;
  passed: boolean;
  results: CriterionResult[];
}

interface StandardsState {
  activeProfileId: string | null;
  activeProfileLabel: string | null;
  lastResult: ComplianceResult | null;

  setActive: (id: string | null, label: string | null) => void;
  setResult: (r: ComplianceResult | null) => void;
  reset: () => void;
}

export const useStandardsStore = create<StandardsState>((set) => ({
  activeProfileId: null,
  activeProfileLabel: null,
  lastResult: null,

  setActive: (id, label) => set({ activeProfileId: id, activeProfileLabel: label }),
  setResult: (r) => set({ lastResult: r }),
  reset: () => set({ activeProfileId: null, activeProfileLabel: null, lastResult: null }),
}));

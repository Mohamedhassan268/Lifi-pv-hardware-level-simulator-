/**
 * useDraftField — one-line helper for Builder form rows.
 *
 *   const distance = useDraftField<number>("distance_m");
 *   <NumberField label="Distance" value={distance.value} onChange={distance.set} />
 *
 * Returns the current draft value (falling back to live config), plus a
 * setter that writes only to the draft via configStore.patchDraft.
 */

import { useConfigStore } from "@/store/configStore";

export function useDraftField<T>(field: string): {
  value: T | undefined;
  set: (v: T) => void;
} {
  const value = useConfigStore((s) => {
    const src = s.draft ?? s.config;
    return src[field] as T | undefined;
  });
  const patchDraft = useConfigStore((s) => s.patchDraft);
  return {
    value,
    set: (v: T) => patchDraft({ [field]: v as never }),
  };
}

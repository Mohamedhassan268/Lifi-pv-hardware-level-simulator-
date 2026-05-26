/**
 * useDebouncedValidate — pings POST /api/config/validate on a debounced cadence
 * whenever the watched config changes. Returns `{ valid, errors }` for the
 * BuilderRail to render. Errors come back as a list of strings (the backend
 * splits the multi-line ValueError message).
 */

import { useEffect, useState } from "react";

import { api, type ConfigDict } from "@/api/client";

interface ValidateResult {
  valid: boolean;
  errors: string[];
}

const IDLE: ValidateResult = { valid: true, errors: [] };

export function useDebouncedValidate(
  config: ConfigDict | null,
  delayMs = 220,
): ValidateResult {
  const [result, setResult] = useState<ValidateResult>(IDLE);

  useEffect(() => {
    if (!config) {
      setResult(IDLE);
      return;
    }
    let cancelled = false;
    const id = setTimeout(() => {
      api
        .validateConfig(config)
        .then((r) => {
          if (cancelled) return;
          setResult({ valid: r.valid, errors: r.errors ?? [] });
        })
        .catch((e) => {
          if (cancelled) return;
          setResult({
            valid: false,
            errors: [e instanceof Error ? e.message : String(e)],
          });
        });
    }, delayMs);
    return () => {
      cancelled = true;
      clearTimeout(id);
    };
  }, [config, delayMs]);

  return result;
}

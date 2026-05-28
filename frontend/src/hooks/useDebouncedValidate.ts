/**
 * useDebouncedValidate — pings POST /api/config/validate on a debounced cadence
 * whenever the watched config changes. Returns `{ valid, errors, issues }`.
 *
 * `errors` is the legacy list of ERROR-level messages (back-compat with
 * existing call sites). `issues` is the structured list from
 * cosim.validation: ERROR + WARNING + INFO, with field/rule_id/suggestion.
 * Consumers that want color-coding or inline placement read `issues`.
 */

import { useEffect, useState } from "react";

import {
  api,
  type ConfigDict,
  type ValidationIssue,
} from "@/api/client";

interface ValidateResult {
  valid: boolean;
  errors: string[];
  issues: ValidationIssue[];
}

const IDLE: ValidateResult = { valid: true, errors: [], issues: [] };

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
          setResult({
            valid: r.valid,
            errors: r.errors ?? [],
            issues: r.issues ?? [],
          });
        })
        .catch((e) => {
          if (cancelled) return;
          const msg = e instanceof Error ? e.message : String(e);
          setResult({
            valid: false,
            errors: [msg],
            issues: [
              {
                level: "error",
                field: "network",
                message: msg,
                rule_id: "client.validate_request_failed",
              },
            ],
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

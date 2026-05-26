/**
 * ProbeDock — tiles all open ProbeWindows in a column under the schematic.
 *
 * Also runs the boot-time fetch of `/api/probes` so the registry is available
 * the moment the user opens the Builder route. Cheap to mount multiple times —
 * the inner effect guards against duplicate fetches.
 */

import { useEffect } from "react";

import { getBackendBase } from "@/api/backend";
import { usePipelineExecutionStore } from "@/store/pipelineExecutionStore";
import type { ProbeSpec } from "@/types/probes";

import { ProbeWindow } from "./ProbeWindow";

let registryFetched = false;

export function ProbeDock() {
  const open = usePipelineExecutionStore((s) => s.openProbeWindows);
  const setRegistry = usePipelineExecutionStore((s) => s.setRegistry);

  useEffect(() => {
    if (registryFetched) return;
    registryFetched = true;
    (async () => {
      try {
        const res = await fetch(`${getBackendBase()}/api/probes`);
        if (!res.ok) return;
        const data = (await res.json()) as { probes: ProbeSpec[] };
        if (Array.isArray(data.probes)) setRegistry(data.probes);
      } catch {
        registryFetched = false; // allow retry
      }
    })();
  }, [setRegistry]);

  if (open.length === 0) return null;

  return (
    <div className="border-t border-slate-800 bg-slate-950/60 p-2">
      <div className="mb-2 text-[10px] uppercase tracking-[0.2em] text-slate-500">
        Virtual probes ({open.length})
      </div>
      <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
        {open.map((pid) => (
          <ProbeWindow key={pid} probeId={pid} />
        ))}
      </div>
    </div>
  );
}

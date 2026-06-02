/**
 * DuplexResultsPanel — per-system results for a two-system run, shown side by
 * side (System A above System B). Floating panel in the builder canvas, only
 * when a second system exists and a duplex run has produced/started results.
 */

import { useTwoSystem } from "@/store/configStore";
import { useDuplexStore, type SysMetrics } from "@/store/duplexStore";

export function DuplexResultsPanel() {
  const twoSystem = useTwoSystem();
  const running = useDuplexStore((s) => s.running);
  const current = useDuplexStore((s) => s.current);
  const results = useDuplexStore((s) => s.results);
  const error = useDuplexStore((s) => s.error);

  if (!twoSystem) return null;
  const hasContent = running || error || results.A || results.B;
  if (!hasContent) return null;

  return (
    <div
      className="pointer-events-auto absolute bottom-4 right-4 w-[250px] rounded-xl border
                 border-white/10 bg-slate-950/90 p-3 backdrop-blur
                 shadow-[0_8px_30px_-12px_rgba(0,0,0,0.7)]"
    >
      <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
        Duplex results
      </h3>
      {error && <p className="mb-2 text-xs text-rose-400">{error}</p>}
      <div className="space-y-2">
        <SysRow id="A" m={results.A} active={running && current === "A"} />
        <SysRow id="B" m={results.B} active={running && current === "B"} />
      </div>
    </div>
  );
}

function SysRow({ id, m, active }: { id: "A" | "B"; m: SysMetrics | null; active: boolean }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.02] px-2.5 py-2">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold text-slate-200">System {id}</span>
        {active && <span className="text-[9px] uppercase tracking-wider text-beam-300">running…</span>}
      </div>
      <div className="mt-1 space-y-0.5 text-[11px]">
        <Metric label="BER" value={m?.ber != null ? m.ber.toExponential(2) : "—"} accent />
        <Metric label="SNR" value={m?.snr_est_dB != null ? `${m.snr_est_dB.toFixed(1)} dB` : "—"} />
        <Metric
          label="P_rx"
          value={m?.P_rx_avg_uW != null ? `${m.P_rx_avg_uW.toFixed(1)} µW` : "—"}
        />
      </div>
    </div>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-400">{label}</span>
      <span className={"font-mono " + (accent ? "text-beam-200" : "text-slate-300")}>{value}</span>
    </div>
  );
}

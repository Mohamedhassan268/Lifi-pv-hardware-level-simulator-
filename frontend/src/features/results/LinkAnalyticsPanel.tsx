/**
 * LinkAnalyticsPanel — the derived-feature table for the current run, plus
 * dataset export. Features + their labels/units/groups come from the backend
 * (cosim/features.py via /api/features/schema); the per-run values arrive on
 * the WS `metrics.features` payload. The dataset control sweeps one config
 * field server-side and downloads a CSV/JSONL ML dataset (one row per point).
 */

import { useEffect, useMemo, useState } from "react";

import { api, type FeatureSchema } from "@/api/client";
import { Button } from "@/primitives/Button";
import { useConfigStore } from "@/store/configStore";
import { useResultsStore } from "@/store/resultsStore";

function downloadText(filename: string, text: string, mime: string) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** SI-aware value formatter. Picks a prefix by magnitude and appends the unit. */
function fmt(value: number | string | null, unit: string): string {
  if (value === null) return "—";
  if (typeof value === "string") return value;
  if (!Number.isFinite(value)) return Number.isNaN(value) ? "—" : "∞";
  if (value === 0) return `0${unit ? " " + unit : ""}`;

  // Time / power / Hz / bps benefit from SI prefixes; ratios & dB & % don't.
  const siUnits = ["s", "W", "Hz", "bps", "J", "A", "V"];
  if (siUnits.includes(unit)) {
    const prefixes: [number, string][] = [
      [1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""],
      [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"], [1e-12, "p"],
    ];
    const abs = Math.abs(value);
    const [scale, p] = prefixes.find(([s]) => abs >= s) ?? [1e-12, "p"];
    const v = value / scale;
    const s = Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(3);
    return `${s} ${p}${unit}`;
  }
  // Dimensionless / dB / %: small or large → exponential, else 3 sig figs.
  const abs = Math.abs(value);
  const body = abs !== 0 && (abs < 1e-3 || abs >= 1e5) ? value.toExponential(2)
    : Math.abs(value) >= 100 ? value.toFixed(1) : value.toPrecision(3);
  return unit ? `${body} ${unit}` : body;
}

export function LinkAnalyticsPanel() {
  const features = useResultsStore((s) => s.metrics?.features ?? null);
  const config = useConfigStore((s) => s.config);

  const [schema, setSchema] = useState<FeatureSchema | null>(null);
  const [sweepable, setSweepable] = useState<string[]>([]);
  const [param, setParam] = useState("distance_m");
  const [min, setMin] = useState(0.3);
  const [max, setMax] = useState(3);
  const [points, setPoints] = useState(10);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.getFeatureSchema()
      .then((r) => { setSchema(r.features); setSweepable(r.sweepable); })
      .catch((e) => setErr(String(e)));
  }, []);

  // Group features by their `group` field, preserving schema declaration order.
  const groups = useMemo(() => {
    if (!schema) return [];
    const out: { name: string; keys: string[] }[] = [];
    for (const [key, meta] of Object.entries(schema)) {
      let g = out.find((x) => x.name === meta.group);
      if (!g) { g = { name: meta.group, keys: [] }; out.push(g); }
      g.keys.push(key);
    }
    return out;
  }, [schema]);

  const exportRun = (kind: "csv" | "json") => {
    if (!features || !schema) return;
    if (kind === "json") {
      downloadText("link_features.json", JSON.stringify(features, null, 2), "application/json");
      return;
    }
    const keys = Object.keys(schema);
    const header = keys.join(",");
    const row = keys.map((k) => {
      const v = features[k];
      return v === null || (typeof v === "number" && Number.isNaN(v)) ? "" : String(v);
    }).join(",");
    downloadText("link_features.csv", `${header}\n${row}\n`, "text/csv");
  };

  const exportDataset = async (format: "csv" | "jsonl") => {
    setBusy(true); setErr(null);
    try {
      const n = Math.max(2, Math.floor(points));
      const step = (max - min) / (n - 1);
      const values = Array.from({ length: n }, (_, i) => min + i * step);
      const text = await api.featureDatasetText({ config, param, values, n_bits: 2000 }, format);
      downloadText(`features_${param}.${format}`, text, format === "csv" ? "text/csv" : "application/x-ndjson");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="border border-hair bg-slate-950">
      <header className="flex items-center justify-between gap-3 border-b border-hair px-2 py-1.5">
        <span className="label-inst">Link analytics</span>
        <div className="flex gap-1">
          <Button variant="ghost" onClick={() => exportRun("csv")} disabled={!features}>
            Run CSV
          </Button>
          <Button variant="ghost" onClick={() => exportRun("json")} disabled={!features}>
            Run JSON
          </Button>
        </div>
      </header>

      {!features ? (
        <p className="px-2 py-3 text-[11px] text-slate-500">
          Run a simulation to compute link features.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-px bg-hair md:grid-cols-2">
          {groups.map((g) => (
            <div key={g.name} className="bg-slate-950">
              <div className="label-inst border-b border-hair px-2 py-1 text-slate-400">
                {g.name}
              </div>
              <table className="w-full readout text-[11px]">
                <tbody>
                  {g.keys.map((k) => {
                    const meta = schema![k];
                    return (
                      <tr key={k} className="border-b border-hair last:border-b-0 hover:bg-white/[0.02]">
                        <td className="px-2 py-1 text-slate-400" title={meta.desc}>{meta.label}</td>
                        <td className="px-2 py-1 text-right tab-nums text-slate-100">
                          {fmt(features[k], meta.unit)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}

      {/* Dataset export — sweep one field server-side into an ML dataset. */}
      <div className="flex flex-wrap items-end gap-2 border-t border-hair px-2 py-2">
        <div className="flex flex-col gap-0.5">
          <label className="label-inst text-slate-500">Sweep</label>
          <select
            value={param}
            onChange={(e) => setParam(e.target.value)}
            className="border border-hair bg-slate-900 px-1.5 py-1 text-[11px] text-slate-100"
          >
            {sweepable.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        {([["min", min, setMin], ["max", max, setMax], ["points", points, setPoints]] as const).map(
          ([lbl, val, setter]) => (
            <div key={lbl} className="flex w-20 flex-col gap-0.5">
              <label className="label-inst text-slate-500">{lbl}</label>
              <input
                type="number"
                value={val}
                onChange={(e) => setter(Number(e.target.value))}
                className="w-full border border-hair bg-slate-900 px-1.5 py-1 text-[11px] text-slate-100 tab-nums"
              />
            </div>
          ),
        )}
        <Button onClick={() => exportDataset("csv")} disabled={busy}>
          {busy ? "…" : "Dataset CSV"}
        </Button>
        <Button variant="ghost" onClick={() => exportDataset("jsonl")} disabled={busy}>
          JSONL
        </Button>
        {err && <span className="readout text-[10px] text-rose-400">{err}</span>}
      </div>
    </section>
  );
}

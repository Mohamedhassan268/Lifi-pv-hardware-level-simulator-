/**
 * MeasuredRoute — overlay a real-board RX capture on the simulator.
 *
 * Upload a CSV exported from the ESP serial monitor (or a DMM log), pick the
 * preset the bench setup mirrors, and the backend aligns the measured trace to
 * the simulated V_rx and scores the agreement (correlation + NRMSE) plus a BER
 * comparison. This is the hardware-faithfulness check: does the sim predict the
 * board?
 */

import { useEffect, useRef, useState } from "react";

import { api, type MeasuredCompareResponse } from "@/api/client";
import { Button } from "@/primitives/Button";
import { Card } from "@/primitives/Card";
import { NumberField } from "@/primitives/NumberField";
import { PlotCanvas } from "@/primitives/PlotCanvas";
import { Select } from "@/primitives/Select";

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <div className="readout text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="readout mt-0.5 text-lg tabular-nums text-slate-100">{value}</div>
    </Card>
  );
}

function fmtBer(v: number | null | undefined): string {
  if (v == null) return "—";
  return v === 0 ? "0" : v.toExponential(2);
}

export function MeasuredRoute() {
  const [presets, setPresets] = useState<string[]>([]);
  const [preset, setPreset] = useState("lifi_poc_breadboard");
  const [csv, setCsv] = useState("");
  const [filename, setFilename] = useState("");
  const [sampleRate, setSampleRate] = useState<number | undefined>(undefined);
  const [measuredBer, setMeasuredBer] = useState<number | undefined>(undefined);
  const [result, setResult] = useState<MeasuredCompareResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.listPresets().then(setPresets).catch(() => {});
  }, []);

  async function onFile(f: File | null) {
    if (!f) return;
    setFilename(f.name);
    setCsv(await f.text());
    setResult(null);
    setError(null);
  }

  async function run() {
    if (!csv.trim()) {
      setError("Upload a CSV capture first.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api.compareMeasured({
          csv,
          preset,
          sample_rate_hz: sampleRate,
          measured_ber: measuredBer,
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const m = result?.metrics;
  const ber = m?.ber;

  return (
    <div className="mx-auto max-w-6xl space-y-4 p-6">
      <header>
        <h2 className="text-lg font-semibold text-slate-100">Measured vs Simulated</h2>
        <p className="text-sm text-slate-400">
          Upload an RX capture from the real board and overlay it on the simulator to
          validate the model against hardware.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr]">
        {/* ---- Controls ---- */}
        <div className="space-y-3">
          <Card>
            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
              Capture
            </h3>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,.txt"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            />
            <Button variant="ghost" onClick={() => fileRef.current?.click()}>
              {filename ? `\u{1F4C4} ${filename}` : "Upload .csv capture"}
            </Button>
            <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
              Two columns <span className="font-mono text-slate-400">t_s,adc</span> — or one{" "}
              <span className="font-mono text-slate-400">adc</span> column with a{" "}
              <span className="font-mono text-slate-400"># sample_rate_hz=…</span> header.
              Optional <span className="font-mono text-slate-400"># ber=…</span> line.
            </p>
          </Card>

          <Card>
            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
              Simulation
            </h3>
            <Select
              label="Preset"
              value={preset}
              onChange={setPreset}
              options={presets.length ? presets : [preset]}
            />
            <div className="mt-2 grid grid-cols-2 gap-2">
              <NumberField
                label="Sample rate"
                unit="Hz"
                value={sampleRate}
                onChange={setSampleRate}
                hint="if no time column"
              />
              <NumberField
                label="Measured BER"
                value={measuredBer}
                onChange={setMeasuredBer}
                step={0.001}
                hint="overrides # ber="
              />
            </div>
            <Button className="mt-3 w-full" onClick={run} disabled={busy}>
              {busy ? "Comparing…" : "Compare"}
            </Button>
            {error && <p className="mt-2 text-xs text-rose-400">{error}</p>}
          </Card>
        </div>

        {/* ---- Results ---- */}
        <div className="space-y-3">
          {!result ? (
            <Card>
              <p className="text-sm text-slate-500">
                Upload a capture and hit Compare to see the overlay and agreement metrics.
              </p>
            </Card>
          ) : (
            <>
              <Card>
                <div className="mb-2 flex items-baseline justify-between">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                    RX waveform overlay
                  </h3>
                  <span className="text-xs text-slate-400">{result.message}</span>
                </div>
                <PlotCanvas
                  data={[
                    {
                      x: result.series.t,
                      y: result.series.measured,
                      type: "scatter",
                      mode: "lines",
                      name: "measured",
                      line: { color: "#fbbf24", width: 1.5 },
                    },
                    {
                      x: result.series.t,
                      y: result.series.simulated,
                      type: "scatter",
                      mode: "lines",
                      name: "simulated",
                      line: { color: "#7dd3fc", width: 1.5 },
                    },
                  ]}
                  layout={{
                    xaxis: { title: { text: "normalised time" } },
                    yaxis: { title: { text: "amplitude (z-score)" } },
                  }}
                  aspect="16/6"
                />
              </Card>

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Metric label="Agreement" value={`${m!.agreement_pct.toFixed(0)}%`} />
                <Metric label="Correlation" value={m!.correlation.toFixed(3)} />
                <Metric label="NRMSE" value={m!.nrmse.toFixed(3)} />
                <Metric label="Time shift" value={`${(m!.lag_frac * 100).toFixed(1)}%`} />
              </div>

              <Card>
                <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
                  BER
                </h3>
                <div className="grid grid-cols-3 gap-3">
                  <Metric label="Simulated" value={fmtBer(ber?.simulated)} />
                  <Metric label="Measured" value={fmtBer(ber?.measured)} />
                  <Metric
                    label="Rel. error"
                    value={ber?.rel_error != null ? `${(ber.rel_error * 100).toFixed(1)}%` : "—"}
                  />
                </div>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

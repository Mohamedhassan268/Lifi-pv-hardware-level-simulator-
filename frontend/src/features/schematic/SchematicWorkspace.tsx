/**
 * SchematicWorkspace — the drag-and-drop circuit editor.
 *
 *   [ Palette ] [ React Flow canvas ] [ Inspector + Simulate ]
 *
 * The canvas lets the user place parts, drag them, and wire pin-to-pin. The
 * Simulate button serializes the graph (schematicStore.toCircuitGraph) and
 * POSTs it to /api/schematic-sim; the returned BER/metrics render in the
 * inspector. This is the UI half of the graph -> netlist -> solver -> results
 * round trip proven headless in cosim/graph_netlist.py.
 */

import { Background, BackgroundVariant, ConnectionMode, Controls, ReactFlow } from "@xyflow/react";
import { useCallback, useEffect, useState } from "react";

import { api, type FirmwareFinding, type FirmwareInfo } from "@/api/client";
import { Button } from "@/primitives/Button";
import { Card } from "@/primitives/Card";
import { Palette } from "@/features/schematic/Palette";
import { PartNode } from "@/features/schematic/PartNode";
import { CANVAS_PRESETS } from "@/features/schematic/presets";
import { useSchematicStore } from "@/store/schematicStore";

const nodeTypes = { part: PartNode };

interface SimResult {
  ber: number | null;
  ber_incircuit?: number | null;
  n_bits?: number;
  message?: string;
  warnings?: string[];
  diagnostics?: Record<string, unknown>;
}

export function SchematicWorkspace() {
  const nodes = useSchematicStore((s) => s.nodes);
  const edges = useSchematicStore((s) => s.edges);
  const onNodesChange = useSchematicStore((s) => s.onNodesChange);
  const onEdgesChange = useSchematicStore((s) => s.onEdgesChange);
  const onConnect = useSchematicStore((s) => s.onConnect);
  const selectNode = useSchematicStore((s) => s.selectNode);
  const selectedNodeId = useSchematicStore((s) => s.selectedNodeId);
  const setNodeValue = useSchematicStore((s) => s.setNodeValue);
  const removeSelected = useSchematicStore((s) => s.removeSelected);
  const clear = useSchematicStore((s) => s.clear);
  const loadGraph = useSchematicStore((s) => s.loadGraph);
  const toCircuitGraph = useSchematicStore((s) => s.toCircuitGraph);
  const pendingPin = useSchematicStore((s) => s.pendingPin);
  const clearPending = useSchematicStore((s) => s.clearPending);

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SimResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingPreset, setLoadingPreset] = useState(false);

  // Run operating point (sent to the two-pass cosim; reference defaults).
  const [distanceM, setDistanceM] = useState("0.325");
  const [dataRateKbps, setDataRateKbps] = useState("2.5");
  const [nBits, setNBits] = useState("128");
  const [modulation, setModulation] = useState("OOK");

  // Per-ESP firmware: merged PHY overrides + per-role parse info for display.
  const [firmwareParams, setFirmwareParams] = useState<Record<string, number>>({});
  type FwInfo = {
    name: string;
    findings: FirmwareFinding[];
    info: FirmwareInfo[];
    warnings: string[];
  };
  const [firmwareInfo, setFirmwareInfo] = useState<{ tx: FwInfo | null; rx: FwInfo | null }>(
    { tx: null, rx: null },
  );

  const onFirmware = useCallback(async (role: "tx" | "rx", file: File) => {
    setError(null);
    try {
      const source = await file.text();
      const res = await api.parseFirmware(role, source, file.name);
      setFirmwareInfo((p) => ({
        ...p,
        [role]: { name: file.name, findings: res.findings, info: res.info, warnings: res.warnings },
      }));
      setFirmwareParams((p) => ({ ...p, ...res.params }));
      if (res.params.data_rate_bps) setDataRateKbps(String(res.params.data_rate_bps / 1000));
      setModulation("PWM_ASK"); // firmware upload implies the PoC PWM-ASK link
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const selected = nodes.find((n) => n.id === selectedNodeId) ?? null;

  // Esc cancels an in-progress Ctrl+click wire.
  useEffect(() => {
    if (!pendingPin) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") clearPending();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pendingPin, clearPending]);

  const loadPreset = useCallback(
    async (key: string) => {
      const preset = CANVAS_PRESETS.find((p) => p.key === key);
      if (!preset) return;
      setLoadingPreset(true);
      setError(null);
      setResult(null);
      try {
        const { nodes: ns, edges: es } = await preset.build();
        loadGraph(ns, es);
        if (preset.modulation) setModulation(preset.modulation);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoadingPreset(false);
      }
    },
    [loadGraph],
  );

  const simulate = useCallback(async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const graph = toCircuitGraph();
      const body = {
        ...graph,
        ...firmwareParams, // carrier/pwm/mod-depth/adc/sample-rate from uploaded .ino
        distance_m: Number(distanceM) || undefined,
        data_rate_bps: (Number(dataRateKbps) || 0) * 1000 || undefined,
        n_bits: Number(nBits) || undefined,
        modulation,
      };
      const res = await api.simulateSchematic(body);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, [toCircuitGraph, distanceM, dataRateKbps, nBits, modulation, firmwareParams]);

  return (
    <div className="grid h-[calc(100vh-3rem)] grid-cols-[260px_1fr_300px] gap-3 p-3">
      <div className="overflow-y-auto">
        <Palette />
      </div>

      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-slate-950">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={(_, n) => selectNode(n.id)}
          onPaneClick={() => {
            selectNode(null);
            clearPending();
          }}
          connectionMode={ConnectionMode.Loose}
          defaultEdgeOptions={{ type: "smoothstep", style: { stroke: "#94a3b8", strokeWidth: 1.6 } }}
          proOptions={{ hideAttribution: true }}
          fitView
        >
          {/* Proteus-style boxed grid: fine minor cells + bolder major lines. */}
          <Background
            id="grid-minor"
            variant={BackgroundVariant.Lines}
            gap={20}
            lineWidth={1}
            color="rgba(148,163,184,0.07)"
          />
          <Background
            id="grid-major"
            variant={BackgroundVariant.Lines}
            gap={100}
            lineWidth={1}
            color="rgba(148,163,184,0.16)"
          />
          <Controls showInteractive={false} />
        </ReactFlow>
        <div className="pointer-events-none absolute left-4 top-3 text-[10px] uppercase tracking-[0.2em] text-slate-500">
          Drag to place · drag or Ctrl+click pins to wire
        </div>
        {pendingPin && (
          <div className="pointer-events-none absolute left-1/2 top-3 -translate-x-1/2 rounded-full
                          border border-beam-400/50 bg-beam-400/10 px-3 py-1 text-[10px]
                          uppercase tracking-[0.2em] text-beam-200">
            Wiring — Ctrl+click the second pin (Esc to cancel)
          </div>
        )}
      </div>

      <div className="flex flex-col gap-3 overflow-y-auto">
        <Card>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
            Inspector
          </h3>
          {selected ? (
            <div className="space-y-2">
              <div className="text-xs text-slate-400">
                <span className="font-mono text-slate-100">{selected.id}</span> ·{" "}
                {selected.data.label}
                {selected.data.partCode && selected.data.partCode !== selected.data.label && (
                  <span className="ml-1 font-mono text-slate-500">({selected.data.partCode})</span>
                )}
              </div>
              {(selected.data.ctype === "R" || selected.data.ctype === "V") && (
                <label className="block text-xs text-slate-400">
                  Value
                  <input
                    value={selected.data.value ?? ""}
                    onChange={(e) => setNodeValue(selected.id, e.target.value)}
                    className="mt-1 w-full rounded-lg border border-white/10 bg-white/[0.04] px-2 py-1
                               font-mono text-sm text-slate-100 focus:outline-none focus:ring-2
                               focus:ring-beam-400/40"
                  />
                </label>
              )}
              <ul className="text-[11px] text-slate-500">
                {selected.data.ports.map((p) => (
                  <li key={p.name}>
                    <span className="font-mono text-slate-300">{p.name}</span> — {p.role}
                  </li>
                ))}
              </ul>
              <Button variant="ghost" onClick={removeSelected}>
                Delete part
              </Button>
            </div>
          ) : (
            <p className="text-xs text-slate-500">Select a part to edit.</p>
          )}
        </Card>

        <Card>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
            Presets
          </h3>
          <div className="space-y-1.5">
            {CANVAS_PRESETS.map((p) => (
              <button
                key={p.key}
                type="button"
                disabled={loadingPreset}
                onClick={() => loadPreset(p.key)}
                className="w-full rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-left
                           hover:bg-white/[0.08] disabled:opacity-50"
              >
                <div className="text-xs text-slate-100">{p.label}</div>
                <div className="mt-0.5 text-[10px] text-slate-500">{p.description}</div>
              </button>
            ))}
            {loadingPreset && <p className="text-[10px] text-slate-500">Loading preset…</p>}
          </div>
        </Card>

        <Card>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
            Run settings
          </h3>
          <label className="mb-2 block text-[10px] text-slate-400">
            Modulation
            <select
              value={modulation}
              onChange={(e) => setModulation(e.target.value)}
              className="mt-1 w-full rounded-lg border border-white/10 bg-white/[0.04] px-2 py-1
                         text-xs text-slate-100 focus:outline-none focus:ring-2 focus:ring-beam-400/40"
            >
              <option value="OOK">OOK</option>
              <option value="OOK_Manchester">Manchester (OOK)</option>
              <option value="BFSK">BFSK</option>
              <option value="OFDM">OFDM</option>
              <option value="PWM_ASK">PWM-ASK</option>
            </select>
          </label>
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: "Distance (m)", value: distanceM, set: setDistanceM },
              { label: "Rate (kbps)", value: dataRateKbps, set: setDataRateKbps },
              { label: "Bits", value: nBits, set: setNBits },
            ].map((f) => (
              <label key={f.label} className="block text-[10px] text-slate-400">
                {f.label}
                <input
                  value={f.value}
                  onChange={(e) => f.set(e.target.value)}
                  inputMode="decimal"
                  className="mt-1 w-full rounded-lg border border-white/10 bg-white/[0.04] px-2 py-1
                             font-mono text-xs text-slate-100 focus:outline-none focus:ring-2
                             focus:ring-beam-400/40"
                />
              </label>
            ))}
          </div>
        </Card>

        <Card>
          <h3 className="mb-1 text-sm font-semibold uppercase tracking-wider text-slate-300">
            ESP firmware
          </h3>
          <p className="mb-2 text-[10px] text-slate-500">
            Upload each ESP&apos;s .ino — the PHY constants are parsed into the run.
          </p>
          {(["tx", "rx"] as const).map((role) => (
            <FirmwareSlot
              key={role}
              role={role}
              info={firmwareInfo[role]}
              onFile={(f) => onFirmware(role, f)}
            />
          ))}
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
              Simulate
            </h3>
            <button
              type="button"
              onClick={clear}
              className="text-[10px] uppercase tracking-wider text-slate-500 hover:text-slate-300"
            >
              Clear
            </button>
          </div>
          <Button onClick={simulate} disabled={running || nodes.length === 0} className="mt-2 w-full">
            {running ? "Running…" : "Run circuit"}
          </Button>

          {error && <p className="mt-3 text-xs text-rose-400">{error}</p>}

          {result && (
            <div className="mt-3 space-y-1 text-xs">
              {result.ber !== null && (
                <div className="flex justify-between">
                  <span className="text-slate-400">BER (with noise)</span>
                  <span className="font-mono text-beam-200">{result.ber.toExponential(2)}</span>
                </div>
              )}
              {result.ber_incircuit !== null && result.ber_incircuit !== undefined && (
                <div className="flex justify-between">
                  <span className="text-slate-400">BER (in-circuit)</span>
                  <span className="font-mono text-slate-300">
                    {result.ber_incircuit.toExponential(2)}
                  </span>
                </div>
              )}
              {result.n_bits !== undefined && (
                <div className="flex justify-between">
                  <span className="text-slate-400">bits</span>
                  <span className="font-mono text-slate-200">{result.n_bits}</span>
                </div>
              )}
              {result.diagnostics && <DiagRows diag={result.diagnostics} />}
              {result.message && <p className="text-slate-500">{result.message}</p>}
              {result.warnings?.map((w) => (
                <p key={w} className="text-amber-400">⚠ {w}</p>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

// One per-ESP firmware upload slot: pick a .ino, show the parsed PHY constants.
function FirmwareSlot({
  role,
  info,
  onFile,
}: {
  role: "tx" | "rx";
  info: { name: string; findings: FirmwareFinding[]; info: FirmwareInfo[]; warnings: string[] } | null;
  onFile: (f: File) => void;
}) {
  const label = role === "tx" ? "TX ESP" : "RX ESP";
  return (
    <div className="mb-2 rounded-lg border border-white/10 bg-white/[0.02] p-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold text-slate-300">{label}</span>
        <label
          className="cursor-pointer rounded border border-white/10 bg-white/[0.04] px-2 py-0.5
                     text-[10px] text-slate-200 hover:bg-white/[0.08]"
        >
          {info ? "Replace .ino" : "Upload .ino"}
          <input
            type="file"
            accept=".ino,.c,.cpp,.h,.txt"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onFile(f);
              e.target.value = "";
            }}
          />
        </label>
      </div>
      {info && (
        <div className="mt-1.5">
          <p className="truncate font-mono text-[9px] text-slate-500">{info.name}</p>
          {info.findings.length === 0 && (
            <p className="text-[10px] text-slate-500">No PHY constants recognised.</p>
          )}
          <dl className="mt-1 space-y-0.5">
            {info.findings.map((f) => (
              <div key={f.config_field} className="flex items-center justify-between gap-2 text-[10px]">
                <span className="text-slate-400">{f.label}</span>
                <span className="font-mono text-emerald-300">
                  {fmtFw(f.config_field, f.value)}
                  {f.confidence !== "high" && (
                    <span className="ml-1 text-slate-500">({f.confidence})</span>
                  )}
                </span>
              </div>
            ))}
          </dl>
          {info.info.length > 0 && (
            <dl className="mt-1 space-y-0.5 border-t border-white/5 pt-1">
              {info.info.map((it) => (
                <div key={it.label} className="flex items-center justify-between gap-2 text-[10px]">
                  <span className="text-slate-500">{it.label}</span>
                  <span className="font-mono text-slate-300">{it.value}</span>
                </div>
              ))}
            </dl>
          )}
          {info.warnings.map((w, i) => (
            <p key={i} className="mt-1 text-[9px] text-harvest-300">
              {w}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

function fmtFw(field: string, v: number): string {
  if (["carrier_freq_hz", "pwm_freq_hz", "mcu_sample_rate_hz"].includes(field))
    return v >= 1000 ? `${(v / 1000).toFixed(v >= 1e5 ? 0 : 1)} kHz` : `${v} Hz`;
  if (field === "data_rate_bps") return v >= 1000 ? `${(v / 1000).toFixed(2)} kbps` : `${v} bps`;
  if (field === "modulation_depth") return `${(v * 100).toFixed(0)}%`;
  if (field === "adc_bits") return `${v}-bit`;
  if (field === "bias_current_A") return `${(v * 1000).toFixed(1)} mA`;
  if (field === "led_radiated_power_mW") return `${v} mW`;
  if (field === "mcu_clock_MHz") return `${v} MHz`;
  if (field === "adc_vref") return `${v} V`;
  if (field === "prbs_order") return `PRBS-${v}`;
  return String(v);
}

// Compact key diagnostics from the two-pass run (received optical power + the
// input-referred noise the post-demod applied).
function DiagRows({ diag }: { diag: Record<string, unknown> }) {
  const prx = diag.p_rx_uW as number[] | undefined;
  const noise = diag.noise_sigma_mV as number | undefined;
  return (
    <>
      {prx && (
        <div className="flex justify-between">
          <span className="text-slate-400">P_rx peak</span>
          <span className="font-mono text-slate-300">{prx[1].toFixed(1)} µW</span>
        </div>
      )}
      {typeof noise === "number" && (
        <div className="flex justify-between">
          <span className="text-slate-400">noise σ</span>
          <span className="font-mono text-slate-300">{noise.toFixed(1)} mV</span>
        </div>
      )}
    </>
  );
}

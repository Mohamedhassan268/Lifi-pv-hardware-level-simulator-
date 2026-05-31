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

import { Background, ConnectionMode, Controls, ReactFlow } from "@xyflow/react";
import { useCallback, useState } from "react";

import { api } from "@/api/client";
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

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SimResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingPreset, setLoadingPreset] = useState(false);

  // Run operating point (sent to the two-pass cosim; reference defaults).
  const [distanceM, setDistanceM] = useState("0.325");
  const [dataRateKbps, setDataRateKbps] = useState("2.5");
  const [nBits, setNBits] = useState("128");
  const [modulation, setModulation] = useState("OOK");

  const selected = nodes.find((n) => n.id === selectedNodeId) ?? null;

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
  }, [toCircuitGraph, distanceM, dataRateKbps, nBits, modulation]);

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
          onPaneClick={() => selectNode(null)}
          connectionMode={ConnectionMode.Loose}
          defaultEdgeOptions={{ type: "smoothstep", style: { stroke: "#94a3b8", strokeWidth: 1.6 } }}
          proOptions={{ hideAttribution: true }}
          fitView
        >
          <Background gap={20} size={1} color="rgba(255,255,255,0.06)" />
          <Controls showInteractive={false} />
        </ReactFlow>
        <div className="pointer-events-none absolute left-4 top-3 text-[10px] uppercase tracking-[0.2em] text-slate-500">
          Drag to place · drag pin-to-pin to wire
        </div>
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

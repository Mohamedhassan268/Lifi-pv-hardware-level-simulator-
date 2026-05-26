# Project Status — Hardware-Faithful LiFi/PV Simulator

**Last updated:** 2026-05-20

---

## 1. Purpose

A SPICE-grade co-simulator for visible-light communication systems that double
as photovoltaic energy harvesters. The simulator validates the full
TX → optical channel → RX chain against six peer-reviewed papers (Kadirvelu
2021, Sarwar 2017, Xu 2024, Oliveira 2024, González 2024, Correa 2025) and is
intended to support a Cairo hardware deployment scheduled for July 2026.

Design principles:

- **Hardware-faithful** — every component value comes from a datasheet, not
  empirical fitting. 14 specific parts (Philips LXM5-PD01 LED, IXYS
  KXOB25-04X3F GaAs solar cell, TI INA322 amp, TI TLV7011 comparator, etc.)
  have explicit Python classes with `get_parameters()`.
- **Physics-first** — Lambertian emission, Beer-Lambert atmospheric attenuation,
  Johnson-Nyquist + shot + ambient + amplifier + ADC + processing noise (the
  six-source noise model).
- **Dual engine** — Python for fast iteration; SPICE (LTspice / ngspice) for
  full transient circuit analysis. Auto-falls back to Python when no SPICE
  binary is found.
- **Paper-agnostic** — papers are validation targets, not the architecture.
  Adding a paper is a JSON preset + a 30-line Python file.

---

## 2. Migration status

The project is mid-migration from a PyQt6 desktop GUI to a React/Tauri desktop
app. Original simulation core remains untouched; only the shell is changing.

| Phase | Description | Status |
|---|---|---|
| **1** | FastAPI REST skeleton + React/Vite Setup view + Zustand stores + hero animation | ✅ Done |
| **2** | Pipeline WebSocket + live step cards + Plotly results (waveforms, bits compare) | ✅ Done |
| **3** | Workbench (14-component browser, filter+search, "Use as…" slot wiring) | ✅ Done |
| **4** | Sweep WebSocket (thermal + Monte Carlo, live streaming Plotly) | ✅ Done |
| **5** | Tauri packaging (sidecar lifecycle, PyInstaller spec, MSI installer) | ✅ Done |
| **6** | UI redesign: Landing → Choice modal → modal-based BuilderWorkspace (react-flow schematic, NoiseOverlay, ResultsMiniPanel, ExportButton, EngineRoute results expansion) | ✅ Done |
| **7** | EDA-grade upgrades: noise breakdown, eye/jitter metrics, analytical BER + Wilson CI, BER-vs-SNR / BER-vs-distance sweeps, n_bits bump (100→10k), AC analysis (Bode + GM/PM), per-component tolerance metadata, IEEE 802.15.7 PHY-I compliance | ✅ Done |

**Phase 5 final artifacts** (built 2026-05-16, Rust compile 8m 06s — these
predate the Phase 6/7 work and need to be rebuilt for distribution):

| Artifact | Size | Path |
|---|---|---|
| Portable Tauri shell | 12.83 MB | `dist\installer\LiFi PV Simulator.exe` |
| MSI installer (WiX 3.14) | 73.99 MB | `dist\installer\LiFi PV Simulator_0.1.0_x64_en-US.msi` |
| NSIS .exe installer | 72.81 MB | `dist\installer\LiFi PV Simulator_0.1.0_x64-setup.exe` |

Build notes:
- MSVC toolchain: VS Build Tools 2026 (v18.6.11806.211) at `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\`. `cl.exe` at `VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\`. Activate via `vcvars64.bat` before building.
- `CARGO_TARGET_DIR=D:\cargo-target\lifi-pv` — keep cargo target off C: (only ~9 GB free) and out of OneDrive (repo lives in OneDrive, which causes intermittent file locks during build).
- **Avast shields must be disabled during the build.** Avast's `aswMonFlt` filter driver races cargo on every `.exe` write/spawn and produces intermittent `os error 5` failures (most commonly on the `proc-macro2` build script). Defender exclusions don't help — Avast scans separately. Disable via the tray icon for 10 minutes, build, then re-enable.

The legacy PyQt6 GUI (`python cli.py gui`) **remains fully functional** and
is the recommended day-to-day driver until the Tauri build lands.

### Phase 6 — UI redesign (2026-05-18 → 2026-05-19)

The Setup/Engine/Workbench/Build tabbed shell from Phase 1-4 was reworked
into a launch flow that matches the "design then simulate" mental model:

  Landing splash  →  "Builder" CTA  →  LaunchChoiceModal (preset vs own design)
                  →  BuilderWorkspace (category rail + live react-flow schematic)
                  →  Simulate (no gate)  →  ResultsMiniPanel
                  →  "Open full results →" → EngineRoute (all Plotly panels)

What changed:

- **LandingRoute** + **LaunchChoiceModal** replace the old direct-into-Setup
  default. Presets are *only* offered at the Landing step; the Simulate
  button inside the workspace runs straight through.
- **BuilderWorkspace** with a left **CategoryRail** (Transmitter, Receiver,
  Geometry, Noise — Modulation folded into Transmitter) and a right
  react-flow **SchematicCanvas** with three SVG-rich custom nodes (TxNode,
  ChannelNode, RxNode). A **NoiseOverlay** layer renders animated grain
  and beam distortion on the channel when noise is applied.
- Per-category **ModalShell** wraps the existing TransmitterTab /
  ReceiverTab / ChannelTab / NoiseTab form bodies — no duplicate form
  logic.
- **builderUIStore** drives the `configured[category]` flags that gate
  the schematic visuals + the rail's checkmarks.
- **ResultsMiniPanel** appears inline over the schematic after Simulate;
  the "Open full results →" link is held back until both metrics AND
  waveforms have arrived (avoids the race where EngineRoute mounts with
  half-populated stores).
- **ExportButton** in EngineRoute uses Tauri `dialog.open()` +
  `fs.writeBinaryFile()` to save `config.json` + a PNG per Plotly panel
  to a user-picked folder; disabled outside Tauri.
- Backend bug fix: `pipeline_ws.py` previously only emitted a `waveforms`
  WS message when the Python engine ran. `backend/spice_waveforms.py`
  closes the SPICE path by parsing the `.raw` file and emitting the same
  payload shape, so EngineRoute panels populate regardless of engine.

### Phase 7 — EDA-grade upgrades (2026-05-19 → 2026-05-20)

Eight upgrades shipped across one PR-sized block of work. The three
"engine already exists" wins arrived first; the heavier additions
(AC analysis, standards) followed.

- **Per-stage noise breakdown** (`NoiseBreakdownPanel`) — backend attaches
  `NoiseModel.compute_noise().as_dict()` to every metrics message; the
  panel renders a stacked horizontal bar and a numeric table.
- **Eye opening + jitter σ** (`lib/eyeMetrics.ts`) — pure frontend math
  on `V_rx + bits_tx`. Four chips on the EyeDiagramPanel header.
- **Analytical BER companion** (`lib/dsp.ts::analyticalBer`) — erfc
  (Abramowitz-Stegun 7.1.26), Q-function, Wilson 95% CI helpers. The
  MetricsCard now shows "Measured vs Theory" with Δ in dB.
- **BER vs SNR / BER vs distance sweeps** (`cosim/ber_sweep.py`) — two
  new `/ws/sweep` kinds (`ber_snr`, `ber_distance`), Wilson-CI error bars
  in the SweepChart, analytical theory curve overlaid as a dashed line.
- **Default `n_bits` 100 → 10000** + a **Precision run** button
  (`n_bits = 100000`) in the rail. Paper validation harnesses keep their
  original values for reproducibility.
- **AC analysis route** (`cosim/ac_analysis.py` + `routes/AcRoute.tsx`
  + `BodePanel.tsx` + `useAcSocket.ts`) — closed-form Python cascade of
  sense R → INA (single-pole) → BPF (HP+LP per stage). Reports midband,
  3-dB bandwidth, unity-gain frequency, 180° crossing, GM, PM. Streams
  via `/ws/ac`. **No SPICE round-trip** — runs in <50 ms.
- **Per-component tolerance metadata** (`cosim/tolerance.py`) — registry
  keyed by `(picker_field, part_name) → [ToleranceSpec]`. Monte Carlo
  picks up an `auto_tolerance: true` flag and the backend streams a
  `tolerance_resolved` message with the resolved table.
- **IEEE 802.15.7 PHY-I compliance** (`cosim/standards/`) — first MVP
  profile (OOK / Manchester, 100 kbps anchor, BER ≤ 1e-6 at SNR ≥ 11 dB).
  `pipeline_ws.py` accepts `profile_id` on the start frame and emits a
  `compliance` payload after metrics. UI: `StandardsModal` in the rail,
  `ComplianceBanner` above the ResultsPanel.

New frontend dependencies: `@xyflow/react@^12`, `@tauri-apps/api@^1`. No
others.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LiFi PV Simulator.exe   (Tauri Rust shell — Windows)                    │
│                                                                          │
│  ┌──────────────────────────┐    spawn      ┌──────────────────────────┐ │
│  │  WebView (Chromium)      │ ─────────────►│  lifi-backend.exe        │ │
│  │  React + Vite + Tailwind │               │  PyInstaller-bundled     │ │
│  │  + Framer Motion         │ ◄── REST ─────│  FastAPI + uvicorn       │ │
│  │  + Plotly + Zustand      │ ◄── WS    ────│  wraps cosim/* unmodified│ │
│  └──────────────────────────┘               └────────────┬─────────────┘ │
│                                                          │               │
└──────────────────────────────────────────────────────────┼───────────────┘
                                                           │
                                       ┌───────────────────┴────────────────┐
                                       │ Python simulation core             │
                                       │ ┌────────────────────────────────┐ │
                                       │ │ cosim/pipeline.py              │ │
                                       │ │   TX → Channel → RX orchestr.  │ │
                                       │ └─────┬──────────────────┬───────┘ │
                                       │       │                  │         │
                                       │  ┌────▼────┐       ┌─────▼─────┐   │
                                       │  │ SPICE   │       │ Python    │   │
                                       │  │ engine  │       │ engine    │   │
                                       │  │ (.cir)  │       │ (NumPy)   │   │
                                       │  └────┬────┘       └─────┬─────┘   │
                                       │       └────────┬─────────┘         │
                                       │                │                   │
                                       │  ┌─────────────▼─────────────────┐ │
                                       │  │ Shared modules                │ │
                                       │  │ channel.py · noise.py         │ │
                                       │  │ modulation.py · components/   │ │
                                       │  └───────────────────────────────┘ │
                                       └────────────────────────────────────┘
```

### Why this shape

- **Tauri over Electron:** ~40–80 MB smaller binary, native Rust shell, same
  Chromium webview.
- **Sidecar over PyO3-embedded Python:** the Python world (GIL, NumPy MKL,
  matplotlib) is isolated from the webview process. A crash in cosim can't
  take down the UI.
- **WebSocket for streaming:** pipeline progress, thermal sweeps, and Monte
  Carlo trials all stream point-by-point. The existing
  `SimulationPipeline.on_progress` callback in `cosim/pipeline.py` plugs
  straight into the WS without refactoring simulation code.
- **Zustand over Redux Toolkit:** the source of truth lives in Python; the
  frontend just mirrors it. Less ceremony.
- **Plotly over Recharts/uPlot:** eye diagrams + log-y BER + zoom/pan are
  free out of the box. Heavier bundle but acceptable for a desktop app.

---

## 4. Directory layout

```
hardware_faithful_simulator/
├── cli.py                       # legacy CLI entrypoint (test, gui, validate…)
├── CLAUDE.md                    # project guide for Claude Code
├── PACKAGING.md                 # how to build the .exe
├── PLAN.md / TODO.md            # paper roadmap, phase tracker
├── README.md
│
├── cosim/                       # Simulation infrastructure
│   ├── pipeline.py              #   TX → Channel → RX orchestrator
│   ├── system_config.py         #   SystemConfig dataclass + presets loader
│   │                            #   (n_bits default bumped 100 → 10000 in Phase 7)
│   ├── channel.py               #   Lambertian + Beer-Lambert + multipath
│   ├── noise.py                 #   6-source physical noise model
│   ├── modulation.py            #   5 modulation schemes + BER (predict_ber)
│   ├── python_engine.py         #   pure-Python RX simulation
│   ├── ltspice_runner.py        #   LTspice subprocess wrapper
│   ├── ngspice_runner.py        #   ngspice subprocess wrapper
│   ├── thermal_sweep.py         #   sweep over temperature
│   ├── monte_carlo.py           #   yield analysis
│   ├── ber_sweep.py             #   NEW (Phase 7): BER-vs-SNR + BER-vs-distance
│   │                            #   generators with Wilson 95% CI
│   ├── ac_analysis.py           #   NEW (Phase 7): closed-form Bode + GM/PM
│   │                            #   for the RX cascade (no SPICE round-trip)
│   ├── tolerance.py             #   NEW (Phase 7): per-component tolerance
│   │                            #   registry + derive_tolerance_spec()
│   ├── standards/               #   NEW (Phase 7): PHY compliance profiles
│   │   ├── __init__.py          #     PROFILES registry + evaluate()
│   │   ├── types.py             #     PhyProfile / Criterion / CriterionResult
│   │   └── ieee_802_15_7_phy_i.py  # IEEE 802.15.7 PHY-I (OOK/Manchester) MVP
│   ├── session.py               #   session directory mgmt
│   ├── pwl_writer.py            #   SPICE PWL bridge files
│   ├── raw_parser.py            #   .raw file parser
│   ├── sim_result.py            #   SimulationResult dataclass
│   ├── spice_extract.py         #   BER extraction from SPICE output
│   └── spice_finder.py          #   cross-platform SPICE auto-detect
│
├── components/                  # 14 hardware components  (unchanged)
│   ├── base.py                  #   abstract PhotodetectorBase / LEDBase / AmplifierBase
│   ├── solar_cells.py           #   KXOB25-04X3F, SM141K, GenericGaAsPV
│   ├── photodiodes.py           #   BPW34, SFH206K, VEMD5510
│   ├── leds.py                  #   LXM5-PD01 (Philips + Fraen lens)
│   ├── amplifiers.py            #   INA322, TLV2379, ADA4891, OPA380
│   ├── comparators.py           #   TLV7011
│   ├── mosfets.py               #   BSD235N, NTS4409
│   ├── power.py                 #   BQ25570 (TI energy harvester)
│   └── adc.py                   #   STM32H7_ADC
│
├── physics/                     # PN junction, photodetection, LED emission
├── materials/                   # GaAs / Si / InGaP semiconductor models
├── systems/                     # paper-specific system definitions
├── papers/                      # 6 paper validation scripts
├── presets/                     # 6 JSON configurations  (kadirvelu, sarwar, …)
├── spice_libs/                  # bundled SPICE component libraries
├── ngspice-45.2_64/             # bundled ngspice for Windows
│
├── gui/                         # PyQt6 GUI  (legacy, still functional)
│   ├── main_window.py
│   ├── top_bar.py
│   ├── channel_canvas.py        #   QPainter TX→RX diagram
│   ├── tab_simulation_engine.py
│   ├── tab_results.py
│   ├── tab_system_setup.py
│   ├── tab_component_library.py
│   ├── tab_validation.py
│   ├── tab_message.py
│   ├── tab_schematics.py
│   ├── workbench_window.py
│   └── theme.py
│
├── backend/                     # NEW — FastAPI wrapper
│   ├── __init__.py
│   ├── __main__.py              #   `python -m backend --port N` entrypoint
│   ├── app.py                   #   FastAPI app, CORS, route registration
│   ├── schemas.py               #   Pydantic request/response models
│   ├── requirements.txt
│   ├── lifi-backend.spec        #   PyInstaller spec for sidecar build
│   ├── spice_waveforms.py       #   NEW (Phase 6): parses SPICE .raw and
│   │                            #   emits the same waveforms payload as the
│   │                            #   Python path — fixes empty-results bug.
│   └── routers/
│       ├── presets.py           #   GET /api/presets, /api/presets/{name}
│       ├── config.py            #   POST /api/config/validate
│       ├── components.py        #   GET /api/components, /categories, /{part}
│       ├── link_budget.py       #   POST /api/link-budget
│       ├── standards.py         #   NEW (Phase 7): GET /api/standards, /{id}
│       ├── pipeline_ws.py       #   WS /ws/pipeline. Now emits noise_breakdown
│       │                        #   on metrics + compliance after metrics if a
│       │                        #   profile_id is supplied on the start frame
│       ├── sweep_ws.py          #   WS /ws/sweep — thermal · MC (auto-tol) ·
│       │                        #   ber_snr · ber_distance
│       └── ac_ws.py             #   NEW (Phase 7): WS /ws/ac (Bode batch)
│
├── frontend/                    # NEW — Vite + React + TS + Tailwind + Framer Motion
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json · tsconfig.node.json
│   ├── tailwind.config.ts · postcss.config.js
│   ├── index.html
│   ├── src/
│   │   ├── App.tsx              #   route shell with LayoutGroup + AnimatePresence
│   │   │                        #   (top bar hidden on the landing route)
│   │   ├── main.tsx             #   imports @xyflow/react/dist/style.css
│   │   ├── index.css            #   Tailwind base
│   │   ├── vite-env.d.ts        #   ImportMetaEnv types
│   │   ├── lib/
│   │   │   ├── motion.ts        #   EASE + DUR tokens (ease-in-out)
│   │   │   ├── dsp.ts           #   NEW: radix-2 FFT, erfc, Q, Wilson CI,
│   │   │   │                    #   analyticalBer (mirrors cosim.modulation)
│   │   │   ├── eyeMetrics.ts    #   NEW: eye opening + jitter σ from V_rx + bits
│   │   │   ├── tauriBridge.ts   #   NEW: dialog/fs feature-detected wrapper
│   │   │   └── exporter.ts      #   NEW: writes config.json + plot PNGs to a
│   │   │                        #   user-picked folder via Plotly.toImage
│   │   ├── api/
│   │   │   ├── backend.ts       #   getBackendBase, waitForBackend (Tauri-aware)
│   │   │   ├── client.ts        #   REST wrapper (incl. listStandards/getStandard)
│   │   │   └── ws.ts            #   WebSocket URL helper
│   │   ├── store/
│   │   │   ├── configStore.ts   #   SystemConfig mirror + draft slice
│   │   │   ├── uiStore.ts       #   route ∈ {landing, setup, engine, sweeps,
│   │   │   │                    #            builder, ac} — default "landing"
│   │   │   ├── builderUIStore.ts#   NEW: openCategory, configured[], schematicRev
│   │   │   ├── pipelineStore.ts #   3-step status (TX/Channel/RX)
│   │   │   ├── resultsStore.ts  #   metrics (+ noise_breakdown) + waveforms
│   │   │   ├── sweepStore.ts    #   thermal/MC/BER points + toleranceRows
│   │   │   ├── acStore.ts       #   NEW: AC analysis result + running flag
│   │   │   └── standardsStore.ts#   NEW: active profile + last compliance result
│   │   ├── hooks/
│   │   │   ├── useSimulationSocket.ts  # narrowed message writes; forwards
│   │   │   │                           # profile_id; handles `compliance`
│   │   │   ├── useSweepSocket.ts       # handles `tolerance_resolved`
│   │   │   └── useAcSocket.ts          # NEW: /ws/ac
│   │   ├── primitives/
│   │   │   ├── Card.tsx · Button.tsx · Slider.tsx · StatusDot.tsx
│   │   │   └── PlotCanvas.tsx   #   Plotly wrapper (never inside motion.div+layout)
│   │   ├── components/
│   │   │   └── Hero.tsx         #   shared hero motion
│   │   ├── features/
│   │   │   ├── topbar/TopBar.tsx
│   │   │   ├── landing/LaunchChoiceModal.tsx   # NEW
│   │   │   ├── builder/                        # NEW (Phase 6)
│   │   │   │   ├── BuilderWorkspace.tsx
│   │   │   │   ├── CategoryRail.tsx
│   │   │   │   ├── SchematicCanvas.tsx         # react-flow + custom nodes
│   │   │   │   ├── ResultsMiniPanel.tsx        # inline mini + "Open full →"
│   │   │   │   ├── ModalShell.tsx              # shared draft→commit lifecycle
│   │   │   │   ├── NoiseOverlay.tsx            # animated grain + beam distortion
│   │   │   │   ├── nodes/{TxNode,ChannelNode,RxNode,NodeShell,nodeTypes}
│   │   │   │   └── modals/{Transmitter,Receiver,Geometry,Noise}Modal.tsx
│   │   │   ├── setup/{ChannelCanvas,GeometrySliders,LinkBudgetTable}.tsx
│   │   │   ├── engine/{StepCard,PipelineCards,MetricsCard}.tsx
│   │   │   │   #   MetricsCard now shows Throughput + Theory companion + Wilson CI
│   │   │   ├── results/
│   │   │   │   ├── WaveformsPanel.tsx
│   │   │   │   ├── BitsComparePanel.tsx
│   │   │   │   ├── EyeDiagramPanel.tsx          # NEW (eye + metric chips)
│   │   │   │   ├── ConstellationPanel.tsx      # NEW (decision scatter)
│   │   │   │   ├── SpectrumPanel.tsx           # NEW (FFT power spectrum)
│   │   │   │   ├── NoiseBreakdownPanel.tsx     # NEW (per-source bar + table)
│   │   │   │   ├── BodePanel.tsx               # NEW (AC: mag/phase + GM/PM)
│   │   │   │   ├── ExportButton.tsx            # NEW (Tauri-only save-to-folder)
│   │   │   │   └── ResultsPanel.tsx
│   │   │   ├── workbench/{ComponentLibrary,ComponentParamTable,SystemSetupForm}.tsx
│   │   │   ├── sweeps/{SweepControls,SweepProgress,SweepChart,ToleranceTable}.tsx
│   │   │   │   #   four sweep kinds; auto-tolerance table when active
│   │   │   └── standards/                       # NEW (Phase 7)
│   │   │       ├── StandardsModal.tsx
│   │   │       └── ComplianceBanner.tsx
│   │   └── routes/
│   │       ├── LandingRoute.tsx     # NEW — default
│   │       ├── BuilderRoute.tsx     # NEW — wraps BuilderWorkspace
│   │       ├── SetupRoute.tsx       # legacy, still reachable
│   │       ├── EngineRoute.tsx
│   │       ├── AcRoute.tsx          # NEW
│   │       └── SweepsRoute.tsx
│   └── src-tauri/               #   Rust shell
│       ├── Cargo.toml
│       ├── build.rs
│       ├── tauri.conf.json      #   window config, sidecar allowlist, CSP
│       ├── icons/               #   32x32.png, 128x128.png, icon.ico, icon.icns
│       ├── binaries/            #   lifi-backend-<target-triple>.exe (built)
│       └── src/main.rs          #   sidecar lifecycle (spawn, health, inject port)
│
├── scripts/
│   └── build-sidecar.ps1        #   PyInstaller build wrapper
│
├── workspace/                   # session output dirs (gitignored)
└── venv/                        # Python virtual env
```

---

## 5. Module responsibilities (quick reference)

### Backend (FastAPI)

| Endpoint | Maps to |
|---|---|
| `GET /api/presets` | `SystemConfig.list_presets()` |
| `GET /api/presets/{name}` | `SystemConfig.from_preset(name).to_dict()` |
| `POST /api/config/validate` | `SystemConfig._from_flat_dict(payload)` |
| `GET /api/components` | iterate `COMPONENT_REGISTRY` |
| `GET /api/components/categories` | unique categories from `get_parameters()['type']` |
| `GET /api/components/{part}` | `get_component(part).get_parameters()` |
| `POST /api/link-budget` | `OpticalChannel(...).propagate(P_tx)` + SNR calc |
| `WS /ws/pipeline` | `SimulationPipeline(cfg, ...).run_all()` in thread pool. Streams `step` / `metrics` (incl. `noise_breakdown`) / `waveforms` / `compliance` (if `profile_id` was on the start frame) |
| `WS /ws/sweep` | Thermal, Monte Carlo (with optional `auto_tolerance` + `tolerance_resolved` message), `ber_snr`, `ber_distance` generators |
| `WS /ws/ac` | `cosim/ac_analysis.py::run_ac_analysis(cfg, …)` — Bode + GM/PM in one batch payload |
| `GET /api/standards` | List of PHY compliance profiles (only IEEE 802.15.7 PHY-I today) |
| `GET /api/standards/{id}` | Full profile incl. overrides + criteria labels |
| `GET /health` | sidecar readiness probe (used by Tauri) |

### Frontend (React)

| Route | Purpose |
|---|---|
| `landing` (default) | Splash with single **Builder** CTA. Opens the LaunchChoiceModal (preset vs build-your-own). Hides the top bar. |
| `builder` | CategoryRail (TX / RX / Geometry / Noise) + react-flow SchematicCanvas + ResultsMiniPanel + Standards picker + Simulate / Precision-run buttons |
| `setup` | Legacy: hero + channel canvas + sliders + link budget. Still reachable from top bar. |
| `engine` | Streaming step cards + MetricsCard (incl. Theory companion + Wilson CI + Throughput) + ResultsPanel (Waveforms, Eye, ConstellationDecisionScatter, Spectrum, NoiseBreakdown, BitsCompare). ExportButton + ComplianceBanner in the header. |
| `ac` | "Run AC sweep" → BodePanel (mag/phase subplots, GM/PM/BW chips, crosshairs at f_unity and f_180) |
| `sweeps` | Thermal · Monte Carlo (with auto-tolerance) · BER vs SNR · BER vs distance — all live-streaming with Wilson 95% CI bars and analytical overlay |

### Animation contract

All routes animate via `opacity` + `y` (both transforms) inside
`<LayoutGroup>` + `<AnimatePresence mode="wait">`. Shared `EASE` token
(cubic-bezier(0.4, 0, 0.2, 1)) and `DUR` tokens (`fast: 0.18`, `base: 0.28`,
`slow: 0.45`).

`PlotCanvas` is intentionally **not** wrapped in a `motion.div` with `layout`
to avoid Plotly's internal ResizeObserver firing every animation frame.

---

## 6. Build status

| Artifact | Status | Location |
|---|---|---|
| Python backend module (`python -m backend`) | ✅ Works | `backend/__main__.py` |
| FastAPI app smoke test | ✅ All 12 routes register, REST + WS verified | — |
| Frozen sidecar binary | ✅ Built (71.8 MB) | `dist/lifi-backend.exe` |
| Tauri sidecar binary (with target triple) | ✅ Copied | `frontend/src-tauri/binaries/lifi-backend-x86_64-pc-windows-msvc.exe` |
| Frontend Vite build (`npm run build`) | ✅ Builds cleanly (after `@types/node` + `vite-env.d.ts` fixes) | `frontend/dist/` |
| Tauri icons | ✅ Generated from placeholder source.png | `frontend/src-tauri/icons/` |
| Rust toolchain | ✅ rustc 1.95.0, cargo 1.95.0 | — |
| MSVC linker (`link.exe`) | ✅ Available via `vcvars64.bat` (MSVC 14.51.36231 + Windows 11 SDK 10.0.26100.0) | — |
| Final `.msi` installer | ✅ Built | `dist\installer\LiFi PV Simulator_0.1.0_x64_en-US.msi` |
| NSIS `.exe` installer | ✅ Built (bonus output from `targets: all`) | `dist\installer\LiFi PV Simulator_0.1.0_x64-setup.exe` |

### Reproducing the build

```powershell
# 1. Disable Avast shields for 10 minutes (tray icon → Avast shields control)
# 2. From repo root:
& "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$env:CARGO_TARGET_DIR = "D:\cargo-target\lifi-pv"
cd frontend
npm run tauri:build
# Output:
#   D:\cargo-target\lifi-pv\release\bundle\msi\LiFi PV Simulator_0.1.0_x64_en-US.msi
#   D:\cargo-target\lifi-pv\release\bundle\nsis\LiFi PV Simulator_0.1.0_x64-setup.exe
#   D:\cargo-target\lifi-pv\release\LiFi PV Simulator.exe (portable)
```

---

## 7. How to run today

### Legacy GUI (still the most-validated path)

```powershell
.\venv\Scripts\python.exe cli.py gui
```

### New React UI in dev mode (browser, two-terminal)

```powershell
# Terminal 1 — backend
.\venv\Scripts\python.exe -m uvicorn backend.app:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
# → http://localhost:5173
```

### CLI (always available)

```powershell
.\venv\Scripts\python.exe cli.py test          # 14 module self-tests
.\venv\Scripts\python.exe cli.py validate      # generate 23 paper figures
.\venv\Scripts\python.exe cli.py pipeline --preset kadirvelu2021
.\venv\Scripts\python.exe cli.py compare       # cross-preset validation
.\venv\Scripts\python.exe cli.py components    # list 14 components
```

---

## 8. Validation reference points

| Preset | Modulation | Data rate | Target BER | Notes |
|---|---|---|---|---|
| `kadirvelu2021` | OOK_Manchester | 2.5 kbps | < 1e-3 | Primary reference design |
| `sarwar2017` | OFDM (256-FFT, 16-QAM) | ~5 Mbps | — | 80 subcarriers |
| `xu2024` | BFSK | — | — | Multi-cell reconfigurable PV |
| `oliveira2024` | OFDM | — | — | Enhanced RX with notch filtering |
| `gonzalez2024` | OOK_Manchester | — | — | Temperature-compensated |
| `correa2025` | PWM_ASK | — | — | PWM-ASK with DC-DC integration |
| `lifi_poc_breadboard` | PWM_ASK | 1 kbps | 1e-3 | Starter-kit build: 5 mm LED + small PV + TL072 dual-stage + ESP32. `rx_topology=amp_slicer`, `bpf_stages=0`, `dcdc_enable=false`. `n_bits=20` so BER is informational only — bump n_bits for any publishable claim. See [lifi_poc_breadboard_simulation_notes.txt](lifi_poc_breadboard_simulation_notes.txt) for a per-block parameter walkthrough. |

Pipeline validation: **20% error threshold** for PASS. Per CLAUDE.md, 5 of 7
pipeline validations currently fail (independent bugs flagged as next-after-
refactor work).

---

## 9. Known gaps / next work

### Resolved since 2026-05-16
- ✅ **Eye-diagram + BER-vs-SNR + spectrum + decision-scatter + noise
  breakdown** Plotly panels — all landed in Phase 7.
- ✅ **SPICE results-panel bug** — `backend/spice_waveforms.py` now parses
  the `.raw` file and emits the same `waveforms` shape as the Python path,
  so EngineRoute populates regardless of engine.
- ✅ **AC analysis route** — closed-form Bode + GM/PM via
  `cosim/ac_analysis.py` (no SPICE round-trip needed).
- ✅ **PHY compliance picker (MVP)** — IEEE 802.15.7 PHY-I OOK profile +
  PASS/FAIL banner; the architecture is open for additional PHY modes.

### Still open
1. **Rebuild the `.exe` / MSI** — the artifacts in `dist/installer/` were
   built on 2026-05-16 and predate the Phase 6/7 work (Landing, EDA panels,
   AC analysis, Standards). They install but show the old UI. Re-run the
   Reproducing-the-build steps in §6 before any distribution.
2. **Code signing** — the current MSI/NSIS/EXE are unsigned; Windows shows
   a SmartScreen warning on first launch. For public distribution this
   needs an Authenticode certificate (~$200-400/yr from DigiCert, Sectigo,
   etc.) and the `windows.certificateThumbprint` / `timestampUrl` fields
   filled in `frontend\src-tauri\tauri.conf.json`.
3. **5 failing pipeline validations** — flagged in CLAUDE.md memory as the
   next refactor target after this UI migration lands. Not blocked by
   anything in Phase 6/7.
4. **SPICE-via-sidecar** — the bundled `lifi-backend.exe` does not include
   LTspice or ngspice. The pipeline auto-falls back to Python; full SPICE
   simulation requires the user to have LTspice or ngspice installed
   separately. Acceptable degradation for the first .exe but should be
   addressed (option: ship ngspice in `bundle.resources`).
5. **Additional PHY profiles** — IEEE 802.15.7 PHY-II/III, IEEE 802.11bb,
   ITU-T G.9991. Each is ~1 day on top of the existing `PhyProfile`
   architecture (`cosim/standards/`).
6. **Constellation diagram for OFDM/BFSK** — the current
   `ConstellationPanel` shows a decision scatter for amplitude-modulated
   schemes (OOK / Manchester / PWM-ASK). True I/Q constellations need the
   demodulator's complex symbols exposed over the WebSocket; tracked as a
   small `cosim/python_engine.py` patch.
7. **KiCad export wired to the UI** — `kicad/` module is fully functional
   from the CLI but no backend route or frontend button exists. Plan in
   `C:\Users\HP OMEN\.claude\plans\i-am-building-a-zippy-nebula.md` Part 3
   describes the ~1-hour backend route + frontend button work and the
   per-system graph-builder follow-ups.
8. **InspectorDock (Schematics / Validation / Message sub-panels)** —
   designed but not implemented in the React UI. Schematics in particular
   requires a replacement for the schemdraw → QPixmap rendering path used
   in `gui/tab_schematics.py`.
9. **PM / GM convention on BodePanel** — the AC analysis correctly extracts
   phase margin from the open-loop cascade, but for a band-pass response
   (no feedback loop) the numbers can read negative and confuse a reader
   expecting low-pass conventions. Add an informational note in the panel.
10. **Bump `n_bits` for the validation harnesses** — Phase 7 raised the
    default to 10000, but `papers/*.py` still pin their own values for
    reproducibility. Worth a separate "publication-grade" sweep that uses
    100k+ bits + analytical extrapolation per paper.

---

## 10. Key conventions

- Paper parameters are **locked** — never modify values from publications.
- Component values come from datasheets, not approximations.
- All figures publication-quality (≥150 DPI, matplotlib).
- Validation threshold: **20% error** for PASS.
- `SystemConfig.from_preset(name)` is the canonical config entrypoint.
- React animations: transform/opacity only (never width/height/top/left).
- `PlotCanvas` never inside `motion.div` with `layout`.

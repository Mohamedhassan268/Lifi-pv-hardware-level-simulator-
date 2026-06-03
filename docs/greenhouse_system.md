# Smart-Greenhouse System Architecture & Simulation Roadmap

**Status:** 2026-06-04 — architecture map + **first SLIPT results (§5)**. The
data↔energy node model (milestone A) and a first cut of greenhouse-condition
sensitivity (milestone B) are built and run. The multi-node canvas (Path C) and
the LoRa budget block remain future work.

---

## 1. The system

A self-powered smart greenhouse built from **nodes**. Each node = sensors → MCU
→ a **LiFi/PV** front-end where the photovoltaic cell does double duty: it
**receives data and harvests power**. Nodes interconnect over **802.11bb** (the
LiFi standard), aggregate up to a gateway over **LoRa** (long-range sub-GHz),
and a **dashboard** monitors everything.

```
  [sensors]                [sensors]                 gateway
     |                        |                          |
   [MCU]                    [MCU]                         |
     |  LiFi / PV link         |  LiFi / PV link          |
     +----- 802.11bb hop ------+------ 802.11bb hop ------>+--- LoRa backhaul ---> [Dashboard]
   Tier 1 (edge)            Tier 1 (edge)              Tier 2/3
```

## 2. Layer responsibilities — what THIS simulator owns

The trap is turning a PHY engine into a packet network simulator (that is ns-3's
job). We compose our faithful PHY **upward into budgets**, not packets.

| Tier | Element | Fidelity here | Owned by us? |
|---|---|---|---|
| 1 | Sensor + MCU + **LiFi/PV link** | Full hardware PHY: circuit, 6-source noise, BER/SNR, **harvested power**, MCU ADC | ✅ yes (Phases 1–19) |
| 2 | Node↔node **802.11bb** | PHY only (`ieee_802_11bb` preset); no MAC/multi-access | ◑ PHY yes, MAC no |
| 3 | **LoRa** backhaul | **Link budget only** (Tx power, path loss, SF→sensitivity, airtime/duty) — NOT a faithful chirp PHY | ◑ budget block (new) |
| 4 | **Dashboard** | Data sink / visualization | ✗ out of scope |

**Differentiator:** nobody else models the **data + energy duality** of the
PV-as-receiver. Node energy autonomy is the question a self-powered greenhouse
actually lives or dies on, and it falls straight out of our `harvested_power_W`
feature.

## 3. Path C — the canvas model

Reuse what exists rather than inventing a network sim:

- **Phase 17** already gives a two-system (A/B) block diagram with coupling
  modes (none / cross-links / shared channel) — the foothold for N nodes.
- **Phase 18** MCU block (board profiles, ADC band-limiting) = the node controller.
- **Phase 19/20** feature export (`extract_features` → 33 metrics incl.
  `harvested_power_W`, `goodput_bps`, the three latencies) = the per-link
  characterization each canvas node emits.

New block types to add to the canvas, one per tier:

| Block | Models | Built from |
|---|---|---|
| **Edge node** | sensor draw + MCU + LiFi/PV link at a (x,y / distance,angle) | existing single-link pipeline + `harvested_power_W` |
| **802.11bb hop** | node→node optical link | `ieee_802_11bb` preset, existing channel model |
| **LoRa link-budget** | backhaul reachability + airtime | NEW thin block: Pt, path-loss model, SF→sensitivity table |
| **Gateway / sink** | aggregation point (terminates LoRa) | trivial sink node |

## 4. Staged milestones (each answers one of your three questions)

Build incrementally; every milestone is shippable and reuses the prior.

- **M1 — Coverage / link closure.** Place N edge nodes at positions; each runs
  our LiFi (and/or 802.11bb) PHY at its distance/angle/ambient; render a
  link-closure map (which nodes meet a BER/goodput threshold). *Pure
  composition of the existing single-link engine + a layout.*

- **M2 — Energy autonomy per node.** For each node, roll up PV `harvested_power_W`
  vs. the sensor + MCU + radio draw over a day/night illuminance profile →
  autonomy verdict (self-sustaining? hours of buffer?). *Our unique
  data+energy story; reuses harvested-power feature + an ambient profile.*

- **M3 — End-to-end latency & throughput.** Chain the tiers (edge LiFi → 802.11bb
  hop → LoRa budget) and sum the per-tier latency/goodput budgets to a
  sensor→dashboard figure. *Brings in the LoRa budget block; still a budget,
  not packets.*

For anything genuinely packet-level (contention, routing, retransmission across
many nodes), the clean hand-off is **Path A**: export our per-link curves
(already possible via `cli.py features` / `/api/features`) into ns-3 or your
collaborator's network sim as its PHY layer.

## 5. First results — the SLIPT node (milestones A + B)

Built: `presets/slipt_node.json` (PV harvesting node, `pv_ode_enable: true` so
harvest is real, not NaN); `r_sense_ohm`/`r_load_ohm` added as load levers in
`cosim/feature_sweep.py`; `scripts/slipt_analysis.py` (both figures). Reproduce:

```
python scripts/slipt_analysis.py                 # figures -> workspace/slipt/
python cli.py features --preset slipt_node --sweep r_sense_ohm --values 500,1000,1500,3000
```

### A — Data↔energy Pareto (PV load operating point, at 0 lux)

Sweeping the PV load `r_sense_ohm` walks the operating point short-circuit → MPP
→ open-circuit:

| R (Ω) | harvest (µW) | BER | note |
|---|---|---|---|
| ≤300 | 5–33 | NaN | signal too weak to demod (degenerate) |
| 500 | 54 | 8e-4 | **best data** |
| 1000 | 106 | 5.6e-3 | |
| **1500** | **143** | 7.6e-3 | **MPP (max harvest)** |
| 3300 | 92 | 5.2e-3 | |
| ≥6800 | ≤48 | 3.6e-3 | toward open-circuit |

- Harvest peaks at MPP (~1500 Ω, 143 µW), right where the cell's Vmpp/Impp predicts.
- **The data↔energy coupling is the PV's own IV nonlinearity:** driving the
  operating point toward MPP/Voc for power pushes the OOK swing into the
  nonlinear knee and distorts it. Best data ≈ 500 Ω (BER 8e-4 @ 54 µW); chasing
  MPP buys ~2.6× the harvest at ~10× the BER.
- Tightening this corrected a first-probe artifact: low-R points looked
  error-free at 800 bits but are degenerate (NaN) at higher bit counts.

### B — Greenhouse conditions (at R = 1000 Ω)

**Ambient light (night → daylight):**

| lux | BER | harvest (µW) |
|---|---|---|
| 0 | 0.007 | 106 |
| 100 | 0.31 | 339 |
| 1000 | 0.50 | 478 |
| 10000 | 0.51 | 612 |
| 50000 | 0.51 | 714 |

1. **Ambient light destroys the link fast** — by 100 lux (dim indoor) BER is
   already ~0.31; at daylight (≥1000 lux) it is coin-flip. As configured, this
   node only communicates near-dark.
2. **Harvest rises with ambient** — ambient is modeled as DC optical power on the
   cell (`1 lux ≈ 1.46 µW/cm²`, the same conversion as the shot-noise source),
   shifting the operating point toward Voc and adding harvest: 106 µW (dark) →
   339 (100 lux) → 714 µW (50 klux). The data path stays on the LED-only solve,
   so ambient's *data* impact remains the shot-noise term — no double-count.
   *Design tension:* **daylight is a harvest boon but a comms liability** — the
   node wants optical/IR band-filtering on the data detector while the harvester
   sees full-spectrum light.

**Distance (at night):**

| dist (m) | BER | harvest (µW) |
|---|---|---|
| 0.1 | NaN | 440 (clips) |
| 0.2 | 1.3e-3 | 330 |
| 0.325 | 9e-3 | 106 |
| 0.5 | 8e-3 | 19 |
| 1.0 | 8e-3 | 1.2 |

Data holds (~1e-2 floor) across 0.2–1.0 m at night, but **harvest collapses
~280× from 330 µW (0.2 m) to 1.2 µW (1.0 m)** — the narrow 9° LED beam.
**Energy autonomy is distance-critical: a node must sit ≲0.3 m to harvest
usefully**, even where data still closes.

### Caveats / next

- PV harvest sweeps **used to be slow**; the PV ODE now uses a numba fixed-step
  linearly-implicit integrator (`cosim/pv_model.py`) — ~870× faster solve, ~24×
  per run end-to-end, matching the Radau reference to <0.01% on V/harvest. A full
  Pareto + greenhouse run is now ~30 s. (Radau still available via `simulate(fast=False)`.)
- `harvested_power_W` is raw cell power; the DC-DC *output* is lower (and is
  load-regulated, so it can stay flat while raw harvest rises).
- Ambient shifts the **harvest** operating point, but the **data** demod still
  runs on the LED-only operating point (ambient's data impact = the shot-noise
  term); coupling the demod to the ambient-shifted operating point is a future
  refinement.
- BER at modest `n_bits` (1.5–2.5k) is noisy below ~1e-3.
- Figures: `workspace/slipt/slipt_pareto.png`, `workspace/slipt/slipt_greenhouse.png`.

## 6. Explicit non-goals

- No MAC / multi-access contention modeling (ns-3's domain).
- No faithful LoRa chirp PHY — link budget only.
- No routing/mesh protocol simulation.
- The dashboard is consumed data, not simulated.

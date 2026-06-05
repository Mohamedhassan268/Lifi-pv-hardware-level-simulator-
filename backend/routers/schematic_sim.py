"""POST /api/schematic-sim — simulate a user-drawn circuit graph.

Pipeline:
  1. ERC the incoming graph (cosim.erc). Errors -> 422 with issue list.
  2. Serialise the graph into SPICE RX instance lines (cosim.graph_netlist).
  3. Run the Kadirvelu TX+Channel context with those RX lines injected
     (SimulationPipeline.rx_instances_override), reusing the validated
     subcircuit definitions and BER extraction.
  4. Return BER + any ERC warnings.

This is the backend half of the drag-and-drop round trip; the headless
equivalence (graph -> netlist -> solver -> validated BER) is proven in
cosim/graph_netlist.py.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cosim.erc import check_graph, has_errors
from cosim.graph_netlist import graph_dict_to_instances
from cosim.system_config import SystemConfig
from cosim.pipeline import SimulationPipeline
from cosim import pyspice_graph_runner
from cosim.schematic_cosim import run_two_pass, run_analog_link, run_pwm_ask_link
from components import get_component

router = APIRouter()

# Component types whose .SUBCKT the pipeline already emits, plus primitives
# that are bare SPICE cards (no subcircuit). Anything else must supply its own
# definition via the component's spice_subcircuit().
_PIPELINE_OWNED = {"SOLAR_CELL", "INA", "BPF_STAGE", "COMPARATOR", "R", "V"}


def _collect_subckt_defs(graph: dict) -> str:
    """Return concatenated .SUBCKT bodies for placed parts not owned by the
    pipeline, de-duplicated by component type."""
    defs: dict[str, str] = {}
    for comp in graph.get("components", []):
        ctype = comp.get("component_type", "")
        if ctype in _PIPELINE_OWNED or ctype in defs:
            continue
        try:
            part = get_component(ctype)
        except KeyError:
            continue
        if hasattr(part, "spice_subcircuit"):
            defs[ctype] = part.spice_subcircuit()
    return "\n".join(defs.values())


class CircuitGraphIn(BaseModel):
    title: str = "User circuit"
    components: list[dict[str, Any]]
    nets: list[dict[str, Any]]
    # Optional operating point for the two-pass run (else the reference defaults).
    distance_m: float | None = None
    data_rate_bps: float | None = None
    n_bits: int | None = None
    modulation: str | None = None  # "OOK" or "OOK_Manchester" (line-code schemes)
    # PHY overrides parsed from uploaded ESP firmware (.ino). Applied to the
    # PWM-ASK operating point when present.
    carrier_freq_hz: float | None = None
    pwm_freq_hz: float | None = None
    modulation_depth: float | None = None
    adc_bits: float | None = None
    mcu_sample_rate_hz: float | None = None
    bias_current_A: float | None = None
    led_radiated_power_mW: float | None = None
    prbs_order: float | None = None
    mcu_clock_MHz: float | None = None
    adc_vref: float | None = None
    # User-placed instrument probes (multimeter / scope), each tapping one net.
    # Excluded from the SPICE netlist by the client; we report each net's voltage.
    probes: list[dict[str, Any]] = []


class SchematicSimResponse(BaseModel):
    ber: float | None = None
    ber_incircuit: float | None = None
    n_bits: int | None = None
    message: str | None = None
    warnings: list[str] = []
    diagnostics: dict[str, Any] | None = None
    probes: list[dict[str, Any]] = []


def _assemble_probes(req_probes: list[dict], packed: dict | None) -> list[dict]:
    """Map each requested probe to its solved net data (DC for the multimeter,
    DC + trace for the scope). Probes whose net isn't in the result are marked
    not-found rather than dropped, so the UI can flag them."""
    nets = (packed or {}).get("nets", {})
    out: list[dict] = []
    for p in req_probes:
        net = (p.get("net") or "")
        nd = nets.get(net.lower())
        is_scope = p.get("kind") == "scope"
        out.append({
            "id": p.get("id"),
            "kind": p.get("kind"),
            "net": net,
            "found": nd is not None,
            "dc": nd["dc"] if nd else None,
            "min": nd["min"] if nd else None,
            "max": nd["max"] if nd else None,
            "trace": nd["trace"] if (nd and is_scope) else None,
            "time": nd["t"] if (nd and is_scope) else None,
        })
    return out


@router.post("", response_model=SchematicSimResponse)
def simulate(graph: CircuitGraphIn) -> SchematicSimResponse:
    g = graph.model_dump()
    user_nets = [p.get("net") for p in graph.probes if p.get("net")]

    # 1. ERC
    issues = check_graph(g)
    warnings = [i.message for i in issues if i.level == "warning"]
    if has_errors(issues):
        errs = [i.message for i in issues if i.level == "error"]
        raise HTTPException(status_code=422, detail={"errors": errs, "warnings": warnings})

    # 1b. Two-pass path: if the graph draws a TX (LED) and an RX (photodetector),
    # run the full TX-SPICE -> Python channel -> RX-SPICE co-simulation. Returns
    # None when it isn't a two-pass link, in which case we fall back to the
    # RX-only path below (hardcoded Python TX + channel). The operating point
    # (distance / data rate / bit count) comes from the request when supplied,
    # else the Kadirvelu reference defaults; values are clamped to sane ranges.
    _mod = (graph.modulation or "OOK").upper().replace("-", "_")
    if _mod in ("OFDM", "BFSK"):
        # Continuous-waveform schemes: linear-driver TX + SPICE RX analog front
        # end + Python DSP demod. These need a fast-photodiode RX (the
        # "OFDM / BFSK Link" preset); the kadirvelu close-link budget + the drawn
        # parts give the operating point. OFDM uses a moderate rate the passive
        # photodiode front-end passes (15 Mbps OFDM needs a TIA gain stage —
        # documented gap). Distance / rate / bit count are overridable.
        if _mod == "OFDM":
            # Close kadirvelu link budget (matches the drawn LED + fast
            # photodiode) + OFDM params at a moderate, passable rate.
            _d = SystemConfig.from_preset("kadirvelu2021").to_dict()
            _d.update(modulation="OFDM", dcdc_enable=False, modulation_depth=0.2,
                      ofdm_nfft=256, ofdm_qam_order=16, ofdm_n_subcarriers=80, ofdm_cp_len=32)
            _d["data_rate_bps"] = float(min(max(graph.data_rate_bps or 200_000.0, 1e3), 1e6))
            # >=3 OFDM symbols so the pilot preamble leaves data to score.
            _d["n_bits"] = int(min(max(graph.n_bits or 1280, 960), 4000))
        else:  # BFSK: xu2024's tone/rate/link budget (validated)
            _d = SystemConfig.from_preset("xu2024").to_dict()
            _d.update(modulation="BFSK", dcdc_enable=False)
            _d["n_bits"] = int(min(max(graph.n_bits or 200, 64), 4000))
        if graph.distance_m is not None:
            _d["distance_m"] = float(min(max(graph.distance_m, 0.01), 50.0))
        two = run_analog_link(g, SystemConfig(**_d))
    elif _mod == "PWM_ASK":
        # Breadboard PoC: SPICE TX (2N2222 -> LED) + SPICE RX gain chain on a
        # split +/-5 V supply + Python envelope demod. Operating point comes
        # from the lifi_poc_breadboard preset (10 kHz carrier, +/-5 V rails).
        _d = SystemConfig.from_preset("lifi_poc_breadboard").to_dict()
        _d["dcdc_enable"] = False
        _d["n_bits"] = int(min(max(graph.n_bits or 48, 24), 400))
        if graph.distance_m is not None:
            _d["distance_m"] = float(min(max(graph.distance_m, 0.01), 50.0))
        if graph.data_rate_bps is not None:
            _d["data_rate_bps"] = float(min(max(graph.data_rate_bps, 100.0), 1e5))
        # Firmware-parsed PHY overrides (from the uploaded .ino), clamped sane.
        if graph.carrier_freq_hz is not None:
            _d["carrier_freq_hz"] = float(min(max(graph.carrier_freq_hz, 100.0), 1e6))
        if graph.pwm_freq_hz is not None:
            _d["pwm_freq_hz"] = float(max(graph.pwm_freq_hz, 0.0))
        if graph.modulation_depth is not None:
            _d["modulation_depth"] = float(min(max(graph.modulation_depth, 0.0), 1.0))
        if graph.adc_bits is not None:
            _d["adc_bits"] = int(min(max(graph.adc_bits, 1), 24))
        if graph.mcu_sample_rate_hz is not None:
            _d["mcu_sample_rate_hz"] = float(max(graph.mcu_sample_rate_hz, 0.0))
        if graph.bias_current_A is not None:
            _d["bias_current_A"] = float(min(max(graph.bias_current_A, 0.0), 1.0))
        if graph.led_radiated_power_mW is not None:
            _d["led_radiated_power_mW"] = float(max(graph.led_radiated_power_mW, 0.0))
        if graph.prbs_order is not None:
            _d["prbs_order"] = int(min(max(graph.prbs_order, 3), 31))
        if graph.mcu_clock_MHz is not None:
            _d["mcu_clock_MHz"] = float(max(graph.mcu_clock_MHz, 0.0))
        if graph.adc_vref is not None:
            _d["adc_vref"] = float(min(max(graph.adc_vref, 0.1), 12.0))
        two = run_pwm_ask_link(g, SystemConfig(**_d), user_probe_nets=user_nets)
    else:
        # 2-level line codes (OOK, Manchester) run through the in-circuit
        # comparator path; anything unrecognised falls back to OOK.
        _d = SystemConfig.from_preset("kadirvelu2021").to_dict()
        _d["modulation"] = "OOK_Manchester" if _mod in ("OOK_MANCHESTER", "MANCHESTER") else "OOK"
        _d["dcdc_enable"] = False
        _d["n_bits"] = int(min(max(graph.n_bits or 128, 20), 2000))
        if graph.distance_m is not None:
            _d["distance_m"] = float(min(max(graph.distance_m, 0.01), 50.0))
        if graph.data_rate_bps is not None:
            _d["data_rate_bps"] = float(min(max(graph.data_rate_bps, 100.0), 1e6))
        two = run_two_pass(g, SystemConfig(**_d), user_probe_nets=user_nets)
    if two is not None:
        return SchematicSimResponse(
            ber=two.get("ber"),
            ber_incircuit=two.get("ber_incircuit"),
            n_bits=two.get("n_bits"),
            message=two.get("message"),
            warnings=warnings,
            diagnostics=two.get("diagnostics"),
            probes=_assemble_probes(graph.probes, two.get("probes")),
        )

    # 2. Serialise to SPICE instance lines
    try:
        instances = graph_dict_to_instances(g)
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"errors": [str(e)], "warnings": warnings}) from e

    # 2b. Gather .SUBCKT definitions for each placed library part. Primitives
    # (R/V) and the pipeline-owned types (INA/BPF_STAGE/COMPARATOR/SOLAR_CELL)
    # are already defined by the pipeline; everything else asks its component.
    extra_defs = _collect_subckt_defs(g)

    # 3. Build the optical input (Python TX + channel — fast, deterministic),
    # then simulate the user's drawn circuit in-process via libngspice. LTspice
    # batch mode nondeterministically hangs on these behavioral subcircuits;
    # libngspice runs in-process with full control and never spawns a worker.
    cfg = SystemConfig.from_preset("kadirvelu2021")
    d = cfg.to_dict()
    d["dcdc_enable"] = False
    d["n_bits"] = 200  # enough to show the circuit works; short transient
    cfg = SystemConfig(**d)

    if not pyspice_graph_runner.available():
        raise HTTPException(
            status_code=503,
            detail={"errors": ["No in-process SPICE engine (libngspice) available."],
                    "warnings": warnings},
        )

    # Python TX + channel to get the optical PWL and the transmitted bits.
    session = Path(tempfile.mkdtemp(prefix="optisim_sch_"))
    try:
        for sub in ("pwl", "netlists", "raw", "plots"):
            (session / sub).mkdir(parents=True, exist_ok=True)
        pipe = SimulationPipeline(cfg, session)
        pipe.run_step_tx()
        pipe.run_step_channel()
        if pipe._time is None or pipe._P_rx is None or pipe._tx_bits is None:
            raise HTTPException(status_code=500,
                                detail="TX/Channel did not produce an optical waveform")

        result = pyspice_graph_runner.run_graph(
            subckt_defs=extra_defs,
            instance_lines=instances,
            optical_t=pipe._time,
            optical_v=pipe._P_rx,
            vcc_volts=cfg.vcc_volts,
            t_stop_s=cfg.t_stop_s,
        )
    finally:
        shutil.rmtree(session, ignore_errors=True)

    ber = None
    message = "libngspice simulation complete"
    if result is None:
        message = "Simulation did not produce V(dout) (check circuit wiring)."
    else:
        v_dout, t_dout, _extras = result
        from simulation.analysis import calculate_ber_from_transient
        bit_period = 1.0 / cfg.data_rate_bps
        ber_res = calculate_ber_from_transient(
            tx_bits=pipe._tx_bits,
            rx_waveform=v_dout,
            time=t_dout,
            threshold=cfg.vcc_volts / 2.0,
            bit_period=bit_period,
            skip_bits=2,
        )
        ber = ber_res["ber"]

    return SchematicSimResponse(
        ber=ber,
        n_bits=cfg.n_bits,
        message=message,
        warnings=warnings,
    )

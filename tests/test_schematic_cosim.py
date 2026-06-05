"""Tests for the single-canvas two-pass co-simulator (cosim/schematic_cosim.py).

Milestone 1: a drawn TX (drive -> MOSFET -> LED) -> Python optical channel ->
drawn RX (PV -> sense -> instrumentation amp -> average-value comparator) closes
an OOK link in-circuit, with the validated 6-source noise applied in post.

The headline guarantee here is that the honest BER is *not* trivially zero: as
the link distance grows, P_rx falls, the comparator-input SNR drops, and the
noise model drives BER up — while the noiseless in-circuit recovery stays clean
(proving the errors come from noise, not a circuit failure). Skips if libngspice
isn't available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from cosim import pyspice_graph_runner, tx_spice_runner  # noqa: E402
from cosim.schematic_cosim import run_two_pass  # noqa: E402
from cosim.system_config import SystemConfig  # noqa: E402

spice_unavailable = pytest.mark.skipif(
    not (tx_spice_runner.available() and pyspice_graph_runner.available()),
    reason="libngspice not loadable (run scripts/install_pyspice.py)",
)


def _comp(ref, ctype, pins, value=""):
    return {"ref": ref, "component_type": ctype, "value": value,
            "pins": {i + 1: n for i, n in enumerate(pins)}}


def _net(name, *pins):
    return {"name": name, "pins": [{"component_ref": r, "pin_number": p} for r, p in pins]}


def ook_link_graph() -> dict:
    """The milestone-1 reference OOK link: drawn TX -> channel -> drawn RX with
    an average-value comparator slicer."""
    return {
        "title": "OOK link",
        "components": [
            _comp("Vdrv", "DRIVE", ["out"]),
            _comp("Xled", "LXM5_PD01", ["anode", "cathode"]),
            _comp("Xmos", "BSD235N", ["drain", "gate", "source"]),
            _comp("Rlimit", "R", ["1", "2"], "150"),
            _comp("Ch1", "CHANNEL", ["tx", "rx"]),
            _comp("Xsc", "SM141K", ["anode", "cathode", "photo_in"]),
            _comp("Rsense", "R", ["1", "2"], "10k"),
            _comp("Vgnd", "V", ["1", "2"], "0"),
            _comp("Xina", "INA322", ["INP", "INN", "OUT", "VCC", "VEE", "REF"]),
            _comp("Xcmp", "TLV7011", ["INP", "INN", "OUT", "VCC", "VEE"]),
            _comp("Ravg", "R", ["1", "2"], "1Meg"),
            _comp("Cavg", "C", ["1", "2"], "2n"),
            _comp("OUT1", "OUT", ["out"]),
            _comp("GND1", "GND", ["gnd"]),
        ],
        "nets": [
            _net("gate", ("Vdrv", 1), ("Xmos", 2)),
            _net("led_a", ("Rlimit", 2), ("Xled", 1)),
            _net("led_k", ("Xled", 2), ("Xmos", 1)),
            _net("vcc", ("Rlimit", 1), ("Xina", 4), ("Xcmp", 4)),
            _net("0", ("Xmos", 3), ("GND1", 1), ("Xsc", 1), ("Vgnd", 2)),
            _net("optical_power", ("Ch1", 2), ("Xsc", 3)),
            _net("sc_cathode", ("Xsc", 2), ("Rsense", 1), ("Xina", 2)),
            _net("sense_lo", ("Rsense", 2), ("Vgnd", 1), ("Xina", 1)),
            _net("ina_out", ("Xina", 3), ("Xcmp", 1), ("Ravg", 1)),
            _net("thr", ("Xcmp", 2), ("Ravg", 2), ("Cavg", 1)),
            _net("vref", ("Cavg", 2), ("Xina", 6)),
            _net("vee", ("Xina", 5), ("Xcmp", 5)),
            _net("dout", ("Xcmp", 3), ("OUT1", 1)),
            _net("chtx", ("Ch1", 1)),
        ],
    }


def analog_link_graph() -> dict:
    """Fast-photodiode RX (BPW34 -> sense R) for the analog (OFDM/BFSK) path —
    the solar cell's 798 nF cap is far too slow for a wideband front-end."""
    return {
        "title": "analog link",
        "components": [
            _comp("Vdrv", "DRIVE", ["out"]),
            _comp("Xled", "LXM5_PD01", ["anode", "cathode"]),
            _comp("Xmos", "BSD235N", ["drain", "gate", "source"]),
            _comp("Rlimit", "R", ["1", "2"], "150"),
            _comp("Ch1", "CHANNEL", ["tx", "rx"]),
            _comp("Xpd", "BPW34", ["anode", "cathode", "photo_in"]),
            _comp("Rsense", "R", ["1", "2"], "22k"),
            _comp("OUT1", "OUT", ["out"]),
            _comp("GND1", "GND", ["gnd"]),
        ],
        "nets": [
            _net("gate", ("Vdrv", 1), ("Xmos", 2)),
            _net("led_a", ("Rlimit", 2), ("Xled", 1)),
            _net("led_k", ("Xled", 2), ("Xmos", 1)),
            _net("vcc", ("Rlimit", 1)),
            _net("0", ("Xmos", 3), ("Xpd", 1), ("Rsense", 2), ("GND1", 1)),
            _net("optical_power", ("Ch1", 2), ("Xpd", 3)),
            _net("dout", ("Xpd", 2), ("Rsense", 1), ("OUT1", 1)),
            _net("chtx", ("Ch1", 1)),
        ],
    }


def _cfg(distance_m: float, n_bits: int = 160, modulation: str = "OOK") -> SystemConfig:
    d = SystemConfig.from_preset("kadirvelu2021").to_dict()
    d["modulation"] = modulation
    d["n_bits"] = n_bits
    d["distance_m"] = distance_m
    return SystemConfig(**d)


def test_non_link_graph_returns_none():
    """A graph with no LED (or no PV) is not a two-pass link."""
    g = ook_link_graph()
    g["components"] = [c for c in g["components"] if c["component_type"] != "LXM5_PD01"]
    assert run_two_pass(g, _cfg(0.325)) is None


@spice_unavailable
def test_reference_link_closes():
    """At the nominal distance the in-circuit comparator recovers cleanly and
    the noisy BER is essentially zero (~35 dB SNR)."""
    res = run_two_pass(ook_link_graph(), _cfg(0.325))
    assert res is not None and res["ber"] is not None
    assert res["ber_incircuit"] == 0.0
    assert res["ber"] < 1e-3


@spice_unavailable
def test_user_probes_report_net_voltages():
    """Instrument probes report DC + trace for solved nets on BOTH passes — the
    TX side (gate drive) and the RX side (ina_out) — while unknown nets are
    simply absent. Each net carries its own time axis."""
    res = run_two_pass(ook_link_graph(), _cfg(0.325),
                       user_probe_nets=["gate", "ina_out", "sc_cathode", "nope"])
    nets = res["probes"]["nets"]
    assert "gate" in nets       # TX-side net resolved (the extension)
    assert "ina_out" in nets    # RX-side net resolved
    assert "nope" not in nets
    nd = nets["ina_out"]
    assert nd["min"] <= nd["dc"] <= nd["max"]
    assert len(nd["trace"]) == len(nd["t"]) > 0


@spice_unavailable
def test_manchester_link_closes():
    """Manchester (Milestone 2): the comparator slices the 2-level line code
    in-circuit; the Python decode (MCU) turns half-bit symbols back into bits.
    The link closes at the reference distance and noise still bites when far."""
    near = run_two_pass(ook_link_graph(), _cfg(0.325, modulation="OOK_Manchester"))
    far = run_two_pass(ook_link_graph(), _cfg(1.4, modulation="OOK_Manchester"))
    assert near is not None and far is not None
    assert near["modulation"] == "OOK_Manchester"
    # Same circuit, no rewiring — only the line code + decode changed.
    assert near["ber_incircuit"] == 0.0
    assert near["ber"] < 1e-3
    assert far["ber"] > near["ber"]


@spice_unavailable
def test_bfsk_analog_link_closes():
    """BFSK (Milestone 2, analog path): the transmitter is a linear driver, so
    there's no in-circuit slicing — the SPICE RX analog front-end shapes the
    waveform and the Python Goertzel DSP (the MCU) demodulates. xu2024's
    tone/rate params make the link close through the drawn amp output."""
    from cosim.schematic_cosim import run_analog_link
    d = SystemConfig.from_preset("xu2024").to_dict()
    d["modulation"] = "BFSK"
    d["n_bits"] = 160
    d["dcdc_enable"] = False
    res = run_analog_link(ook_link_graph(), SystemConfig(**d))
    assert res is not None and res["ber"] is not None
    assert res["modulation"] == "BFSK"
    assert res["ber"] < 1e-2


@spice_unavailable
def test_ofdm_analog_link_closes():
    """OFDM (Milestone 2): closes through a fast-photodiode passive front-end via
    a circuit-measured per-subcarrier equalizer (LS channel estimate from the
    known TX symbols, which resolves the inverting front-end's phase). Needs the
    fast photodiode (not the solar cell) and a rate the front-end passes."""
    from cosim.schematic_cosim import run_analog_link
    d = SystemConfig.from_preset("kadirvelu2021").to_dict()
    d.update(modulation="OFDM", dcdc_enable=False, modulation_depth=0.2,
             data_rate_bps=200_000.0, distance_m=0.325, n_bits=1280,
             ofdm_nfft=256, ofdm_qam_order=16, ofdm_n_subcarriers=80, ofdm_cp_len=32)
    res = run_analog_link(analog_link_graph(), SystemConfig(**d))
    assert res is not None and res["ber"] is not None
    assert res["modulation"] == "OFDM"
    # Pilot-based (2-symbol training preamble) equalization, BER excludes the
    # preamble; well within FEC-correctable range.
    assert res["ber"] < 2e-2


@spice_unavailable
def test_mcu_adc_designates_probe():
    """The MCU (ESP32) node makes the digital baseband explicit: its ADC pin
    marks where the DSP samples, and the analog link still closes with it placed."""
    from cosim.schematic_cosim import run_analog_link, _find_mcu_adc, _net_lookup
    g = analog_link_graph()
    g["components"].append(_comp("Mcu1", "MCU", ["adc", "gpio"]))
    for net in g["nets"]:
        if net["name"] == "dout":               # ADC taps the sampled output net
            net["pins"].append({"component_ref": "Mcu1", "pin_number": 1})
    g["nets"].append(_net("mcu_gpio", ("Mcu1", 2)))   # GPIO left dangling

    assert _find_mcu_adc(g, _net_lookup(g)) == "dout"
    d = SystemConfig.from_preset("xu2024").to_dict()   # BFSK: fast, tests the ADC tap
    d.update(modulation="BFSK", dcdc_enable=False, n_bits=160)
    res = run_analog_link(g, SystemConfig(**d))
    assert res is not None and res["ber"] is not None and res["ber"] < 1e-2


@spice_unavailable
def test_noise_bites_with_distance():
    """The honest BER is not trivially zero: as the link spreads out, the
    6-source noise drives BER up, while the noiseless in-circuit recovery stays
    clean (errors come from noise, not a circuit failure)."""
    near = run_two_pass(ook_link_graph(), _cfg(0.325))
    far = run_two_pass(ook_link_graph(), _cfg(1.4))
    assert near is not None and far is not None

    # In-circuit (noiseless) recovery is clean at both ranges.
    assert near["ber_incircuit"] == 0.0
    assert far["ber_incircuit"] == 0.0

    # Noise drives the post BER up substantially as the link degrades.
    assert near["ber"] < 1e-3
    assert far["ber"] > 0.03
    assert far["ber"] > near["ber"]

    # And P_rx really did fall with distance (the physical cause).
    assert far["diagnostics"]["p_rx_uW"][1] < near["diagnostics"]["p_rx_uW"][1]

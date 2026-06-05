"""Two-pass co-simulation of a single-canvas TX -> channel -> RX link.

Orchestrates the architecture locked for milestone 1 (fully-in-circuit OOK):

  1. Python generates the bits and the OOK gate-drive waveform.
  2. TX analog (drive -> MOSFET -> LED) solved in libngspice; probe I(LED).
  3. Comms layer (Python): I(LED) -> optical -> channel -> received P_rx(t).
  4. RX analog (PV -> sense -> amp -> comparator) solved in libngspice; the
     comparator demodulates in-circuit, so V(dout) is the recovered bitstream.
  5. Post-processing (Python): report the in-circuit recovery, and an honest
     noisy BER by adding the validated 6-source noise to the comparator's
     analog input — referred there via the transimpedance *measured from the
     SPICE run itself* (comparator-input swing / photocurrent swing).

Returns None when the graph isn't a two-pass link, so the caller can fall back
to the RX-only path.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from components import get_component
from components.base import AmplifierBase
from components.comparators import ComparatorBase
from cosim import pyspice_graph_runner as rx_runner
from cosim import tx_spice_runner as tx_runner
from cosim.graph_netlist import graph_dict_to_instances
from cosim.graph_partition import Partition, partition_tx_rx, subgraph
from cosim.noise import NoiseModel
from cosim.optical_bridge import led_current_to_p_rx

logger = logging.getLogger(__name__)

_SKIP_BITS = 2  # ignore startup transient at the deck's leading edge


def _collect_defs(components: list[dict]) -> str:
    """Concatenated .SUBCKT bodies for each unique placed library part that has
    one (primitives R/V and label markers have none)."""
    defs: dict[str, str] = {}
    for comp in components:
        ctype = comp.get("component_type", "")
        if ctype in defs:
            continue
        try:
            part = get_component(ctype)
        except Exception:  # noqa: BLE001
            continue
        if hasattr(part, "spice_subcircuit"):
            defs[ctype] = part.spice_subcircuit()
    return "\n".join(defs.values())


def _net_lookup(graph: dict) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for net in graph.get("nets", []):
        for p in net.get("pins", []):
            out[(p["component_ref"], int(p["pin_number"]))] = net["name"]
    return out


def _find_comparator(part: Partition, comps: dict, net_of: dict):
    """Return (cmp_in_net, cmp_th_net) for the RX comparator, or (None, None)."""
    for ref in part.rx_refs:
        try:
            if isinstance(get_component(comps[ref]["component_type"]), ComparatorBase):
                return net_of.get((ref, 1)), net_of.get((ref, 2))
        except Exception:  # noqa: BLE001
            continue
    return None, None


def _sample_centers(t: np.ndarray, sig: np.ndarray, n: int,
                    period: float) -> np.ndarray:
    centers = (np.arange(n) + 0.5) * period
    idx = np.clip(np.searchsorted(t, centers), 0, len(sig) - 1)
    return sig[idx]


def _pack_probes(user_nets, extras: dict, t, n: int = 400) -> Optional[dict]:
    """Bundle user-placed instrument probes (multimeter / scope) for transport.

    ``user_nets`` are the (lowercased) net names each probe touches. For every
    net found in the solved circuit's ``extras`` we return its DC level (mean),
    min/max, and a downsampled trace; ``time`` is the shared (downsampled) axis.
    Returns None when there are no probes.
    """
    nets = [str(x).lower() for x in (user_nets or [])]
    if not nets:
        return None
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return {"time": [], "nets": {}}
    idx = np.linspace(0, t.size - 1, min(n, t.size)).astype(int)
    out = {"time": t[idx].round(6).tolist(), "nets": {}}
    for net in nets:
        w = extras.get(net)
        if w is None:
            continue
        w = np.asarray(w, dtype=float)
        out["nets"][net] = {
            "dc": float(np.mean(w)),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
            "trace": w[idx].round(5).tolist(),
        }
    return out


def _is_manchester(cfg) -> bool:
    return cfg.modulation.upper().replace("-", "_") in ("OOK_MANCHESTER", "MANCHESTER")


def _encode_line(bits: np.ndarray, cfg):
    """bits -> (2-level line symbols, symbols-per-bit). The line code is what the
    LED actually switches; the comparator slices it back in-circuit and the
    Python decode (the MCU model) turns symbols back into bits."""
    if _is_manchester(cfg):
        from cosim.modulation import manchester_encode
        return manchester_encode(bits), 2
    return np.asarray(bits, dtype=int), 1


def _decode_line(symbols: np.ndarray, cfg) -> np.ndarray:
    """Recovered line symbols (sliced 0/1) -> bits (the digital-baseband decode)."""
    if _is_manchester(cfg):
        from cosim.modulation import manchester_decode
        return manchester_decode(symbols)
    return np.asarray(symbols, dtype=int)


def run_two_pass(graph: dict, cfg, user_probe_nets=None) -> Optional[dict]:
    part = partition_tx_rx(graph)
    if part is None:
        return None
    if not tx_runner.available() or not rx_runner.available():
        return None

    comps = {c["ref"]: c for c in graph["components"]}
    net_of = _net_lookup(graph)
    led = get_component(comps[part.led_ref]["component_type"])

    # --- bits + OOK gate drive (Python owns modulation) ---
    n_bits = int(cfg.n_bits)
    bit_period = 1.0 / cfg.data_rate_bps
    t_stop = n_bits * bit_period
    t_step = bit_period / 100.0
    seed = cfg.random_seed if cfg.random_seed is not None else 0
    bits = np.random.default_rng(seed).integers(0, 2, n_bits)
    # Line code: OOK is the bits themselves; Manchester is 2 half-bit symbols
    # per bit at 2x the rate. Either way it's a 2-level NRZ drive on the LED.
    line_syms, spb = _encode_line(bits, cfg)
    n_syms = len(line_syms)
    symbol_period = bit_period / spb
    drive_t, drive_v = tx_runner.ook_gate_drive(line_syms, symbol_period, v_on=cfg.vcc_volts)

    # --- pass 1: TX analog -> I(LED) ---
    tx_sub = subgraph(graph, part.tx_refs)
    tx_res = tx_runner.run_tx_graph(
        _collect_defs(tx_sub["components"]),
        graph_dict_to_instances(tx_sub),
        drive_net=part.drive_net.lower(), drive_t=drive_t, drive_v=drive_v,
        led_ref=part.led_ref, vcc_volts=cfg.vcc_volts,
        t_stop_s=t_stop, t_step_s=t_step,
    )
    if tx_res is None:
        return {"ber": None, "message": "TX pass failed (no I(LED))."}
    t_tx, i_led = tx_res

    # --- comms layer: I(LED) -> optical -> channel -> P_rx ---
    p_rx = led_current_to_p_rx(led, i_led, cfg)

    # --- pass 2: RX analog, comparator demodulates in-circuit ---
    cmp_in, cmp_th = _find_comparator(part, comps, net_of)
    probe = [n.lower() for n in (cmp_in, cmp_th) if n]
    probe += [str(n).lower() for n in (user_probe_nets or []) if n]
    rx_sub = subgraph(graph, part.rx_refs)
    rx_res = rx_runner.run_graph(
        _collect_defs(rx_sub["components"]),
        graph_dict_to_instances(rx_sub),
        optical_t=t_tx, optical_v=p_rx,
        vcc_volts=cfg.vcc_volts, t_stop_s=t_stop, t_step_s=t_step,
        probe_nets=probe,
    )
    if rx_res is None:
        return {"ber": None, "message": "RX pass produced no V(dout)."}
    v_dout, t_rx, extras = rx_res

    # --- post: in-circuit recovery (noiseless) ---
    # The comparator slices each line symbol in-circuit; we sample at symbol
    # centres, then the Python decode (the MCU) turns symbols back into bits.
    # Skip a short leading acquisition window (startup transient + any
    # average-value threshold settling) before scoring.
    n_skip = max(_SKIP_BITS, n_bits // 8)
    score = slice(n_skip, n_bits - _SKIP_BITS)
    tx_scored = bits[score]
    n_scored = int(len(tx_scored))
    thr_half = cfg.vcc_volts / 2.0

    sym_dout = (_sample_centers(t_rx, v_dout, n_syms, symbol_period) > thr_half).astype(int)
    rec_incircuit = _decode_line(sym_dout, cfg)
    ber_incircuit = float(np.mean(rec_incircuit[score] != tx_scored))

    # --- post: honest noisy BER at the comparator input ---
    ber = ber_incircuit
    sigma_v = 0.0
    z_trans = None
    cmp_in_lc = (cmp_in or "").lower()
    if cmp_in_lc in extras:
        v_in = extras[cmp_in_lc]
        cmp_th_lc = (cmp_th or "").lower()
        v_th = extras[cmp_th_lc] if cmp_th_lc in extras else np.full_like(v_in, thr_half)
        # photocurrent on the RX time base, and the transimpedance the real
        # circuit actually realized (measured, not assumed).
        i_ph = cfg.sc_responsivity * np.interp(t_rx, t_tx, p_rx)
        std_iph = float(np.std(i_ph))
        z_trans = float(np.std(v_in) / std_iph) if std_iph > 0 else 0.0
        # Noise bandwidth follows the line/symbol rate (Manchester runs at 2x,
        # paying its well-known ~3 dB bandwidth penalty honestly).
        sigma_a = NoiseModel.from_config(cfg).total_noise_std(i_ph, cfg.data_rate_bps * spb)
        sigma_v = float(sigma_a * z_trans)
        v_in_s = _sample_centers(t_rx, v_in, n_syms, symbol_period)
        v_th_s = _sample_centers(t_rx, v_th, n_syms, symbol_period)
        rng = np.random.default_rng(seed + 1)
        sym_noisy = ((v_in_s + rng.normal(0.0, sigma_v, size=v_in_s.shape)) > v_th_s).astype(int)
        rec_noisy = _decode_line(sym_noisy, cfg)
        ber = float(np.mean(rec_noisy[score] != tx_scored))

    return {
        "ber": ber,
        "ber_incircuit": ber_incircuit,
        "n_bits": n_scored,
        "modulation": cfg.modulation,
        "message": f"two-pass {cfg.modulation} co-simulation complete",
        "diagnostics": {
            "i_led_mA": [float(i_led.min() * 1e3), float(i_led.max() * 1e3)],
            "p_rx_uW": [float(p_rx.min() * 1e6), float(p_rx.max() * 1e6)],
            "transimpedance_V_per_A": z_trans,
            "noise_sigma_mV": sigma_v * 1e3,
        },
        "probes": _pack_probes(user_probe_nets, extras, t_rx),
    }


# =============================================================================
# Analog (continuous-waveform) modulations: OFDM, BFSK
# =============================================================================
#
# These are NOT 2-level line codes, so the comparator can't slice them. The
# transmitter is a *linear* driver (DSP -> DAC -> linear LED driver), so the TX
# optical waveform is the validated Python modulator output directly; the SPICE
# RX analog front-end (fast photodiode -> sense R) shapes the received waveform,
# and the Python DSP (the MCU) ADC-samples + demodulates (FFT for OFDM, Goertzel
# for BFSK). This mirrors real hardware: the analog front-end is the circuit,
# the (de)modulation is the DSP.
#
# OFDM uses pilot-based per-subcarrier equalisation measured from the *drawn*
# circuit (a known training preamble; see _dsp_demod). Known gaps: (a) the RX is
# a passive transimpedance (photodiode + load R) — an active TIA gain stage
# (e.g. OPA380) does NOT converge in libngspice feedback (the behavioral op-amp
# model's ideal-clamp output fails the feedback solve), so it's deferred until
# the model is reworked; (b) TX LED bandwidth roll-off isn't circuit-modelled
# for the analog path (the linear driver is treated as ideal).

_ANALOG_SCHEMES = {"OFDM", "BFSK"}


def is_analog_modulation(modulation: str) -> bool:
    return modulation.upper().replace("-", "_") in _ANALOG_SCHEMES


# =============================================================================
# PWM-ASK: in-circuit TX (BJT switch -> LED) + SPICE RX gain chain + DSP demod
# =============================================================================
#
# The breadboard PoC. Unlike OOK (comparator slices in-circuit) and OFDM/BFSK
# (Python linear-driver TX), PWM-ASK switches the LED on/off at a carrier within
# each data bit, and the RX is an analog gain chain (PV -> AC-couple -> TL072
# x2) read by the ESP32 ADC, where Python firmware envelope-demodulates. So this
# is a *two-pass* co-sim (SPICE TX + SPICE RX) like OOK, but the demod is the
# DSP (per-bit envelope) like the analog path, and the RX op-amps run on a split
# +/-5 V supply (the ICL7660 makes the -5 V rail; VMID is the vref mid-rail).


def _pwm_ask_gate_drive(bits: np.ndarray, t: np.ndarray, carrier_freq: float,
                        bit_period: float, v_on: float):
    """Carrier-gated 2-level gate drive for the BJT LED switch.

    Within a bit=1 interval the LED switches on/off at ``carrier_freq`` (the
    ASK carrier); a bit=0 interval holds it off. Mirrors _modulate_pwm_ask but
    as the GPIO gate waveform (0 / v_on) the 2N2222 base sees through R1.
    """
    n_bits = len(bits)
    bit_idx = np.clip((t / bit_period).astype(int), 0, n_bits - 1)
    data_envelope = bits[bit_idx].astype(float)
    carrier = ((t * carrier_freq) % 1.0 < 0.5).astype(float)
    return v_on * data_envelope * carrier


def run_pwm_ask_link(graph: dict, cfg, user_probe_nets=None) -> Optional[dict]:
    """Two-pass PWM-ASK co-sim: SPICE TX (BJT->LED) -> Python channel ->
    SPICE RX gain chain (split supply) -> Python envelope demod. Returns None
    when the graph isn't a TX->RX link."""
    part = partition_tx_rx(graph)
    if part is None or not tx_runner.available() or not rx_runner.available():
        return None

    comps = {c["ref"]: c for c in graph["components"]}
    net_of = _net_lookup(graph)
    led = get_component(comps[part.led_ref]["component_type"])

    # --- bits + PWM-ASK gate drive (Python owns modulation) ---
    n_bits = int(cfg.n_bits)
    bit_period = 1.0 / cfg.data_rate_bps
    carrier_freq = float(getattr(cfg, "carrier_freq_hz", 10000.0))
    t_stop = n_bits * bit_period
    # Resolve the carrier: >= ~20 steps per carrier period.
    t_step = min(bit_period / 50.0, 1.0 / (carrier_freq * 20.0))
    seed = cfg.random_seed if cfg.random_seed is not None else 0
    bits = np.random.default_rng(seed).integers(0, 2, n_bits)
    drive_t = np.arange(0.0, t_stop, t_step)
    drive_v = _pwm_ask_gate_drive(bits, drive_t, carrier_freq, bit_period,
                                  v_on=3.3)  # ESP32 GPIO high

    # --- pass 1: TX analog (GPIO -> R1 -> 2N2222 -> LED) -> I(LED) ---
    # TX rail is +5 V (the LED anode rail), not the 3.3 V GPIO level.
    tx_sub = subgraph(graph, part.tx_refs)
    tx_res = tx_runner.run_tx_graph(
        _collect_defs(tx_sub["components"]),
        graph_dict_to_instances(tx_sub),
        drive_net=part.drive_net.lower(), drive_t=drive_t, drive_v=drive_v,
        led_ref=part.led_ref, vcc_volts=5.0,
        t_stop_s=t_stop, t_step_s=t_step,
    )
    if tx_res is None:
        return {"ber": None, "message": "TX pass failed (no I(LED))."}
    t_tx, i_led = tx_res

    # --- comms layer: I(LED) -> optical -> channel -> P_rx ---
    p_rx = led_current_to_p_rx(led, i_led, cfg)

    # --- pass 2: RX gain chain on +/-5 V; probe the ESP32 ADC node ---
    amp_out = _find_mcu_adc(graph, net_of) or _find_amp_out(part, comps, net_of)
    if amp_out is None:
        return {"ber": None, "message": "No ADC/amp node to sample for PWM-ASK demod."}
    rx_sub = subgraph(graph, part.rx_refs)
    rx_res = rx_runner.run_graph(
        _collect_defs(rx_sub["components"]),
        graph_dict_to_instances(rx_sub),
        optical_t=t_tx, optical_v=p_rx,
        vcc_volts=5.0, vee_volts=-5.0, vref_volts=1.65,
        t_stop_s=t_stop, t_step_s=t_step,
        probe_nets=[amp_out.lower()] + [str(n).lower() for n in (user_probe_nets or []) if n],
    )
    if rx_res is None:
        return {"ber": None, "message": "RX pass produced no output."}
    _v_dout, t_rx, extras = rx_res
    v_amp = extras.get(amp_out.lower())
    if v_amp is None:
        return {"ber": None, "message": f"ADC node {amp_out!r} not found in RX result."}

    # --- DSP: resample to a uniform per-bit grid, add noise, envelope-demod ---
    from cosim.modulation import demodulate
    sps = 200  # samples per bit for the envelope integrator
    t_uni = np.linspace(0.0, t_stop, n_bits * sps, endpoint=False)
    v_rx = np.interp(t_uni, t_rx, v_amp)
    i_ph = cfg.sc_responsivity * np.interp(t_uni, t_tx, p_rx)
    std_iph = float(np.std(i_ph))
    z_trans = float(np.std(v_rx) / std_iph) if std_iph > 0 else 0.0
    sigma_v = float(NoiseModel.from_config(cfg).total_noise_std(i_ph, cfg.data_rate_bps) * z_trans)
    rng = np.random.default_rng(seed + 1)
    v_rx_noisy = v_rx + rng.normal(0.0, sigma_v, size=v_rx.shape)

    n_skip = max(_SKIP_BITS, n_bits // 8)
    score = slice(n_skip, n_bits - _SKIP_BITS)
    bits_rx = np.asarray(demodulate("PWM_ASK", v_rx_noisy, t_uni, n_bits, config=cfg))
    bits_rx_clean = np.asarray(demodulate("PWM_ASK", v_rx, t_uni, n_bits, config=cfg))
    ber = float(np.mean(bits_rx[score] != bits[score]))
    ber_clean = float(np.mean(bits_rx_clean[score] != bits[score]))

    return {
        "ber": ber,
        "ber_incircuit": ber_clean,
        "n_bits": int(n_bits - n_skip - _SKIP_BITS),
        "modulation": cfg.modulation,
        "message": f"PWM-ASK two-pass co-simulation complete (SPICE TX+RX, DSP demod)",
        "diagnostics": {
            "i_led_mA": [float(i_led.min() * 1e3), float(i_led.max() * 1e3)],
            "p_rx_uW": [float(p_rx.min() * 1e6), float(p_rx.max() * 1e6)],
            "amp_swing_mV": [float(v_amp.min() * 1e3), float(v_amp.max() * 1e3)],
            "transimpedance_V_per_A": z_trans,
            "noise_sigma_mV": sigma_v * 1e3,
        },
        "probes": _pack_probes(user_probe_nets, extras, t_rx),
    }


def _find_mcu_adc(graph: dict, net_of: dict) -> Optional[str]:
    """Net wired to an MCU node's `adc` pin — the explicit point where the
    digital baseband (the ESP/Arduino) samples the analog signal. Lets the user
    designate the DSP probe instead of relying on the auto-detected amp output."""
    for c in graph.get("components", []):
        if c.get("component_type") != "MCU":
            continue
        for k, name in c.get("pins", {}).items():
            if name == "adc":
                n = net_of.get((c["ref"], int(k)))
                if n:
                    return n
    return None


def _find_amp_out(part: Partition, comps: dict, net_of: dict) -> Optional[str]:
    """Net on the RX amplifier's output pin (the analog signal the DSP samples)."""
    for ref in part.rx_refs:
        try:
            c = get_component(comps[ref]["component_type"])
        except Exception:  # noqa: BLE001
            continue
        if isinstance(c, AmplifierBase):
            for i, p in enumerate(c.spice_ports(), start=1):
                if p.role.value == "signal_out":
                    return net_of.get((ref, i))
    return None


def _dsp_demod(scheme: str, v_rx: np.ndarray, t: np.ndarray, bits_tx: np.ndarray, cfg):
    """ADC + DSP demodulation of the analog RX waveform (the MCU/DSP).

    Returns ``(bits_rx, n_skip)`` — ``n_skip`` leading bits are channel-estimation
    overhead to exclude from BER (0 for BFSK).
    """
    from cosim.modulation import demodulate
    n_bits = len(bits_tx)
    if scheme == "BFSK":
        return demodulate("BFSK", v_rx, t, n_bits, config=cfg), 0

    # OFDM: resample the chain output to the OFDM digital rate, estimate the
    # per-subcarrier channel response H (which also undoes the global sign/phase
    # a passive inverting front-end introduces), then demod. Mirrors the
    # cosim.python_engine Phase-14 resample.
    from scipy.signal import resample as sp_resample
    from cosim.modulation import _bits_to_qam
    nfft, cp = cfg.ofdm_nfft, cfg.ofdm_cp_len
    n_sc = min(cfg.ofdm_n_subcarriers, nfft // 2 - 1)
    bps = max(1, int(np.log2(cfg.ofdm_qam_order)))
    bits_per_sym = n_sc * bps
    n_syms = max(1, n_bits // bits_per_sym)
    sym_len = nfft + cp
    n_ofdm = n_syms * sym_len
    v_ac = np.asarray(v_rx, dtype=float) - float(np.mean(v_rx))
    v_ofdm = sp_resample(v_ac, n_ofdm)
    bits_arr = np.asarray(bits_tx, dtype=int)

    def _y(s: int) -> np.ndarray:
        sym = v_ofdm[s * sym_len:(s + 1) * sym_len][cp:]
        return np.fft.fft(sym)[1:nfft // 2][:n_sc]

    # Pilot-based channel estimation: dedicate the first OFDM symbol as a known
    # training preamble, estimate H from it alone, and demod the data symbols
    # with it (realistic training overhead — the data symbols never see their own
    # transmitted values). Falls back to genie-aided LS over all symbols only
    # when there's a single symbol and no room for a preamble.
    n_skip = 0
    if n_syms >= 2:
        # 1-2 symbol training preamble (averaged), leaving >=1 data symbol.
        n_train = min(2, n_syms - 1)
        h_num = np.zeros(n_sc, dtype=complex)
        h_den = np.zeros(n_sc, dtype=float)
        for s in range(n_train):
            x = _bits_to_qam(bits_arr[s * bits_per_sym:(s + 1) * bits_per_sym],
                             cfg.ofdm_qam_order, n_sc)
            h_num += _y(s) * np.conj(x)
            h_den += np.abs(x) ** 2
        equalizer = h_num / np.maximum(h_den, 1e-30)
        n_skip = n_train * bits_per_sym
    else:
        h_num = np.zeros(n_sc, dtype=complex)
        h_den = np.zeros(n_sc, dtype=float)
        for s in range(n_syms):
            frame = bits_arr[s * bits_per_sym:(s + 1) * bits_per_sym]
            if len(frame) < bits_per_sym:
                break
            x = _bits_to_qam(frame, cfg.ofdm_qam_order, n_sc)
            h_num += _y(s) * np.conj(x)
            h_den += np.abs(x) ** 2
        equalizer = h_num / np.maximum(h_den, 1e-30)

    fs = n_ofdm / (len(t) * (t[1] - t[0]))
    t_ofdm = np.arange(n_ofdm) / fs
    bits_rx = demodulate("OFDM", v_ofdm, t_ofdm, n_bits, config=cfg,
                         bits_tx=bits_tx, equalizer=equalizer)
    return bits_rx, n_skip


def run_analog_link(graph: dict, cfg) -> Optional[dict]:
    """Two-stage co-sim for OFDM / BFSK: Python linear-driver TX + channel ->
    SPICE RX analog front-end -> Python DSP demod. Returns None if not a link."""
    import shutil
    import tempfile
    from pathlib import Path

    part = partition_tx_rx(graph)
    if part is None or not rx_runner.available():
        return None

    comps = {c["ref"]: c for c in graph["components"]}
    net_of = _net_lookup(graph)
    # Prefer an explicit MCU ADC tap; else the amplifier output.
    amp_out = _find_mcu_adc(graph, net_of) or _find_amp_out(part, comps, net_of)
    if amp_out is None:
        # Passive front-end (no amp): sample the photodiode sense node directly.
        try:
            pv_ports = get_component(comps[part.pv_ref]["component_type"]).spice_ports()
            cath = next((i for i, p in enumerate(pv_ports, 1) if p.name == "cathode"), 2)
            amp_out = net_of.get((part.pv_ref, cath))
        except Exception:  # noqa: BLE001
            amp_out = None
    if amp_out is None:
        return {"ber": None, "message": "No analog node to sample for DSP demod."}

    # --- Python TX + channel (validated modulators handle OFDM/BFSK) ---
    from cosim.pipeline import SimulationPipeline
    session = Path(tempfile.mkdtemp(prefix="optisim_analog_"))
    try:
        for sub in ("pwl", "netlists", "raw", "plots"):
            (session / sub).mkdir(parents=True, exist_ok=True)
        pipe = SimulationPipeline(cfg, session)
        pipe.run_step_tx()
        pipe.run_step_channel()
        if pipe._time is None or pipe._P_rx is None or pipe._tx_bits is None:
            return {"ber": None, "message": "TX/channel produced no waveform."}
        # Optical power can't go negative; the DCO clip + FFT resample leaves a
        # little Gibbs overshoot below zero. Clamp it (physical).
        t, p_rx, bits = pipe._time, np.clip(pipe._P_rx, 0.0, None), pipe._tx_bits

        # --- SPICE RX analog front-end; probe the amplifier output ---
        rx_sub = subgraph(graph, part.rx_refs)
        rx_res = rx_runner.run_graph(
            _collect_defs(rx_sub["components"]),
            graph_dict_to_instances(rx_sub),
            optical_t=t, optical_v=p_rx,
            vcc_volts=cfg.vcc_volts, t_stop_s=float(t[-1]), t_step_s=float(t[1] - t[0]),
            probe_nets=[amp_out.lower()],
        )
    finally:
        shutil.rmtree(session, ignore_errors=True)

    if rx_res is None:
        return {"ber": None, "message": "RX pass produced no output."}
    _v_dout, t_rx, extras = rx_res
    v_amp = extras.get(amp_out.lower())
    if v_amp is None:
        return {"ber": None, "message": f"amplifier output net {amp_out!r} not found in RX result."}

    # --- align to physics time base, apply 6-source noise, DSP-demod ---
    v_rx = np.interp(t, t_rx, v_amp)
    i_ph = cfg.sc_responsivity * p_rx
    std_iph = float(np.std(i_ph))
    z_trans = float(np.std(v_rx) / std_iph) if std_iph > 0 else 0.0
    sigma_a = NoiseModel.from_config(cfg).total_noise_std(i_ph, cfg.data_rate_bps)
    sigma_v = float(sigma_a * z_trans)
    seed = cfg.random_seed if cfg.random_seed is not None else 0
    rng = np.random.default_rng(seed + 1)
    v_rx_noisy = v_rx + rng.normal(0.0, sigma_v, size=v_rx.shape)

    scheme = cfg.modulation.upper().replace("-", "_")
    bits_rx, n_skip = _dsp_demod(scheme, v_rx_noisy, t, bits, cfg)
    bits_rx_clean, _ = _dsp_demod(scheme, v_rx, t, bits, cfg)
    n = min(len(bits_rx), len(bits))
    lo = min(n_skip, n)  # exclude the channel-estimation preamble from BER
    bits_rx = np.asarray(bits_rx)
    bits_rx_clean = np.asarray(bits_rx_clean)
    ber = float(np.mean(bits_rx[lo:n] != bits[lo:n]))
    ber_clean = float(np.mean(bits_rx_clean[lo:n] != bits[lo:n]))

    return {
        "ber": ber,
        "ber_incircuit": ber_clean,  # noiseless DSP recovery (circuit-only)
        "n_bits": int(n),
        "modulation": cfg.modulation,
        "message": f"analog {cfg.modulation} co-simulation complete (DSP demod)",
        "diagnostics": {
            "p_rx_uW": [float(p_rx.min() * 1e6), float(p_rx.max() * 1e6)],
            "transimpedance_V_per_A": z_trans,
            "noise_sigma_mV": sigma_v * 1e3,
        },
    }

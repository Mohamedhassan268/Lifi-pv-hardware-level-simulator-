# cosim/python_engine.py
"""
Python System-Level Simulation Engine.

Provides a pure-Python simulation pipeline for papers that cannot use SPICE
(OFDM, BFSK, PWM-ASK, MIMO architectures). Adapted from the validated
system-level simulator (lifi_pv_simulator).

Supports 5 modulation schemes: OOK, OOK_Manchester, OFDM, BFSK, PWM_ASK.

Channel, noise, and modulation logic are now in dedicated modules:
    - cosim.channel      → OpticalChannel
    - cosim.noise        → NoiseModel
    - cosim.modulation   → modulate(), demodulate(), predict_ber(), etc.

This file retains: PVReceiver (Phase 2 target) and run_python_simulation().

Usage:
    from cosim.python_engine import run_python_simulation
    result = run_python_simulation(config)
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Dict, Optional

# Import from dedicated modules
from cosim.channel import OpticalChannel
from cosim.noise import NoiseModel, K_BOLTZMANN, Q_ELECTRON
from cosim.modulation import (
    modulate, demodulate, calculate_ber,
    generate_ofdm_digital,
    # Re-export BER functions for backward compatibility
    predict_ber_ook, predict_ber_ook_db, predict_ber_bpsk,
    predict_ber_bfsk, predict_ber_mqam,
    # Re-export Manchester codec for backward compatibility
    manchester_encode, manchester_decode,
)
from cosim.probes import ProbeCapture

# Phase 2 models (imported lazily to keep backward compat if scipy missing)
from cosim.pv_model import PVCellModel
from cosim.rx_chain import ReceiverChain
from cosim.dcdc_model import BoostConverter
from cosim.tx_model import LEDTransmitter

# Simulation defaults
SAMPLES_PER_BIT = 500        # Oversampling ratio for time-domain simulation
TIA_GAIN_OHM = 50e3          # Transimpedance gain for noise domain conversion


# =============================================================================
# PV RECEIVER (kept here — Phase 2 will enhance with ODE, V-dependent Cj)
# =============================================================================

class PVReceiver:
    """PV cell receiver with TIA and signal chain processing."""

    def __init__(self, responsivity=0.457, capacitance_nF=798,
                 shunt_resistance_kOhm=138.8, n_cells=1, temperature_K=300,
                 responsivity_ref_T_K=300.0,
                 responsivity_tempco_per_K=4e-4):
        """
        Args:
            responsivity: A/W at responsivity_ref_T_K
            responsivity_ref_T_K: Reference temperature for the given
                responsivity value (typically 300 K / 25°C datasheet spec).
            responsivity_tempco_per_K: Fractional change per Kelvin.
                Typical values: Si ~4e-4/K (0.04%/°C), GaAs ~6e-4/K.
        """
        self.R_ref = responsivity
        self.R_ref_T = responsivity_ref_T_K
        self.R_tempco = responsivity_tempco_per_K
        self.C_j = capacitance_nF * 1e-9
        self.R_sh = shunt_resistance_kOhm * 1e3
        self.n_cells = n_cells
        self.T = temperature_K
        self.V_T = K_BOLTZMANN * temperature_K / Q_ELECTRON

    @property
    def R(self) -> float:
        """Responsivity adjusted to the operating temperature."""
        return self.R_ref * (1.0 + self.R_tempco * (self.T - self.R_ref_T))

    def optical_to_current(self, P_rx):
        """I_ph = R(T) * P_rx."""
        return self.R * P_rx

    def apply_tia(self, I_ph, t, R_tia=50e3, f_3db=3e6):
        """TIA: V = R_tia * I_ph with bandwidth limit."""
        V_ideal = R_tia * I_ph
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        wn = f_3db / (fs / 2)
        if wn >= 1.0:
            return V_ideal
        b, a = sp_signal.butter(1, wn, btype='low')
        return sp_signal.filtfilt(b, a, V_ideal)

    def apply_bandpass(self, signal_in, t, f_low=700, f_high=10000, order=2):
        """Butterworth bandpass filter."""
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        nyq = fs / 2
        low_n = f_low / nyq
        high_n = f_high / nyq
        if low_n >= 1.0 or high_n >= 1.0 or low_n <= 0:
            return signal_in
        b, a = sp_signal.butter(order, [low_n, high_n], btype='band')
        try:
            return sp_signal.filtfilt(b, a, signal_in)
        except ValueError:
            return signal_in

    def apply_notch(self, signal_in, t, f_notch=100, Q=30):
        """IIR notch filter for mains rejection (González 2024)."""
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        if f_notch >= fs / 2:
            return signal_in
        b, a = sp_signal.iirnotch(f_notch, Q, fs)
        return sp_signal.filtfilt(b, a, signal_in)


# =============================================================================
# MAIN SIMULATION RUNNER
# =============================================================================

def run_python_simulation(config, bits_override=None,
                          probes: Optional[ProbeCapture] = None,
                          stage_hook=None) -> Dict:
    """
    Run a full system-level Python simulation using SystemConfig parameters.

    Maps SystemConfig fields to the simulator classes and runs the appropriate
    modulation pipeline.

    Args:
        config: SystemConfig instance
        bits_override: Optional user-supplied TX bit array. When given, n_bits
            is taken from the array length and the PRBS generator is skipped —
            used by the text TX/RX GUI tab to send a known message.
        probes: Optional ProbeCapture. When given, intermediate arrays are
            captured at each block boundary (P_tx, P_rx, I_ph, per-source
            noise, V_sense, V_ina, V_bpf, V_comp, bits_rx, ...). No-op when
            None — pipeline runs as before.
        stage_hook: Optional callable(stage: str). Invoked after TX, Channel,
            and RX blocks complete. Used by the WebSocket runner for
            cooperative pause/resume between blocks.

    Returns:
        Dict with keys: 'ber', 'n_errors', 'n_bits_tested', 'snr_est_dB',
                        'time', 'P_tx', 'P_rx', 'I_ph', 'V_rx', 'bits_tx',
                        'bits_rx', 'engine', 'modulation'
    """
    cfg = config
    mod_scheme = cfg.modulation.upper().replace('-', '_')

    # Compute timing
    bit_period = 1.0 / cfg.data_rate_bps
    samples_per_bit = SAMPLES_PER_BIT
    if bits_override is not None:
        bits_tx = np.asarray(bits_override, dtype=int).ravel()
        n_bits = len(bits_tx)
    else:
        n_bits = cfg.n_bits
        bits_tx = None
    n_samples = n_bits * samples_per_bit
    dt = bit_period / samples_per_bit
    fs = 1.0 / dt
    t = np.arange(n_samples) * dt

    # Generate TX bits if none were provided
    if bits_tx is None:
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)
        bits_tx = np.random.randint(0, 2, n_bits)

    # ========== TX: Modulate ==========
    P_tx = modulate(mod_scheme, bits_tx, t, config=cfg)

    # Apply LED bandwidth limiting if enabled (Phase 2)
    tx_result = None
    if cfg.led_bandwidth_limit_enable:
        led_tx = LEDTransmitter.from_config(cfg)
        # P_tx from modulate() is already in watts; normalize for LED model
        P_tx_max = np.max(P_tx) if np.max(P_tx) > 0 else 1.0
        P_tx_norm = P_tx / P_tx_max
        tx_result = led_tx.process(P_tx_norm, t)
        P_tx = tx_result.P_tx

    # ----- TX probes + pause point -----
    if probes is not None:
        probes.capture("tx.bits", bits_tx)
        probes.capture("tx.P_tx", P_tx)
    if stage_hook is not None:
        stage_hook("TX")

    # ========== Channel: Propagate ==========
    channel = OpticalChannel.from_config(cfg)
    P_rx = channel.propagate(P_tx)

    # ----- Channel probes + pause point -----
    if probes is not None:
        ch_result = channel.compute_gain()
        probes.capture("channel.G_los", ch_result.gain_los)
        probes.capture("channel.G_diffuse", ch_result.gain_diffuse)
        probes.capture("channel.beer_lambert", ch_result.beer_lambert_factor)
        probes.capture("channel.G_total", channel.channel_gain())
        probes.capture("channel.P_rx", P_rx)
    if stage_hook is not None:
        stage_hook("Channel")

    # ========== RX: Photodetection ==========
    rx = PVReceiver(
        responsivity=cfg.sc_responsivity,
        capacitance_nF=cfg.sc_cj_nF,
        shunt_resistance_kOhm=cfg.sc_rsh_kOhm,
        n_cells=cfg.n_cells_series,
        temperature_K=cfg.temperature_K,
        responsivity_ref_T_K=getattr(cfg, 'sc_responsivity_ref_T_K', 300.0),
        responsivity_tempco_per_K=getattr(cfg, 'sc_responsivity_tempco_per_K', 4e-4),
    )

    # Phase 2: PV cell ODE model or simple I = R * P
    pv_result = None
    if cfg.pv_ode_enable:
        pv_model = PVCellModel.from_config(cfg)
        pv_result = pv_model.simulate(t, P_rx, R_load=cfg.r_sense_ohm)
        I_ph = pv_result.I_cell
    else:
        I_ph = rx.optical_to_current(P_rx)

    # Capture clean photocurrent before noise
    if probes is not None:
        probes.capture("rx.I_ph", I_ph)

    # ========== Noise injection ==========
    bandwidth = cfg.data_rate_bps / 2
    noise_model = NoiseModel.from_config(cfg)

    if cfg.noise_enable:
        rng = (np.random.default_rng(cfg.random_seed)
               if cfg.random_seed is not None
               else np.random.default_rng())

        if probes is not None:
            # Per-source decomposition is only computed when probes are
            # active — keeps the hot path identical for non-observability runs.
            per_source = noise_model.generate_per_source_time_domain(
                len(I_ph), I_ph, bandwidth, rng=rng,
            )
            probes.capture("rx.noise.shot", per_source["shot"])
            probes.capture("rx.noise.thermal", per_source["thermal"])
            probes.capture("rx.noise.ambient", per_source["ambient"])
            probes.capture("rx.noise.amplifier", per_source["amplifier"])
            probes.capture("rx.noise.adc", per_source["adc"])
            probes.capture("rx.noise.processing", per_source["processing"])
            noise_awgn = per_source["total"]
        else:
            noise_awgn = noise_model.generate_time_domain(len(I_ph), I_ph, bandwidth)

        # Real-world non-AWGN sources. Disabled by default in the constructor;
        # opt-in via the preset's `enable_mains_flicker` / `enable_amp_flicker`.
        noise_flicker = noise_model.mains_flicker_current(t, rng=rng)
        noise_pink = noise_model.amp_flicker_current(len(I_ph), dt, rng=rng)

        # Sum at the photocurrent domain BEFORE any voltage amplification so
        # the gain stages amplify (signal + noise) together — this is the
        # invariant that lets the demodulator's bit decision actually see noise.
        I_ph_noisy = I_ph + noise_awgn + noise_flicker + noise_pink

        if probes is not None:
            probes.capture("rx.noise.mains_flicker", noise_flicker)
            probes.capture("rx.noise.pink", noise_pink)
    else:
        I_ph_noisy = I_ph

    if probes is not None:
        probes.capture("rx.I_ph_noisy", I_ph_noisy)

    # ========== RX Signal Chain (topology-aware) ==========
    chain_waveforms = None
    topology = getattr(cfg, 'rx_topology', 'ina_bpf_comp')

    if topology == 'direct':
        # Direct: V_rx = I_ph * R_sense (no analog chain)
        V_rx = I_ph_noisy * cfg.r_sense_ohm

    elif topology == 'amp_slicer':
        # Amp + slicer: R_sense → amplifier → optional notch → threshold
        V_sense = I_ph_noisy * cfg.r_sense_ohm

        # Stage gains — each only applied if its field is meaningfully above unity.
        # Previously the engine multiplied INA gain in, then multiplied
        # amp_gain_linear in again, which produced ×G_ina × G_amp instead of
        # the intended ×G_ina for presets that encoded the same ×G in both
        # fields (e.g. lifi_poc_breadboard).
        ina_gain  = 10 ** (cfg.ina_gain_dB / 20) if cfg.ina_gain_dB > 0 else 1.0
        post_gain = cfg.amp_gain_linear           if cfg.amp_gain_linear > 1.0 else 1.0
        V_amp = V_sense * ina_gain * post_gain

        # Apply notch filter if configured (González 2024: mains rejection)
        if cfg.notch_freq_hz is not None:
            V_amp = rx.apply_notch(V_amp, t, f_notch=cfg.notch_freq_hz, Q=cfg.notch_Q)

        V_rx = V_amp

    elif topology == 'ina_bpf_comp' or cfg.rx_chain_enable:
        # Full receiver chain (R_sense → INA → BPF → Comparator)
        rx_chain = ReceiverChain.from_config(cfg)
        chain_waveforms = rx_chain.process(I_ph_noisy, t)
        V_rx = chain_waveforms.V_comp

    else:
        # Fallback: Simple TIA + signal conditioning
        V_tia = rx.apply_tia(I_ph_noisy, t, R_tia=50e3, f_3db=min(bandwidth * 5, fs / 3))

        # Apply notch filter if configured
        if cfg.notch_freq_hz is not None:
            V_tia = rx.apply_notch(V_tia, t, f_notch=cfg.notch_freq_hz, Q=cfg.notch_Q)

        # Apply amplifier gain
        if cfg.amp_gain_linear > 1:
            V_tia = V_tia * cfg.amp_gain_linear

        V_rx = V_tia

    # ========== Demodulation ==========
    if mod_scheme == 'OFDM':
        # OFDM uses digital-domain coherent demodulation.
        # The OFDM TX waveform is dimensionless (normalized); we apply the
        # physical SNR from the link budget by scaling noise to match the
        # signal's electrical SNR at the receiver.
        n_data = cfg.ofdm_nfft // 2 - 1
        n_sc = min(cfg.ofdm_n_subcarriers, n_data)

        ofdm_tx_signal = generate_ofdm_digital(
            bits_tx, cfg.ofdm_qam_order, cfg.ofdm_nfft, cfg.ofdm_cp_len, n_sc)

        if cfg.noise_enable:
            # Physical SNR: (signal current)^2 / (noise current std)^2
            # Signal current ~ R * mean(P_rx) (photocurrent swing)
            I_signal = rx.R * np.mean(P_rx)
            sigma_I = noise_model.total_noise_std(np.mean(I_ph), bandwidth)
            if sigma_I > 0 and I_signal > 0:
                snr_linear = (I_signal / sigma_I) ** 2
            else:
                snr_linear = 1e12
            # Scale noise to match SNR relative to OFDM signal power
            signal_power = np.var(ofdm_tx_signal)
            noise_power = signal_power / max(snr_linear, 1e-12)
            noise_std = np.sqrt(noise_power)
            ofdm_rx_signal = ofdm_tx_signal + np.random.normal(
                0, noise_std, len(ofdm_tx_signal))
        else:
            ofdm_rx_signal = ofdm_tx_signal

        # Channel gain is absorbed (ZF equalization): pass through directly.
        bits_rx = demodulate('OFDM', ofdm_rx_signal, t, n_bits,
                             config=cfg, bits_tx=bits_tx)
    else:
        bits_rx = demodulate(mod_scheme, V_rx, t, n_bits, config=cfg)

    # ========== DC-DC Converter (Phase 2) ==========
    dcdc_result = None
    if pv_result is not None:
        # Compute harvested power through DC-DC converter
        V_cell_avg = np.mean(pv_result.V_cell)
        if V_cell_avg > 0:
            dcdc = BoostConverter.from_config(cfg)
            dcdc_result = dcdc.compute(V_in=V_cell_avg, V_out_target=cfg.vcc_volts)

    # ========== BER calculation ==========
    ber_result = calculate_ber(bits_tx, bits_rx)

    # SNR — link-budget number (closed-form, includes ALL modelled sources).
    # Previously this only summed the six AWGN variances, which made the
    # number near-constant across ambient sweeps because the mains-flicker
    # contribution (which scales as I_amb²) wasn't in the sum.  The
    # link_budget_snr_db helper now adds analytical variances for mains
    # flicker (Σ a_k²·(depth·I_amb)²/2) and 1/f pink (e_n²·f_c·ln(B/f_low)/R²)
    # alongside the AWGN sum, so this number tracks the physical link.
    from cosim.snr import link_budget_snr_db as _link_budget_snr_db
    T_total = n_samples * dt
    snr_db = _link_budget_snr_db(I_ph, bandwidth, noise_model, t_total_s=T_total)

    # ----- RX probes: signal chain + recovered bits + SNR + pause point -----
    if probes is not None:
        if chain_waveforms is not None:
            # V_bpf may be (n_stages, N) when multiple BPF stages are cascaded;
            # store only the final stage so probe shape stays (N,).
            v_bpf_arr = np.asarray(chain_waveforms.V_bpf)
            if v_bpf_arr.ndim > 1:
                v_bpf_arr = v_bpf_arr[-1]
            probes.capture("rx.V_sense", chain_waveforms.V_sense)
            probes.capture("rx.V_ina", chain_waveforms.V_ina)
            probes.capture("rx.V_bpf", v_bpf_arr)
            probes.capture("rx.V_comp", chain_waveforms.V_comp)
        else:
            # Non-INA topologies expose V_rx as the comparator-equivalent output.
            probes.capture("rx.V_sense", I_ph_noisy * cfg.r_sense_ohm)
            probes.capture("rx.V_comp", V_rx)
        probes.capture("rx.bits_rx", bits_rx)
        probes.capture("rx.snr_db", snr_db)
    if stage_hook is not None:
        stage_hook("RX")

    # Eb/N0 against the AWGN floor only (mains flicker + pink are
    # non-flat-spectrum, so they cannot honestly fold into N0).
    from cosim.snr import eb_n0_db_awgn
    eb_n0_awgn = eb_n0_db_awgn(
        I_ph, bandwidth, bit_period, noise_model,
    ) if cfg.noise_enable else float('nan')

    result = {
        'ber': ber_result['ber'],
        'n_errors': ber_result['n_errors'],
        'n_bits_tested': ber_result['n_bits_tested'],
        # Three scientifically distinct SNR-family numbers — see cosim/snr.py.
        'snr_est_dB': snr_db,                        # legacy alias (kept for compat)
        'snr_link_budget_dB': snr_db,
        'snr_measured_dB': float('nan'),             # populated below if enabled
        'eb_n0_dB_awgn_floor': eb_n0_awgn,
        'time': t,
        'P_tx': P_tx,
        'P_rx': P_rx,
        'I_ph': I_ph,
        'V_rx': V_rx,
        'bits_tx': bits_tx,
        'bits_rx': bits_rx,
        'engine': 'python',
        'modulation': cfg.modulation,
        'channel_gain': channel.channel_gain(),
        'P_rx_avg_uW': np.mean(P_rx) * 1e6,
        'I_ph_avg_uA': np.mean(I_ph) * 1e6,
    }

    # Phase 2: Add per-node waveforms when enhanced models are active
    if pv_result is not None:
        result['V_cell'] = pv_result.V_cell
        result['I_cell'] = pv_result.I_cell
        result['I_dark'] = pv_result.I_dark

    if chain_waveforms is not None:
        result['V_sense'] = chain_waveforms.V_sense
        result['V_ina'] = chain_waveforms.V_ina
        result['V_bpf'] = chain_waveforms.V_bpf
        result['V_comp'] = chain_waveforms.V_comp

    if dcdc_result is not None:
        result['dcdc_V_out'] = dcdc_result.V_out
        result['dcdc_efficiency'] = dcdc_result.efficiency
        result['dcdc_mode'] = dcdc_result.mode
        result['dcdc_P_out_uW'] = dcdc_result.P_out * 1e6
        result['dcdc_P_loss_uW'] = dcdc_result.P_loss_total * 1e6

    if tx_result is not None:
        result['P_optical'] = tx_result.P_optical

    # ---- Measured SNR (two-run technique) ----
    # Opt-in via cfg.measure_snr_enable. Doubles per-run cost because it
    # spawns one noise-disabled reference run; the result is the SNR a lab
    # instrument would actually measure at the demodulator input. Guarded
    # against recursion by checking cfg.noise_enable.
    if getattr(cfg, 'measure_snr_enable', False) and cfg.noise_enable:
        from cosim.snr import measure_snr_db
        try:
            snr_nodes = measure_snr_db(
                cfg, nodes=("rx.I_ph_noisy", "rx.V_rx"),
                bits_override=bits_tx,
            )
            result['snr_measured_dB'] = snr_nodes.get("rx.V_rx", float('nan'))
            result['snr_measured_at_Iph_dB'] = snr_nodes.get("rx.I_ph_noisy", float('nan'))
        except Exception:
            # Never let SNR measurement crash the main pipeline.
            pass

    return result

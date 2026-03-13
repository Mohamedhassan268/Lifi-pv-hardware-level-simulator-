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
from typing import Dict

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
                 shunt_resistance_kOhm=138.8, n_cells=1, temperature_K=300):
        self.R = responsivity
        self.C_j = capacitance_nF * 1e-9
        self.R_sh = shunt_resistance_kOhm * 1e3
        self.n_cells = n_cells
        self.T = temperature_K
        self.V_T = K_BOLTZMANN * temperature_K / Q_ELECTRON

    def optical_to_current(self, P_rx):
        """I_ph = R * P_rx."""
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

def run_python_simulation(config) -> Dict:
    """
    Run a full system-level Python simulation using SystemConfig parameters.

    Maps SystemConfig fields to the simulator classes and runs the appropriate
    modulation pipeline.

    Args:
        config: SystemConfig instance

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
    n_bits = cfg.n_bits
    n_samples = n_bits * samples_per_bit
    dt = bit_period / samples_per_bit
    fs = 1.0 / dt
    t = np.arange(n_samples) * dt

    # Generate TX bits
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

    # ========== Channel: Propagate ==========
    channel = OpticalChannel.from_config(cfg)
    P_rx = channel.propagate(P_tx)

    # ========== RX: Photodetection ==========
    rx = PVReceiver(
        responsivity=cfg.sc_responsivity,
        capacitance_nF=cfg.sc_cj_nF,
        shunt_resistance_kOhm=cfg.sc_rsh_kOhm,
        n_cells=cfg.n_cells_series,
        temperature_K=cfg.temperature_K,
    )

    # Phase 2: PV cell ODE model or simple I = R * P
    pv_result = None
    if cfg.pv_ode_enable:
        pv_model = PVCellModel.from_config(cfg)
        pv_result = pv_model.simulate(t, P_rx, R_load=cfg.r_sense_ohm)
        I_ph = pv_result.I_cell
    else:
        I_ph = rx.optical_to_current(P_rx)

    # ========== Noise injection ==========
    bandwidth = cfg.data_rate_bps / 2
    noise_model = NoiseModel.from_config(cfg)

    if cfg.noise_enable:
        noise = noise_model.generate_time_domain(len(I_ph), I_ph, bandwidth)
        I_ph_noisy = I_ph + noise
    else:
        I_ph_noisy = I_ph

    # ========== RX Signal Chain (topology-aware) ==========
    chain_waveforms = None
    topology = getattr(cfg, 'rx_topology', 'ina_bpf_comp')

    if topology == 'direct':
        # Direct: V_rx = I_ph * R_sense (no analog chain)
        V_rx = I_ph_noisy * cfg.r_sense_ohm

    elif topology == 'amp_slicer':
        # Amp + slicer: R_sense → amplifier → optional notch → threshold
        V_sense = I_ph_noisy * cfg.r_sense_ohm

        # Amplifier gain (from INA or standalone amp)
        gain = 10 ** (cfg.ina_gain_dB / 20) if cfg.ina_gain_dB > 0 else cfg.amp_gain_linear
        V_amp = V_sense * gain

        # Apply notch filter if configured (González 2024: mains rejection)
        if cfg.notch_freq_hz is not None:
            V_amp = rx.apply_notch(V_amp, t, f_notch=cfg.notch_freq_hz, Q=cfg.notch_Q)

        # Apply additional voltage amplifier gain if both INA and amp are present
        if cfg.amp_gain_linear > 1 and cfg.ina_gain_dB > 0:
            V_amp = V_amp * cfg.amp_gain_linear

        V_rx = V_amp

    elif cfg.rx_chain_enable:
        # Phase 2: Full receiver chain (R_sense → INA → BPF → Comparator)
        rx_chain = ReceiverChain.from_config(cfg)
        chain_waveforms = rx_chain.process(I_ph_noisy, t)
        V_rx = chain_waveforms.V_comp

    else:
        # Default (ina_bpf_comp without Phase 2): Simple TIA + signal conditioning
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
        # OFDM uses digital-domain coherent demodulation
        n_data = cfg.ofdm_nfft // 2 - 1
        n_sc = min(cfg.ofdm_n_subcarriers, n_data)

        ofdm_tx_signal = generate_ofdm_digital(
            bits_tx, cfg.ofdm_qam_order, cfg.ofdm_nfft, cfg.ofdm_cp_len, n_sc)

        G = channel.channel_gain() * rx.R
        ofdm_rx_signal = ofdm_tx_signal * G
        if cfg.noise_enable:
            sigma = noise_model.total_noise_std(np.mean(I_ph), bandwidth)
            ofdm_rx_signal += np.random.normal(0, sigma * TIA_GAIN_OHM, len(ofdm_rx_signal))

        # Zero-forcing equalization
        if G > 0:
            ofdm_eq = ofdm_rx_signal / G
        else:
            ofdm_eq = ofdm_rx_signal

        bits_rx = demodulate('OFDM', ofdm_eq, t, n_bits, config=cfg, bits_tx=bits_tx)
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

    # SNR estimate
    signal_power = np.var(I_ph) if np.var(I_ph) > 0 else 1e-30
    noise_power = noise_model.total_noise_std(I_ph, bandwidth)**2
    snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-30))

    result = {
        'ber': ber_result['ber'],
        'n_errors': ber_result['n_errors'],
        'n_bits_tested': ber_result['n_bits_tested'],
        'snr_est_dB': snr_db,
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

    return result

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

def _adc_sample(v, t, fs_adc):
    """Model the MCU's finite-rate ADC: sample ``v`` at ``fs_adc`` then hold the
    result back onto the physics grid ``t``. Below ~2x the signal bandwidth this
    aliases (faithful degradation); when fs_adc is faster than the sim grid it is
    a no-op. ``fs_adc <= 0`` disables it (ideal ADC)."""
    span = float(t[-1] - t[0])
    if fs_adc <= 0 or span <= 0:
        return v
    n_adc = int(round(span * fs_adc))
    if n_adc < 2 or n_adc >= len(t):  # ADC at/above the sim rate -> nothing lost
        return v
    t_adc = np.linspace(t[0], t[-1], n_adc)
    return np.interp(t, t_adc, np.interp(t_adc, t, v))


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

    # ========== FEC: encode (optional) ==========
    # When fec_enable is set, cfg.n_bits is the *message* bit budget; the
    # encoded bit count expands by n/k. We rebuild timing arrays around the
    # coded length so the modulator still sees a contiguous bit stream.
    from cosim.fec import build_fec_codec
    fec = build_fec_codec(cfg)
    bits_tx_msg = bits_tx                       # preserved for post-decode BER
    if fec is not None:
        bits_tx = fec.encode(bits_tx_msg)       # coded bits go on the wire
        n_bits = len(bits_tx)
        n_samples = n_bits * samples_per_bit
        t = np.arange(n_samples) * dt

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
    pv_harvest = None
    if cfg.pv_ode_enable:
        pv_model = PVCellModel.from_config(cfg)
        pv_result = pv_model.simulate(t, P_rx, R_load=cfg.r_sense_ohm)
        I_ph = pv_result.I_cell
        # Ambient light is extra DC optical power on the same cell: it shifts the
        # operating point and adds harvested power. Model its harvest effect on a
        # second (cheap) solve over the total incident light, leaving the data +
        # noise path on the LED-only solve — ambient's *data* impact already enters
        # as shot noise (NoiseModel source 3), so this avoids double-counting it.
        # Conversion matches noise.py: 1 lux ≈ 1.46 µW/cm² over the cell area.
        amb_lux = getattr(cfg, 'ambient_illuminance_lux', 0.0)
        if amb_lux > 0:
            P_ambient = amb_lux * 1.46e-6 * cfg.sc_area_cm2  # W (DC)
            pv_harvest = pv_model.simulate(t, P_rx + P_ambient, R_load=cfg.r_sense_ohm)
        else:
            pv_harvest = pv_result
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

    # ========== ADC sampling (MCU) ==========
    # If the controller's ADC rate is set, band-limit V_rx to what it can
    # actually capture before the DSP demodulates. 0 = ideal ADC (no-op).
    fs_adc = float(getattr(cfg, 'mcu_sample_rate_hz', 0.0) or 0.0)
    if fs_adc > 0:
        V_rx = _adc_sample(V_rx, t, fs_adc)

    # ========== Demodulation ==========
    if mod_scheme == 'OFDM':
        # Phase 14: OFDM consumes the actual chain output V_rx (DC-centred),
        # rather than bypassing with a matched-SNR AWGN on a digital twin.
        # All physical noise (shot/thermal/ambient/amp) is already in I_ph
        # and flowed naturally into V_rx; LED bandwidth and PV-cap RC pole
        # colour each subcarrier per the chain transfer function H(f), which
        # the equalizer divides out per-bin.
        from cosim.ofdm_equalizer import (
            chain_response_at,
            subcarrier_frequencies,
        )
        from scipy.signal import resample as sp_resample

        n_data = cfg.ofdm_nfft // 2 - 1
        n_sc = min(cfg.ofdm_n_subcarriers, n_data)

        # Centre V_rx at zero (the chain output has a large DC component
        # from the bias photocurrent that must not enter the FFT bins).
        V_rx_ac = np.asarray(V_rx, dtype=float) - float(np.mean(V_rx))

        # ADC: resample from the physics rate (data_rate × SAMPLES_PER_BIT)
        # down to the OFDM digital rate.  The TX modulator (_modulate_ofdm)
        # uses n_sc carriers (not n_data) to compute n_symbols, so we must
        # match that here to get the same sample count after resampling.
        bps_ofdm = int(np.log2(cfg.ofdm_qam_order))
        n_ofdm_tx_syms = max(1, len(bits_tx) // (n_sc * bps_ofdm))
        sym_len_ofdm = cfg.ofdm_nfft + cfg.ofdm_cp_len
        n_ofdm_samples = n_ofdm_tx_syms * sym_len_ofdm
        V_rx_ofdm = sp_resample(V_rx_ac, n_ofdm_samples)

        # OFDM digital sample rate: always derive from n_ofdm_samples and the
        # physics window duration.  The modulator used this implicit rate when
        # it stretched the OFDM signal to the physics timeline, so the same
        # rate gives the correct subcarrier-to-Hz mapping for the equalizer.
        fs_ofdm = n_ofdm_samples / (n_samples * dt)

        # Equalizer: only apply when the chain has actual frequency-selective
        # components implemented in the time-domain simulation.  For the
        # 'direct' topology V_rx = I_ph × R_sense (scalar, no RC filter
        # coded in the sim), so the chain is flat and equalization is a no-op
        # that would otherwise introduce phantom phase rotation.  Non-direct
        # topologies run through the ReceiverChain which does apply filters.
        topology = getattr(cfg, 'rx_topology', 'direct')
        if topology != 'direct':
            sc_freqs = subcarrier_frequencies(fs_ofdm, cfg.ofdm_nfft, n_sc)
            equalizer = chain_response_at(cfg, sc_freqs)
        else:
            equalizer = None

        t_ofdm = np.arange(n_ofdm_samples) * (1.0 / fs_ofdm)

        ofdm_symbols: list = []
        bfsk_energies: list = []
        # Soft-demap LLRs only when FEC is going to decode them — avoids
        # the per-bit constellation distance loop on runs that won't use it.
        ofdm_llrs: list = [] if fec is not None else None
        bits_rx = demodulate('OFDM', V_rx_ofdm, t_ofdm, n_bits,
                             config=cfg, bits_tx=bits_tx,
                             symbols_out=ofdm_symbols,
                             llr_out=ofdm_llrs,
                             equalizer=equalizer)
    else:
        ofdm_symbols = []
        bfsk_energies = []
        ofdm_llrs = None
        if mod_scheme.upper() == 'BFSK':
            bits_rx = demodulate(mod_scheme, V_rx, t, n_bits, config=cfg,
                                 energies_out=bfsk_energies)
        else:
            bits_rx = demodulate(mod_scheme, V_rx, t, n_bits, config=cfg)

    # ========== DC-DC Converter (Phase 2) ==========
    dcdc_result = None
    if pv_result is not None:
        # Compute harvested power through DC-DC converter (on the harvest solve,
        # which includes ambient DC light).
        V_cell_avg = np.mean(pv_harvest.V_cell)
        if V_cell_avg > 0:
            dcdc = BoostConverter.from_config(cfg)
            dcdc_result = dcdc.compute(V_in=V_cell_avg, V_out_target=cfg.vcc_volts)

    # ========== FEC: decode (optional) ==========
    # When FEC is active, bits_rx is the channel's coded output; decode it
    # back to message bits and run BER on the (message_tx, message_rx) pair.
    # The pre-FEC channel BER is preserved on the result dict as 'ber_uncoded'
    # for diagnostics.
    ber_uncoded = None
    if fec is not None:
        ber_uncoded = calculate_ber(bits_tx, bits_rx)['ber']
        # Prefer soft demap when LLRs are available (OFDM only). Hard-bit
        # fallback (decode_bits + fixed-SNR fake LLRs) stays for non-OFDM
        # FEC users; semantics are identical when fec_decode_snr_db is set
        # to whatever the user had before.
        if ofdm_llrs:
            llr_arr = np.asarray(ofdm_llrs, dtype=float)
            # Truncate the LLR stream to the multiple-of-n length the
            # decoder consumes — matches the hard-bit path's behaviour.
            n_used = (len(llr_arr) // fec.codeword_length) * fec.codeword_length
            bits_rx = fec.decode_llrs(llr_arr[:n_used])
        else:
            bits_rx = fec.decode_bits(bits_rx, snr_db=cfg.fec_decode_snr_db)
        bits_tx = bits_tx_msg                   # report message-bit pair upstream

    # ========== BER calculation ==========
    ber_result = calculate_ber(bits_tx, bits_rx)
    if ber_uncoded is not None:
        ber_result['ber_uncoded'] = ber_uncoded

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
        # Constellation payload — see ConstellationPanel.tsx.
        # OFDM: list of complex post-FFT data-carrier symbols (I/Q).
        # BFSK: list of (E0, E1) Goertzel-energy pairs per bit.
        'ofdm_symbols': ofdm_symbols,
        'bfsk_energies': bfsk_energies,
        # Pre-FEC channel BER (None when FEC is disabled). See Phase 8/13.
        'ber_uncoded': ber_uncoded,
    }

    # Phase 2: Add per-node waveforms when enhanced models are active.
    # Harvest waveforms come from the total-light (LED + ambient) solve; the
    # LED-only data waveform stays on I_ph above.
    if pv_result is not None:
        result['V_cell'] = pv_harvest.V_cell
        result['I_cell'] = pv_harvest.I_cell
        result['I_dark'] = pv_harvest.I_dark

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

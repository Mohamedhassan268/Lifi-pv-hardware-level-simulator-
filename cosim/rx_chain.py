# cosim/rx_chain.py
"""
Receiver Signal Chain — Python model matching SPICE subcircuits.

Chain: R_sense → INA322 → BPF×N → TLV7011 Comparator → digital output

Each stage matches its SPICE subcircuit transfer function so the Python
engine can approximate SPICE waveforms without requiring a SPICE installation.

References:
    - INA322 datasheet (SBOS163): GBW=700kHz, G=100.5
    - TLV2379 datasheet: GBW=100kHz (BPF op-amp)
    - TLV7011 datasheet (SBOS819): t_pd=260ns
    - SPICE subcircuits in systems/kadirvelu2021_netlist.py
"""

import numpy as np
from scipy import signal as sp_signal
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChainWaveforms:
    """Waveforms at each node of the receiver chain."""
    t: np.ndarray
    V_sense: np.ndarray       # After R_sense
    V_ina: np.ndarray         # After INA322
    V_bpf: list               # After each BPF stage [V_bpf1, V_bpf2, ...]
    V_comp: np.ndarray        # After comparator (digital output)


class RsenseStage:
    """Current-sense resistor: V = I_cell * R_sense."""

    def __init__(self, R_sense: float = 1.0):
        self.R_sense = R_sense

    def process(self, I_cell: np.ndarray) -> np.ndarray:
        return I_cell * self.R_sense


class INAStage:
    """
    INA322 Instrumentation Amplifier — 2-pole behavioral model.

    Matches SPICE subcircuit:
        Gain = 100.5 (40 dB)
        Pole 1 (dominant): 6965 Hz
        Pole 2 (non-dominant): 69652 Hz
        Output: G*(Vinp - Vinn) + Vref, clamped to rails
    """

    def __init__(self, gain: float = 100.5,
                 f_pole1_Hz: float = 6965.0,
                 f_pole2_Hz: float = 69652.0,
                 V_ref: float = 1.65,
                 V_cc: float = 3.3,
                 V_ee: float = 0.0):
        self.gain = gain
        self.f_pole1 = f_pole1_Hz
        self.f_pole2 = f_pole2_Hz
        self.V_ref = V_ref
        self.V_cc = V_cc
        self.V_ee = V_ee

    def process(self, V_inp: np.ndarray, V_inn: np.ndarray,
                t: np.ndarray) -> np.ndarray:
        """
        Apply INA322 transfer function.

        Args:
            V_inp: Positive input voltage
            V_inn: Negative input voltage
            t: Time array

        Returns:
            Output voltage with gain, 2-pole rolloff, Vref offset, rail clamping
        """
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt

        # Differential gain
        V_diff = self.gain * (V_inp - V_inn)

        # Pole 1 (dominant)
        V_diff = self._apply_pole(V_diff, self.f_pole1, fs)

        # Pole 2 (non-dominant)
        V_diff = self._apply_pole(V_diff, self.f_pole2, fs)

        # Add Vref offset and clamp to rails
        V_out = V_diff + self.V_ref
        V_out = np.clip(V_out, self.V_ee + 0.05, self.V_cc - 0.05)
        return V_out

    @staticmethod
    def _apply_pole(signal, f_pole, fs):
        """Apply single-pole low-pass filter."""
        wn = f_pole / (fs / 2)
        if wn <= 0 or wn >= 1:
            return signal
        b, a = sp_signal.butter(1, wn, btype='low')
        return sp_signal.filtfilt(b, a, signal)

    @classmethod
    def from_config(cls, config) -> 'INAStage':
        gain_linear = 10 ** (config.ina_gain_dB / 20)
        gbw = config.ina_gbw_kHz * 1e3
        f_3db = gbw / gain_linear
        return cls(
            gain=gain_linear,
            f_pole1_Hz=f_3db,
            f_pole2_Hz=f_3db * 10,  # Non-dominant pole ~10× higher
            V_ref=config.vcc_volts / 2,
            V_cc=config.vcc_volts,
            V_ee=0.0,
        )


class BPFStage:
    """
    Band-Pass Filter — TLV2379-based active filter.

    Matches SPICE subcircuit:
        HP section: C_hp + R_hp → f_HP = 1/(2π·R_hp·C_hp)
        LP section (inverting): R_in + R_fb + C_fb → f_LP = 1/(2π·R_fb·C_fb)
        Passband gain: -R_fb/R_in (inverting)
        Op-amp with rail clamping
    """

    def __init__(self, f_hp_Hz: float = 3.39,
                 f_lp_Hz: float = 10610.0,
                 passband_gain: float = -1.0,
                 V_cc: float = 3.3,
                 V_ee: float = 0.0,
                 V_ref: float = 1.65):
        self.f_hp = f_hp_Hz
        self.f_lp = f_lp_Hz
        self.passband_gain = passband_gain
        self.V_cc = V_cc
        self.V_ee = V_ee
        self.V_ref = V_ref

    def process(self, V_in: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Apply BPF transfer function."""
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt

        # High-pass section (AC coupling)
        wn_hp = self.f_hp / (fs / 2)
        if 0 < wn_hp < 1:
            b, a = sp_signal.butter(1, wn_hp, btype='high')
            V_hp = sp_signal.filtfilt(b, a, V_in - self.V_ref) + self.V_ref
        else:
            V_hp = V_in

        # Low-pass section (active, inverting)
        wn_lp = self.f_lp / (fs / 2)
        if 0 < wn_lp < 1:
            b, a = sp_signal.butter(1, wn_lp, btype='low')
            V_ac = V_hp - self.V_ref
            V_filt = sp_signal.filtfilt(b, a, V_ac)
            V_out = self.passband_gain * V_filt + self.V_ref
        else:
            V_out = self.passband_gain * (V_hp - self.V_ref) + self.V_ref

        # Rail clamping
        V_out = np.clip(V_out, self.V_ee + 0.02, self.V_cc - 0.02)
        return V_out

    @classmethod
    def from_config(cls, config) -> 'BPFStage':
        # Compute from component values
        R_hp = config.bpf_rhp
        C_hp = config.bpf_chp_pF * 1e-12
        R_fb = config.bpf_rlp
        C_fb = config.bpf_clf_nF * 1e-9
        R_in = R_fb  # Assuming matched (Rin = Rfb for gain = -1)

        f_hp = 1.0 / (2 * np.pi * R_hp * C_hp)
        f_lp = 1.0 / (2 * np.pi * R_fb * C_fb)
        gain = -R_fb / R_in

        return cls(
            f_hp_Hz=f_hp,
            f_lp_Hz=f_lp,
            passband_gain=gain,
            V_cc=config.vcc_volts,
            V_ee=0.0,
            V_ref=config.vcc_volts / 2,
        )


class ComparatorStage:
    """
    TLV7011 Comparator — behavioral model with propagation delay.

    Matches SPICE subcircuit:
        Decision: Vout = Vcc if Vinp > Vinn, else Vee
        Propagation delay: RC model (tau = 260 ns)
    """

    def __init__(self, prop_delay_ns: float = 260.0,
                 V_cc: float = 3.3,
                 V_ee: float = 0.0,
                 offset_mV: float = 0.0):
        self.tau = prop_delay_ns * 1e-9
        self.V_cc = V_cc
        self.V_ee = V_ee
        self.offset_V = offset_mV * 1e-3

    def process(self, V_inp: np.ndarray, V_inn: np.ndarray,
                t: np.ndarray) -> np.ndarray:
        """
        Apply comparator decision + propagation delay.

        Args:
            V_inp: Positive input (signal)
            V_inn: Negative input (threshold, e.g. Vref)
            t: Time array
        """
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt

        # Decision with optional offset
        decision = np.where(V_inp - V_inn > self.offset_V, self.V_cc, self.V_ee)

        # Propagation delay via single-pole RC filter
        if self.tau > 0:
            f_delay = 1.0 / (2 * np.pi * self.tau)
            wn = f_delay / (fs / 2)
            if 0 < wn < 1:
                b, a = sp_signal.butter(1, wn, btype='low')
                delayed = sp_signal.lfilter(b, a, decision)
                # Re-threshold the delayed signal
                mid = (self.V_cc + self.V_ee) / 2
                return np.where(delayed > mid, self.V_cc, self.V_ee)
            return decision
        return decision

    @classmethod
    def from_config(cls, config) -> 'ComparatorStage':
        return cls(
            prop_delay_ns=config.comparator_prop_delay_ns,
            V_cc=config.vcc_volts,
            V_ee=0.0,
            offset_mV=config.comparator_offset_mV,
        )


class ReceiverChain:
    """
    Complete receiver signal chain: R_sense → INA → BPF×N → Comparator.

    Usage:
        chain = ReceiverChain.from_config(cfg)
        waveforms = chain.process(I_cell, t)
    """

    def __init__(self, r_sense: RsenseStage,
                 ina: INAStage,
                 bpf_stages: list,
                 comparator: ComparatorStage):
        self.r_sense = r_sense
        self.ina = ina
        self.bpf_stages = bpf_stages
        self.comparator = comparator

    @classmethod
    def from_config(cls, config) -> 'ReceiverChain':
        r_sense = RsenseStage(config.r_sense_ohm)
        ina = INAStage.from_config(config)
        bpf_list = [BPFStage.from_config(config) for _ in range(config.bpf_stages)]
        comp = ComparatorStage.from_config(config)
        return cls(r_sense, ina, bpf_list, comp)

    def process(self, I_cell: np.ndarray, t: np.ndarray) -> ChainWaveforms:
        """
        Run the full receiver chain.

        Args:
            I_cell: PV cell output current (A)
            t: Time array (s)

        Returns:
            ChainWaveforms with voltage at each node
        """
        # R_sense: I → V
        V_sense = self.r_sense.process(I_cell)

        # INA322: differential amp
        # INP = sense_lo (0V ref), INN = sc_cathode (negative V_sense)
        # In the actual circuit: V_inp = 0 (ground ref), V_inn = -V_sense
        # So differential = V_inp - V_inn = V_sense
        V_zero = np.zeros_like(V_sense)
        V_ina = self.ina.process(V_zero, -V_sense, t)

        # BPF stages
        V_bpf_list = []
        V_current = V_ina
        for bpf in self.bpf_stages:
            V_current = bpf.process(V_current, t)
            V_bpf_list.append(V_current.copy())

        # Comparator: signal vs Vref
        V_ref = np.full_like(V_current, self.ina.V_ref)
        V_comp = self.comparator.process(V_current, V_ref, t)

        return ChainWaveforms(
            t=t,
            V_sense=V_sense,
            V_ina=V_ina,
            V_bpf=V_bpf_list,
            V_comp=V_comp,
        )

    def __repr__(self) -> str:
        return (f"ReceiverChain(Rsense={self.r_sense.R_sense}ohm, "
                f"INA_gain={self.ina.gain:.1f}, "
                f"BPF_stages={len(self.bpf_stages)}, "
                f"comp_delay={self.comparator.tau*1e9:.0f}ns)")

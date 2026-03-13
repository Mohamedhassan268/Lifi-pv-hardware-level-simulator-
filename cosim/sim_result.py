# cosim/sim_result.py
"""
Unified Simulation Result — Standard output format for all engines.

Both the Python engine and SPICE pipeline produce a SimulationResult
with the same fields, enabling engine-agnostic downstream processing
(plotting, comparison, serialization).

Usage:
    from cosim.sim_result import SimulationResult

    result = SimulationResult.from_python_dict(py_result)
    result = SimulationResult.from_spice_raw(raw_path, bits_tx, config)
    result.save('output_dir/')
    loaded = SimulationResult.load('output_dir/')
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path


@dataclass
class SimulationResult:
    """Unified simulation output from any engine."""

    # -- Core waveforms (always present) --
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    P_tx: np.ndarray = field(default_factory=lambda: np.array([]))
    P_rx: np.ndarray = field(default_factory=lambda: np.array([]))
    I_ph: np.ndarray = field(default_factory=lambda: np.array([]))
    V_rx: np.ndarray = field(default_factory=lambda: np.array([]))

    # -- Digital data --
    bits_tx: np.ndarray = field(default_factory=lambda: np.array([]))
    bits_rx: np.ndarray = field(default_factory=lambda: np.array([]))

    # -- Metrics --
    ber: float = 0.0
    n_errors: int = 0
    n_bits_tested: int = 0
    snr_est_dB: float = 0.0

    # -- Metadata --
    engine: str = 'python'          # 'python', 'spice', 'hybrid'
    modulation: str = 'OOK'
    channel_gain: float = 0.0
    P_rx_avg_uW: float = 0.0
    I_ph_avg_uA: float = 0.0

    # -- Per-node waveforms (optional, from Phase 2 models) --
    V_cell: Optional[np.ndarray] = None
    I_cell: Optional[np.ndarray] = None
    V_sense: Optional[np.ndarray] = None
    V_ina: Optional[np.ndarray] = None
    V_bpf: Optional[list] = None       # [V_bpf1, V_bpf2, ...]
    V_comp: Optional[np.ndarray] = None
    V_dcdc: Optional[np.ndarray] = None

    # -- DC-DC results (optional) --
    dcdc_efficiency: Optional[float] = None
    dcdc_mode: Optional[str] = None
    dcdc_V_out: Optional[float] = None
    dcdc_P_out_uW: Optional[float] = None

    # -- Noise breakdown (optional) --
    noise_breakdown: Optional[Dict] = None

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_python_dict(cls, d: Dict) -> 'SimulationResult':
        """Create from run_python_simulation() output dict."""
        result = cls(
            time=d.get('time', np.array([])),
            P_tx=d.get('P_tx', np.array([])),
            P_rx=d.get('P_rx', np.array([])),
            I_ph=d.get('I_ph', np.array([])),
            V_rx=d.get('V_rx', np.array([])),
            bits_tx=d.get('bits_tx', np.array([])),
            bits_rx=d.get('bits_rx', np.array([])),
            ber=d.get('ber', 0.0),
            n_errors=d.get('n_errors', 0),
            n_bits_tested=d.get('n_bits_tested', 0),
            snr_est_dB=d.get('snr_est_dB', 0.0),
            engine=d.get('engine', 'python'),
            modulation=d.get('modulation', 'OOK'),
            channel_gain=d.get('channel_gain', 0.0),
            P_rx_avg_uW=d.get('P_rx_avg_uW', 0.0),
            I_ph_avg_uA=d.get('I_ph_avg_uA', 0.0),
        )

        # Optional per-node waveforms
        if 'V_cell' in d:
            result.V_cell = d['V_cell']
        if 'I_cell' in d:
            result.I_cell = d['I_cell']
        if 'V_sense' in d:
            result.V_sense = d['V_sense']
        if 'V_ina' in d:
            result.V_ina = d['V_ina']
        if 'V_bpf' in d:
            result.V_bpf = d['V_bpf']
        if 'V_comp' in d:
            result.V_comp = d['V_comp']

        # DC-DC
        if 'dcdc_efficiency' in d:
            result.dcdc_efficiency = d['dcdc_efficiency']
            result.dcdc_mode = d.get('dcdc_mode')
            result.dcdc_V_out = d.get('dcdc_V_out')
            result.dcdc_P_out_uW = d.get('dcdc_P_out_uW')

        return result

    @classmethod
    def from_spice_raw(cls, raw_path, bits_tx: np.ndarray,
                       config) -> 'SimulationResult':
        """Create from SPICE .raw file + TX bits."""
        from .spice_extract import extract_spice_waveforms, compute_ber_from_spice

        waveforms = extract_spice_waveforms(raw_path)
        ber_result = compute_ber_from_spice(
            raw_path, bits_tx,
            data_rate_bps=config.data_rate_bps,
            vcc=config.vcc_volts,
        )

        from .channel import OpticalChannel
        channel = OpticalChannel.from_config(config)

        result = cls(
            time=waveforms.get('time', np.array([])),
            P_tx=np.array([]),  # Not in .raw
            P_rx=waveforms.get('P_rx', np.array([])),
            I_ph=waveforms.get('I_sense', np.array([])),
            V_rx=waveforms.get('V_comp', np.array([])),
            bits_tx=bits_tx,
            bits_rx=ber_result.get('bits_rx', np.array([])),
            ber=ber_result['ber'],
            n_errors=ber_result['n_errors'],
            n_bits_tested=ber_result['n_bits_tested'],
            snr_est_dB=ber_result.get('snr_est_dB', 0.0),
            engine='spice',
            modulation=config.modulation,
            channel_gain=channel.channel_gain(),
        )

        # Map available waveforms
        if 'V_cell' in waveforms:
            result.V_cell = waveforms['V_cell']
        if 'V_ina' in waveforms:
            result.V_ina = waveforms['V_ina']
        if 'V_bpf1' in waveforms or 'V_bpf2' in waveforms:
            bpf_list = []
            if 'V_bpf1' in waveforms:
                bpf_list.append(waveforms['V_bpf1'])
            if 'V_bpf2' in waveforms:
                bpf_list.append(waveforms['V_bpf2'])
            result.V_bpf = bpf_list
        if 'V_comp' in waveforms:
            result.V_comp = waveforms['V_comp']
        if 'V_dcdc' in waveforms:
            result.V_dcdc = waveforms['V_dcdc']

        return result

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Convert to the legacy dict format for backward compatibility."""
        d = {
            'time': self.time,
            'P_tx': self.P_tx,
            'P_rx': self.P_rx,
            'I_ph': self.I_ph,
            'V_rx': self.V_rx,
            'bits_tx': self.bits_tx,
            'bits_rx': self.bits_rx,
            'ber': self.ber,
            'n_errors': self.n_errors,
            'n_bits_tested': self.n_bits_tested,
            'snr_est_dB': self.snr_est_dB,
            'engine': self.engine,
            'modulation': self.modulation,
            'channel_gain': self.channel_gain,
            'P_rx_avg_uW': self.P_rx_avg_uW,
            'I_ph_avg_uA': self.I_ph_avg_uA,
        }

        if self.V_cell is not None:
            d['V_cell'] = self.V_cell
        if self.I_cell is not None:
            d['I_cell'] = self.I_cell
        if self.V_sense is not None:
            d['V_sense'] = self.V_sense
        if self.V_ina is not None:
            d['V_ina'] = self.V_ina
        if self.V_bpf is not None:
            d['V_bpf'] = self.V_bpf
        if self.V_comp is not None:
            d['V_comp'] = self.V_comp
        if self.V_dcdc is not None:
            d['V_dcdc'] = self.V_dcdc
        if self.dcdc_efficiency is not None:
            d['dcdc_efficiency'] = self.dcdc_efficiency
            d['dcdc_mode'] = self.dcdc_mode
            d['dcdc_V_out'] = self.dcdc_V_out
            d['dcdc_P_out_uW'] = self.dcdc_P_out_uW

        return d

    def save(self, output_dir) -> None:
        """
        Save result to directory as .npz (waveforms) + .json (summary).

        Args:
            output_dir: Directory to save files to
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Waveform arrays
        arrays = {
            'time': self.time,
            'P_tx': self.P_tx,
            'P_rx': self.P_rx,
            'I_ph': self.I_ph,
            'V_rx': self.V_rx,
            'bits_tx': self.bits_tx,
            'bits_rx': self.bits_rx,
        }
        for key, arr in [('V_cell', self.V_cell), ('I_cell', self.I_cell),
                         ('V_sense', self.V_sense), ('V_ina', self.V_ina),
                         ('V_comp', self.V_comp), ('V_dcdc', self.V_dcdc)]:
            if arr is not None:
                arrays[key] = arr
        if self.V_bpf:
            for i, bpf in enumerate(self.V_bpf):
                arrays[f'V_bpf{i}'] = bpf

        np.savez(out / 'waveforms.npz', **arrays)

        # Summary JSON
        summary = {
            'engine': self.engine,
            'modulation': self.modulation,
            'ber': self.ber,
            'n_errors': self.n_errors,
            'n_bits_tested': self.n_bits_tested,
            'snr_est_dB': self.snr_est_dB,
            'channel_gain': self.channel_gain,
            'P_rx_avg_uW': self.P_rx_avg_uW,
            'I_ph_avg_uA': self.I_ph_avg_uA,
        }
        if self.dcdc_efficiency is not None:
            summary['dcdc_efficiency'] = self.dcdc_efficiency
            summary['dcdc_mode'] = self.dcdc_mode
            summary['dcdc_V_out'] = self.dcdc_V_out
            summary['dcdc_P_out_uW'] = self.dcdc_P_out_uW

        (out / 'summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, output_dir) -> 'SimulationResult':
        """Load result from directory saved by save()."""
        out = Path(output_dir)

        # Load waveforms
        npz = np.load(out / 'waveforms.npz')

        # Load summary
        summary = json.loads(
            (out / 'summary.json').read_text(encoding='utf-8'))

        result = cls(
            time=npz['time'],
            P_tx=npz['P_tx'],
            P_rx=npz['P_rx'],
            I_ph=npz['I_ph'],
            V_rx=npz['V_rx'],
            bits_tx=npz['bits_tx'],
            bits_rx=npz['bits_rx'],
            ber=summary['ber'],
            n_errors=summary['n_errors'],
            n_bits_tested=summary['n_bits_tested'],
            snr_est_dB=summary['snr_est_dB'],
            engine=summary['engine'],
            modulation=summary['modulation'],
            channel_gain=summary['channel_gain'],
            P_rx_avg_uW=summary.get('P_rx_avg_uW', 0.0),
            I_ph_avg_uA=summary.get('I_ph_avg_uA', 0.0),
        )

        # Load optional waveforms
        for key in ('V_cell', 'I_cell', 'V_sense', 'V_ina', 'V_comp', 'V_dcdc'):
            if key in npz:
                setattr(result, key, npz[key])

        # Load BPF stages
        bpf_list = []
        for i in range(10):
            bpf_key = f'V_bpf{i}'
            if bpf_key in npz:
                bpf_list.append(npz[bpf_key])
        if bpf_list:
            result.V_bpf = bpf_list

        # DC-DC from summary
        if 'dcdc_efficiency' in summary:
            result.dcdc_efficiency = summary['dcdc_efficiency']
            result.dcdc_mode = summary.get('dcdc_mode')
            result.dcdc_V_out = summary.get('dcdc_V_out')
            result.dcdc_P_out_uW = summary.get('dcdc_P_out_uW')

        return result

    def __repr__(self) -> str:
        return (f"SimulationResult(engine={self.engine}, mod={self.modulation}, "
                f"BER={self.ber:.4e}, SNR={self.snr_est_dB:.1f}dB)")

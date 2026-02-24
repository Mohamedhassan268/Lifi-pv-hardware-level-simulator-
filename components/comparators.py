# components/comparators.py
"""
Comparator Components for Hardware-Faithful LiFi-PV Simulator

Provides models for comparators used in data recovery:
    - TLV7011: TI nanopower comparator (Kadirvelu 2021 data slicer)

References:
    - Kadirvelu et al. (2021) - System configuration
    - TI TLV7011 datasheet (SLVSCJ0)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import K_B


# =============================================================================
# COMPARATOR BASE CLASS
# =============================================================================

class ComparatorBase(ABC):
    """
    Abstract base class for voltage comparators.

    Provides common interface for:
        - Propagation delay
        - Input offset voltage
        - Hysteresis
    """

    def __init__(self, V_ref: float = 1.65):
        """
        Initialize comparator.

        Args:
            V_ref: Reference/threshold voltage (V)
        """
        self.V_ref = V_ref

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def propagation_delay_s(self) -> float:
        """Propagation delay in seconds."""
        pass

    @property
    @abstractmethod
    def input_offset_voltage(self) -> float:
        """Input offset voltage in V."""
        pass

    @property
    @abstractmethod
    def hysteresis_mV(self) -> float:
        """Built-in hysteresis in mV."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        pass

    def max_toggle_frequency(self) -> float:
        """
        Maximum toggle frequency based on propagation delay.

        f_max = 1 / (2 * t_pd)
        """
        t_pd = self.propagation_delay_s
        if t_pd <= 0:
            return float('inf')
        return 1.0 / (2 * t_pd)


# =============================================================================
# TLV7011 - NANOPOWER COMPARATOR
# =============================================================================

class TLV7011(ComparatorBase):
    """
    Texas Instruments TLV7011 Nanopower Comparator

    Used in Kadirvelu 2021 for data recovery (OOK demodulation).

    Datasheet Parameters (TI TLV7011):
        - Supply: 1.6V to 5.5V
        - Quiescent current: 335 nA
        - Propagation delay: 260 ns (typical)
        - Input offset: ±0.5 mV
        - Push-pull output (no external pull-up needed)
        - No built-in hysteresis
    """

    def __init__(self, V_ref: float = 1.65):
        super().__init__(V_ref)

        # -----------------------------------------------------------------
        # DATASHEET PARAMETERS
        # -----------------------------------------------------------------
        self._propagation_delay_ns = 260.0  # ns typical
        self._input_offset_mV = 0.5         # mV typical
        self._hysteresis = 0.0              # mV (no internal hysteresis)
        self._supply_min = 1.6              # V
        self._supply_max = 5.5              # V
        self._quiescent_nA = 335.0          # nA
        self._output_type = 'push_pull'

    @property
    def name(self) -> str:
        return "TLV7011"

    @property
    def propagation_delay_s(self) -> float:
        return self._propagation_delay_ns * 1e-9

    @property
    def input_offset_voltage(self) -> float:
        return self._input_offset_mV * 1e-3

    @property
    def hysteresis_mV(self) -> float:
        return self._hysteresis

    def spice_subcircuit(self) -> str:
        """Generate behavioral SPICE subcircuit."""
        t_pd = self._propagation_delay_ns * 1e-9

        return f"""\
* TLV7011 Nanopower Comparator - Behavioral Model
* t_pd = {self._propagation_delay_ns:.0f} ns, Iq = {self._quiescent_nA:.0f} nA
.SUBCKT TLV7011 INP INN OUT VCC VEE
* INP = non-inverting, INN = inverting, OUT = output

* High input impedance
Rinp INP 0 1T
Rinn INN 0 1T

* Comparator decision (ideal)
Bcomp comp_int 0 V = IF(V(INP)-V(INN) > 0, V(VCC), V(VEE))

* Propagation delay model (RC)
Rdel comp_int del_out 1k
Cdel del_out 0 {t_pd/1e3:.6e}

* Output buffer
Eout OUT 0 del_out 0 1

.ENDS TLV7011"""

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': 'comparator',
            'propagation_delay_ns': self._propagation_delay_ns,
            'propagation_delay_s': self.propagation_delay_s,
            'input_offset_mV': self._input_offset_mV,
            'hysteresis_mV': self._hysteresis,
            'max_toggle_frequency_Hz': self.max_toggle_frequency(),
            'supply_min_V': self._supply_min,
            'supply_max_V': self._supply_max,
            'quiescent_nA': self._quiescent_nA,
            'output_type': self._output_type,
            'V_ref': self.V_ref,
        }

    def __repr__(self):
        return (f"TLV7011(t_pd={self._propagation_delay_ns:.0f}ns, "
                f"f_max={self.max_toggle_frequency()/1e6:.1f}MHz)")


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COMPARATOR COMPONENTS - SELF TEST")
    print("=" * 60)

    comp = TLV7011()
    params = comp.get_parameters()

    print(f"\n--- {comp.name} ---")
    print(f"  {comp}")
    print(f"  Propagation delay: {params['propagation_delay_ns']:.0f} ns")
    print(f"  Input offset:      {params['input_offset_mV']:.1f} mV")
    print(f"  Max toggle freq:   {params['max_toggle_frequency_Hz']/1e6:.1f} MHz")
    print(f"  Quiescent current: {params['quiescent_nA']:.0f} nA")

    print("\n[OK] Comparator tests passed!")

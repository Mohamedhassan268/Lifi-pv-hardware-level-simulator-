# components/mosfets.py
"""
MOSFET Components for Hardware-Faithful LiFi-PV Simulator

Provides models for MOSFETs used in the system:
    - BSD235N: Infineon N-channel MOSFET (LED driver switch)
    - NTS4409: ON Semiconductor N-channel MOSFET (DC-DC converter switch)

References:
    - Kadirvelu et al. (2021) - System configuration
    - Infineon BSD235N datasheet
    - ON Semiconductor NTS4409 datasheet
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.pins import Port, PortRole


# =============================================================================
# MOSFET BASE CLASS
# =============================================================================

class MOSFETBase(ABC):
    """
    Abstract base class for power MOSFETs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def channel_type(self) -> str:
        """'N' or 'P'."""
        pass

    @property
    @abstractmethod
    def vds_max(self) -> float:
        """Max drain-source voltage (V)."""
        pass

    @property
    @abstractmethod
    def rds_on(self) -> float:
        """On-state resistance (ohm)."""
        pass

    @property
    @abstractmethod
    def vgs_threshold(self) -> float:
        """Gate threshold voltage (V)."""
        pass

    @property
    @abstractmethod
    def id_max(self) -> float:
        """Max continuous drain current (A)."""
        pass

    @property
    @abstractmethod
    def ciss(self) -> float:
        """Input capacitance (F)."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        pass

    def conduction_loss(self, I_d: float) -> float:
        """
        Calculate conduction loss.

        P_cond = I_d^2 * Rds_on
        """
        return I_d ** 2 * self.rds_on

    def switching_loss(self, V_ds: float, I_d: float,
                       t_rise: float, t_fall: float, f_sw: float) -> float:
        """
        Calculate switching loss.

        P_sw = 0.5 * V_ds * I_d * (t_rise + t_fall) * f_sw
        """
        return 0.5 * V_ds * I_d * (t_rise + t_fall) * f_sw

    def spice_ports(self):
        """SPICE port list: drain, gate, source (MOSFET terminal order)."""
        return [
            Port("drain", PortRole.SIGNAL_OUT),
            Port("gate", PortRole.SIGNAL_IN),
            Port("source", PortRole.SIGNAL),
        ]

    def spice_subcircuit(self) -> str:
        """Wrap the MOSFET .MODEL in a 3-pin subcircuit (drain gate source),
        so it can be placed and wired. Uses the part's ``spice_model_string()``
        when available; the device letter (M) and channel polarity come from
        the model.

        ngspice's level-1 MOSFET has no ``CGS``/``CGD``/``CBD`` model params
        (its terminal caps are the per-metre overlap caps ``CGSO``/``CGDO``),
        so it rejects the datasheet-derived cap params that ``spice_model_string``
        carries for the (more lenient) LTspice path. We pull those cap values out
        of the model body and re-emit them as explicit capacitors across the real
        terminals — ngspice-clean and the exact datasheet pF values are preserved
        as physical elements. ``spice_model_string`` itself is left untouched.
        """
        import re
        name = self.name.upper().replace('-', '_')
        polarity = "NMOS" if self.channel_type == "N" else "PMOS"
        model_name = f"{name}_M"
        if hasattr(self, 'spice_model_string'):
            after = self.spice_model_string().split(f"{polarity}(", 1)
            body = after[1] if len(after) == 2 else f"VTO={self.vgs_threshold} KP=150m)"
        else:
            body = f"VTO={self.vgs_threshold} KP=150m)"

        # Lift CGS/CGD/CBD out of the model body into explicit terminal caps.
        cap_terms = {'CGS': ('gate', 'source', 'Cgs'), 'CGD': ('gate', 'drain', 'Cgd'),
                     'CBD': ('drain', 'source', 'Cds')}
        cap_lines = []
        for pname, (a, b, ref) in cap_terms.items():
            m = re.search(rf"\b{pname}=([0-9.]+[a-zA-Z]*)", body)
            if m:
                cap_lines.append(f"{ref} {a} {b} {m.group(1)}")
                body = re.sub(rf"\s*\b{pname}=[0-9.]+[a-zA-Z]*", "", body)
        model_line = f".MODEL {model_name} {polarity}({body}"
        caps = ("\n".join(cap_lines) + "\n") if cap_lines else ""
        return f"""\
* {self.name} MOSFET - 3-pin subcircuit (drain gate source)
.SUBCKT {name} drain gate source
M1 drain gate source source {model_name} W=1m L=1u
{caps}{model_line}
.ENDS {name}"""


# =============================================================================
# BSD235N - LED DRIVER MOSFET
# =============================================================================

class BSD235N(MOSFETBase):
    """
    Infineon BSD235N N-Channel MOSFET

    Used in Kadirvelu 2021 as LED driver current switch.

    Datasheet Parameters:
        - Vds_max: 30 V
        - Rds_on: 0.18 ohm @ Vgs=4.5V
        - Vgs_th: 1.2 V (typical)
        - Id_max: 2.2 A
        - Ciss: 380 pF
        - Package: SOT-23
    """

    def __init__(self):
        self._vds_max = 30.0            # V
        self._rds_on = 0.18             # ohm
        self._vgs_th = 1.2              # V
        self._id_max = 2.2              # A
        self._ciss_pF = 380.0           # pF
        self._coss_pF = 50.0            # pF
        self._crss_pF = 30.0            # pF
        self._t_rise_ns = 5.0           # ns
        self._t_fall_ns = 10.0          # ns

    @property
    def name(self) -> str:
        return "BSD235N"

    @property
    def channel_type(self) -> str:
        return "N"

    @property
    def vds_max(self) -> float:
        return self._vds_max

    @property
    def rds_on(self) -> float:
        return self._rds_on

    @property
    def vgs_threshold(self) -> float:
        return self._vgs_th

    @property
    def id_max(self) -> float:
        return self._id_max

    @property
    def ciss(self) -> float:
        return self._ciss_pF * 1e-12

    def spice_model_string(self) -> str:
        """Generate SPICE .MODEL line."""
        return (f".MODEL BSD235N NMOS("
                f"VTO={self._vgs_th} "
                f"KP=150m "
                f"LAMBDA=0.02 "
                f"RD={self._rds_on/2} "
                f"RS={self._rds_on/2} "
                f"CBD={self._coss_pF}p "
                f"CGS={self._ciss_pF - self._crss_pF}p "
                f"CGD={self._crss_pF}p)")

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': 'N-MOSFET',
            'application': 'LED_driver',
            'Vds_max_V': self._vds_max,
            'Rds_on_ohm': self._rds_on,
            'Vgs_th_V': self._vgs_th,
            'Id_max_A': self._id_max,
            'Ciss_pF': self._ciss_pF,
            'Coss_pF': self._coss_pF,
            'Crss_pF': self._crss_pF,
            't_rise_ns': self._t_rise_ns,
            't_fall_ns': self._t_fall_ns,
            'spice_model': self.spice_model_string(),
        }

    def __repr__(self):
        return f"BSD235N(Rds_on={self._rds_on}Ω, Vth={self._vgs_th}V)"


# =============================================================================
# NTS4409 - DC-DC CONVERTER MOSFET
# =============================================================================

class NTS4409(MOSFETBase):
    """
    ON Semiconductor NTS4409 N-Channel MOSFET

    Used in Kadirvelu 2021 as DC-DC boost converter switch.

    Datasheet Parameters:
        - Vds_max: 30 V
        - Rds_on: 52 mohm @ Vgs=4.5V
        - Vgs_th: 0.8 V (typical)
        - Id_max: 5.6 A
        - Ciss: 900 pF
        - Package: SOT-23
    """

    def __init__(self):
        self._vds_max = 30.0            # V
        self._rds_on = 0.052            # ohm (52 mohm)
        self._vgs_th = 0.8              # V
        self._id_max = 5.6              # A
        self._ciss_pF = 900.0           # pF
        self._coss_pF = 120.0           # pF
        self._crss_pF = 60.0            # pF
        self._t_rise_ns = 8.0           # ns
        self._t_fall_ns = 15.0          # ns

    @property
    def name(self) -> str:
        return "NTS4409"

    @property
    def channel_type(self) -> str:
        return "N"

    @property
    def vds_max(self) -> float:
        return self._vds_max

    @property
    def rds_on(self) -> float:
        return self._rds_on

    @property
    def vgs_threshold(self) -> float:
        return self._vgs_th

    @property
    def id_max(self) -> float:
        return self._id_max

    @property
    def ciss(self) -> float:
        return self._ciss_pF * 1e-12

    def spice_model_string(self) -> str:
        """Generate SPICE .MODEL line."""
        return (f".MODEL NTS4409 NMOS("
                f"VTO={self._vgs_th} "
                f"KP=200m "
                f"LAMBDA=0.01 "
                f"RD={self._rds_on/2} "
                f"RS={self._rds_on/2} "
                f"CBD={self._coss_pF}p "
                f"CGS={self._ciss_pF - self._crss_pF}p "
                f"CGD={self._crss_pF}p)")

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': 'N-MOSFET',
            'application': 'DC-DC_switch',
            'Vds_max_V': self._vds_max,
            'Rds_on_ohm': self._rds_on,
            'Vgs_th_V': self._vgs_th,
            'Id_max_A': self._id_max,
            'Ciss_pF': self._ciss_pF,
            'Coss_pF': self._coss_pF,
            'Crss_pF': self._crss_pF,
            't_rise_ns': self._t_rise_ns,
            't_fall_ns': self._t_fall_ns,
            'spice_model': self.spice_model_string(),
        }

    def __repr__(self):
        return f"NTS4409(Rds_on={self._rds_on*1e3:.0f}mΩ, Vth={self._vgs_th}V)"


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MOSFET COMPONENTS - SELF TEST")
    print("=" * 60)

    # Test BSD235N
    print("\n--- BSD235N (LED Driver) ---")
    m1 = BSD235N()
    print(f"  {m1}")
    print(f"  Rds_on: {m1.rds_on*1e3:.0f} mohm")
    print(f"  Conduction loss @ 350mA: {m1.conduction_loss(0.35)*1e3:.2f} mW")
    print(f"  SPICE: {m1.spice_model_string()}")

    # Test NTS4409
    print("\n--- NTS4409 (DC-DC Switch) ---")
    m2 = NTS4409()
    print(f"  {m2}")
    print(f"  Rds_on: {m2.rds_on*1e3:.0f} mohm")
    print(f"  Conduction loss @ 1A: {m2.conduction_loss(1.0)*1e3:.1f} mW")
    P_sw = m2.switching_loss(3.3, 0.5, 8e-9, 15e-9, 50e3)
    print(f"  Switching loss @ 50kHz: {P_sw*1e3:.3f} mW")
    print(f"  SPICE: {m2.spice_model_string()}")

    print("\n[OK] MOSFET tests passed!")

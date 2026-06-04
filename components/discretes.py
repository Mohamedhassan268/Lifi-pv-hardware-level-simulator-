# components/discretes.py
"""
Discrete semiconductor components for the breadboard LiFi PoC.

    - BJT2N2222    : NPN BJT used as the LED low-side switch (TX driver)
    - ZenerBZX84C3V3: 3.3 V Zener clamp protecting the ESP32 ADC pin (RX)

Both expose the same ``spice_ports()`` / ``spice_subcircuit()`` interface the
schematic editor and the graph->netlist runner use, so they can be placed and
wired like any library part. Unlike the behavioral op-amp / PV macromodels,
these wrap ngspice's built-in device primitives (Q card + .model, D + .model)
with standard datasheet Gummel-Poon / diode parameters.
"""

from typing import Any, Dict, List

from components.pins import Port, PortRole


class BJT2N2222:
    """NPN BJT (2N2222 / PN2222). Common-emitter LED switch in the PoC TX:
    GPIO -> R1 -> base, collector -> LED cathode via R_LED, emitter -> GND.

    SPICE: ngspice built-in BJT with standard 2N2222 Gummel-Poon parameters,
    wrapped in a 3-pin subcircuit (collector base emitter)."""

    def __init__(self, temperature_K: float = 300.0):
        self.temperature_K = temperature_K
        # Datasheet-derived Gummel-Poon parameters (ON Semi PN2222A).
        self.spice_BF = 200.0      # forward beta
        self.spice_VAF = 100.0     # Early voltage (V)
        self.spice_IS = 1e-14      # transport saturation current (A)
        self.spice_RB = 10.0       # base resistance (ohm)
        self.spice_RC = 0.4        # collector resistance (ohm)
        self.spice_RE = 0.2        # emitter resistance (ohm)
        self.spice_CJC = 8e-12     # B-C junction cap (F)
        self.spice_CJE = 25e-12    # B-E junction cap (F)
        self.spice_TF = 0.4e-9     # forward transit time (s)

    @property
    def name(self) -> str:
        return "2N2222"

    def spice_ports(self) -> List[Port]:
        return [
            Port("collector", PortRole.SIGNAL),
            Port("base", PortRole.SIGNAL_IN),
            Port("emitter", PortRole.SIGNAL),
        ]

    def spice_subcircuit(self) -> str:
        return f"""\
* 2N2222 NPN BJT - ngspice built-in device, datasheet Gummel-Poon params
.SUBCKT BJT_2N2222 collector base emitter
Q1 collector base emitter QN2222
.MODEL QN2222 NPN(IS={self.spice_IS:.2e} BF={self.spice_BF} VAF={self.spice_VAF} \
RB={self.spice_RB} RC={self.spice_RC} RE={self.spice_RE} \
CJC={self.spice_CJC:.2e} CJE={self.spice_CJE:.2e} TF={self.spice_TF:.2e})
.ENDS BJT_2N2222"""

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "bjt",
            "forward_beta": self.spice_BF,
            "early_voltage_V": self.spice_VAF,
            "saturation_current_A": self.spice_IS,
            "transit_time_s": self.spice_TF,
        }

    def __repr__(self) -> str:
        return f"BJT2N2222(BF={self.spice_BF:.0f}, VAF={self.spice_VAF:.0f}V)"


class ZenerBZX84C3V3:
    """3.3 V Zener diode (BZX84-C3V3). ADC overvoltage clamp on the PoC RX:
    anode at the ADC node, cathode to GND, so it conducts (clamps) only if the
    signal swings beyond ~3.3 V. Below that it's a reverse-biased junction with
    no effect on the link.

    SPICE: ngspice diode with a breakdown voltage (BV) at the Zener voltage."""

    def __init__(self, temperature_K: float = 300.0):
        self.temperature_K = temperature_K
        self.zener_voltage_V = 3.3
        self.spice_IS = 1e-14
        self.spice_N = 1.2
        self.spice_RS = 10.0
        self.spice_CJO = 200e-12
        self.spice_IBV = 5e-3      # current at breakdown reference

    @property
    def name(self) -> str:
        return "BZX84C3V3"

    def spice_ports(self) -> List[Port]:
        return [
            Port("anode", PortRole.SIGNAL),
            Port("cathode", PortRole.SIGNAL),
        ]

    def spice_subcircuit(self) -> str:
        return f"""\
* BZX84-C3V3 3.3V Zener clamp - ngspice diode with breakdown
.SUBCKT BZX84C3V3 anode cathode
D1 anode cathode DZ33
.MODEL DZ33 D(IS={self.spice_IS:.2e} N={self.spice_N} RS={self.spice_RS} \
BV={self.zener_voltage_V} IBV={self.spice_IBV:.2e} CJO={self.spice_CJO:.2e})
.ENDS BZX84C3V3"""

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "zener_diode",
            "zener_voltage_V": self.zener_voltage_V,
            "junction_capacitance_F": self.spice_CJO,
        }

    def __repr__(self) -> str:
        return f"ZenerBZX84C3V3(Vz={self.zener_voltage_V}V)"

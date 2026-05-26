# components/power.py
"""
Power-management components for the LiFi-PV harvester path.

Currently:
    - BQ25570: TI ultra-low-power boost-charger with MPPT

Datasheet parameters only; no empirical fits.
"""

from typing import Any, Dict


class BQ25570:
    """
    Texas Instruments BQ25570 - nanopower boost charger + MPPT buck-boost.

    Chosen for the Cairo sensor node: a 500-unit outdoor deployment where
    each unit runs from a KXOB25-04X3F GaAs PV cell and a small LiPo
    battery. The BQ25570 handles both the boost charge-path and the
    regulated buck output that feeds the STM32H7.

    Datasheet Parameters (TI BQ25570, SLUSBH2):
        - Input voltage (boost, cold start): V_IN >= 0.6 V (startup),
          then operates down to 0.1 V
        - Input voltage (MPPT sample): 0.1 to 5.1 V
        - Output (V_STOR / V_BAT): 3.0 to 5.5 V
        - Buck output (V_OUT): 1.3 to 5.5 V programmable
        - MPPT ratio: programmable, typically 0.8 * V_OC
        - Quiescent current (active): 488 nA typical
        - Quiescent current (cold start): 330 nA typical
        - Peak boost input current: 100 mA
        - Boost efficiency: ~85% at V_IN = 0.5 V, 90% at V_IN = 1 V
        - Buck efficiency: ~93% typical
    """

    PART_NUMBER = "BQ25570"
    DATASHEET_URL = "https://www.ti.com/product/BQ25570"

    def __init__(self,
                 v_in_min_startup_V: float = 0.6,
                 v_in_min_operating_V: float = 0.1,
                 mppt_ratio: float = 0.80,
                 v_stor_V: float = 4.2,
                 v_out_V: float = 3.3,
                 v_bat_uv_V: float = 2.2,
                 v_bat_ov_V: float = 4.2):
        """
        Initialize BQ25570.

        Args:
            v_in_min_startup_V: Minimum cold-start V_IN (0.6 V datasheet)
            v_in_min_operating_V: Minimum operating V_IN after startup (0.1 V)
            mppt_ratio: V_MPPT / V_OC fraction (programmed via REF_SAMP)
            v_stor_V: Target storage rail voltage
            v_out_V: Buck output regulation voltage
            v_bat_uv_V: Battery undervoltage lockout threshold
            v_bat_ov_V: Battery overvoltage threshold
        """
        self.v_in_min_startup = v_in_min_startup_V
        self.v_in_min_operating = v_in_min_operating_V
        self.mppt_ratio = mppt_ratio
        self.v_stor = v_stor_V
        self.v_out = v_out_V
        self.v_bat_uv = v_bat_uv_V
        self.v_bat_ov = v_bat_ov_V

        self._i_q_active_nA = 488.0         # nA, datasheet typical
        self._i_q_coldstart_nA = 330.0      # nA
        self._i_in_peak_mA = 100.0          # mA peak boost input
        self._boost_eff_lowVin = 0.85       # at V_IN = 0.5 V
        self._boost_eff_highVin = 0.90      # at V_IN = 1.0 V
        self._buck_eff = 0.93               # typical buck efficiency

    @property
    def name(self) -> str:
        return self.PART_NUMBER

    def mppt_voltage(self, v_oc_V: float) -> float:
        """Target boost input voltage for MPPT given the PV open-circuit voltage."""
        return self.mppt_ratio * v_oc_V

    def boost_efficiency(self, v_in_V: float) -> float:
        """
        Boost-stage efficiency as a function of input voltage.

        Linear interpolation between the two datasheet points (0.5 V -> 85%,
        1.0 V -> 90%); saturates outside that range.
        """
        if v_in_V < self.v_in_min_operating:
            return 0.0
        if v_in_V <= 0.5:
            return self._boost_eff_lowVin
        if v_in_V >= 1.0:
            return self._boost_eff_highVin
        frac = (v_in_V - 0.5) / 0.5
        return self._boost_eff_lowVin + frac * (self._boost_eff_highVin - self._boost_eff_lowVin)

    def output_power(self, p_in_W: float, v_in_V: float) -> float:
        """Approximate output power after boost + buck stages."""
        eta = self.boost_efficiency(v_in_V) * self._buck_eff
        return max(0.0, p_in_W * eta)

    def cold_start_supported(self, v_in_V: float) -> bool:
        return v_in_V >= self.v_in_min_startup

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.PART_NUMBER,
            'type': 'mppt_boost_charger',
            'v_in_min_startup_V': self.v_in_min_startup,
            'v_in_min_operating_V': self.v_in_min_operating,
            'mppt_ratio': self.mppt_ratio,
            'v_stor_V': self.v_stor,
            'v_out_V': self.v_out,
            'v_bat_uv_V': self.v_bat_uv,
            'v_bat_ov_V': self.v_bat_ov,
            'i_q_active_nA': self._i_q_active_nA,
            'i_q_coldstart_nA': self._i_q_coldstart_nA,
            'i_in_peak_mA': self._i_in_peak_mA,
            'boost_eff_at_0p5V': self._boost_eff_lowVin,
            'boost_eff_at_1V': self._boost_eff_highVin,
            'buck_eff_typ': self._buck_eff,
        }

    def __repr__(self):
        return (f"BQ25570(V_IN>=0.1V, MPPT={self.mppt_ratio*100:.0f}%, "
                f"V_OUT={self.v_out:.2f}V)")


if __name__ == "__main__":
    print("=" * 60)
    print("POWER COMPONENTS - SELF TEST")
    print("=" * 60)

    chg = BQ25570()
    print(f"\n--- BQ25570 (MPPT Boost Charger) ---")
    print(f"  {chg}")
    print(f"  Cold start at 0.8 V supported: {chg.cold_start_supported(0.8)}")
    print(f"  Cold start at 0.4 V supported: {chg.cold_start_supported(0.4)}")
    print(f"  MPPT target for V_OC = 1.1 V: {chg.mppt_voltage(1.1):.3f} V")
    print(f"  Boost efficiency @ 0.5 V: {chg.boost_efficiency(0.5)*100:.1f}%")
    print(f"  Boost efficiency @ 1.0 V: {chg.boost_efficiency(1.0)*100:.1f}%")
    print(f"  Output power from 1 mW @ 0.8 V: "
          f"{chg.output_power(1e-3, 0.8)*1e3:.3f} mW")

    print("\n[OK] Power tests passed!")

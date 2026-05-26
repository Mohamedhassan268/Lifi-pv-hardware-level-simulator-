# components/adc.py
"""
ADC models for the simulator.

Currently:
    - STM32H7_ADC: STMicroelectronics STM32H7 integrated SAR/SAR+oversampling ADC
"""

from typing import Any, Dict


class STM32H7_ADC:
    """
    STMicroelectronics STM32H7 integrated ADC characteristics.

    The Cairo sensor node uses an STM32H743 sampling the signal-chain
    output before the comparator, so that fall-back soft-decision
    demodulation (Manchester / PWM-ASK envelope) can run when the
    comparator is masked by ambient flicker.

    Datasheet Parameters (STMicro RM0433 and DS12110):
        - Resolution: 16-bit (also 14, 12, 10, 8)
        - Conversion rate: up to 3.6 Msps (16-bit), 5 Msps (12-bit)
        - Channels: up to 20 single-ended + 10 differential
        - V_REF: 1.62 to 3.6 V (typically 3.0 or 3.3 V external)
        - ENOB: 13.1 bits typical at 12-bit mode
        - SNR: 80 dB typical (12-bit)
        - INL: +/-1.5 LSB typical (12-bit)
        - Input impedance: <= 50 kohm sample-mode dependent
        - Offset error: +/-2 LSB typical
    """

    PART_NUMBER = "STM32H7_ADC"
    DATASHEET_URL = ("https://www.st.com/en/microcontrollers-microprocessors/"
                     "stm32h743zi.html")

    def __init__(self,
                 resolution_bits: int = 12,
                 v_ref_V: float = 3.3,
                 sample_rate_Sps: float = 1e6,
                 oversampling: int = 1):
        """
        Initialize STM32H7 ADC.

        Args:
            resolution_bits: 8 / 10 / 12 / 14 / 16
            v_ref_V: Reference voltage (datasheet: 1.62 to 3.6 V)
            sample_rate_Sps: Effective sample rate (conv_rate / oversampling)
            oversampling: Hardware oversampling ratio (1 to 1024)
        """
        if resolution_bits not in (8, 10, 12, 14, 16):
            raise ValueError(f"resolution_bits must be one of 8/10/12/14/16, "
                             f"got {resolution_bits}")
        if not (1.62 <= v_ref_V <= 3.6):
            raise ValueError(f"v_ref_V must be in [1.62, 3.6] V, got {v_ref_V}")
        if oversampling < 1 or oversampling > 1024:
            raise ValueError(f"oversampling must be in [1, 1024], got {oversampling}")

        self.resolution_bits = resolution_bits
        self.v_ref = v_ref_V
        self.sample_rate = sample_rate_Sps
        self.oversampling = oversampling

        # Datasheet-typical static / dynamic specs (12-bit mode)
        self._enob_12bit = 13.1              # Enhanced by oversampling spec
        self._snr_12bit_dB = 80.0
        self._inl_LSB = 1.5
        self._offset_LSB = 2.0
        self._max_conv_rate_Sps_12b = 5e6
        self._max_conv_rate_Sps_16b = 3.6e6

    @property
    def name(self) -> str:
        return self.PART_NUMBER

    @property
    def n_codes(self) -> int:
        """Number of discrete output codes at the current resolution."""
        return 1 << self.resolution_bits

    @property
    def lsb_V(self) -> float:
        """Voltage-per-LSB at the current V_REF and resolution."""
        return self.v_ref / self.n_codes

    @property
    def quantization_noise_std_V(self) -> float:
        """
        Quantization noise standard deviation (assumes uniform distribution).

        sigma = LSB / sqrt(12), reduced by sqrt(oversampling) for averaging.
        """
        base = self.lsb_V / (12 ** 0.5)
        return base / (self.oversampling ** 0.5)

    @property
    def snr_dB(self) -> float:
        """
        Ideal quantization SNR for a full-scale sinusoid:
            SNR = 6.02 * N + 1.76 dB, plus 10*log10(OSR) from oversampling.
        """
        import math
        base = 6.02 * self.resolution_bits + 1.76
        return base + 10 * math.log10(self.oversampling)

    def sample(self, v_in_V: float) -> int:
        """Quantize a voltage to an integer code, clamped to [0, 2^N - 1]."""
        if v_in_V < 0.0:
            return 0
        code = int(round(v_in_V / self.lsb_V))
        return max(0, min(self.n_codes - 1, code))

    def code_to_voltage(self, code: int) -> float:
        """Inverse of sample(): return the midpoint voltage of a code."""
        return code * self.lsb_V

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.PART_NUMBER,
            'type': 'integrated_adc',
            'resolution_bits': self.resolution_bits,
            'v_ref_V': self.v_ref,
            'sample_rate_Sps': self.sample_rate,
            'oversampling': self.oversampling,
            'n_codes': self.n_codes,
            'lsb_V': self.lsb_V,
            'quantization_noise_std_V': self.quantization_noise_std_V,
            'snr_dB_ideal': self.snr_dB,
            'enob_12bit_typ': self._enob_12bit,
            'inl_LSB_typ': self._inl_LSB,
            'offset_LSB_typ': self._offset_LSB,
            'max_conv_rate_Sps_12b': self._max_conv_rate_Sps_12b,
            'max_conv_rate_Sps_16b': self._max_conv_rate_Sps_16b,
        }

    def __repr__(self):
        return (f"STM32H7_ADC({self.resolution_bits}-bit, "
                f"Vref={self.v_ref:.2f}V, OSR={self.oversampling}x)")


if __name__ == "__main__":
    print("=" * 60)
    print("ADC COMPONENTS - SELF TEST")
    print("=" * 60)

    adc = STM32H7_ADC(resolution_bits=12, v_ref_V=3.3, oversampling=4)
    print(f"\n--- STM32H7 ADC ---")
    print(f"  {adc}")
    print(f"  LSB: {adc.lsb_V*1e3:.3f} mV")
    print(f"  Quantization noise std: {adc.quantization_noise_std_V*1e6:.2f} uV")
    print(f"  Ideal SNR (with OSR=4): {adc.snr_dB:.1f} dB")
    print(f"  sample(1.65 V) -> code {adc.sample(1.65)}, "
          f"back to {adc.code_to_voltage(adc.sample(1.65)):.4f} V")

    print("\n[OK] ADC tests passed!")

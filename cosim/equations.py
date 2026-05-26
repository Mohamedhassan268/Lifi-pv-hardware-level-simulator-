# cosim/equations.py
"""
Equation Registry — first-class metadata for every physical equation used
by the pipeline. Each entry carries:

  - latex:        KaTeX-renderable LaTeX source
  - plain:        ASCII fallback for terminals / hover tooltips
  - symbols:      per-symbol meaning + units + how to compute the live value
  - references:   paper / datasheet citations
  - compute_inputs: lambda(cfg) -> {symbol: value} returning the *current*
                  numeric value of each symbol for the active SystemConfig

The frontend ScienceCard component renders the LaTeX next to the live symbol
table — proving to the user that the same equations they see in the UI are
the ones the engine is actually evaluating. No re-derivation: the resolvers
delegate to the existing physics modules (channel.py, noise.py,
python_engine.PVReceiver).

Probe IDs in `cosim.probes` reference equation keys here via `equation_key`,
so clicking a probe's "Equation" tab in the UI lands on the right entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SymbolSpec:
    """One symbol within an equation."""
    symbol: str          # LaTeX-form symbol, e.g. "m", "A_{rx}", "\\sigma^2_{shot}"
    name: str            # Human name, e.g. "Lambertian order"
    units: str           # SI units string ("m", "A", "W", "1", "rad", ...)
    source: str = ""     # Where the value comes from (datasheet, config field, derived)


@dataclass(frozen=True)
class EquationSpec:
    """Specification for one equation in the registry."""
    key: str
    title: str
    latex: str
    plain: str
    symbols: List[SymbolSpec]
    references: List[str]
    # Resolver returns {symbol_key_in_latex: numeric value}. Symbol keys here
    # match SymbolSpec.symbol so the frontend can join them in a table.
    compute_inputs: Callable[[Any], Dict[str, float]]


# =============================================================================
# RESOLVERS — call into the same physics modules the pipeline uses
# =============================================================================

def _channel(cfg):
    from cosim.channel import OpticalChannel
    return OpticalChannel.from_config(cfg)


def _noise_model(cfg):
    from cosim.noise import NoiseModel
    return NoiseModel.from_config(cfg)


def _bandwidth_hz(cfg) -> float:
    return float(getattr(cfg, "data_rate_bps", 1.0)) / 2.0


def _pv_responsivity_at_T(cfg) -> float:
    R_ref = float(cfg.sc_responsivity)
    T_ref = float(getattr(cfg, "sc_responsivity_ref_T_K", 300.0))
    tempco = float(getattr(cfg, "sc_responsivity_tempco_per_K", 4e-4))
    T = float(cfg.temperature_K)
    return R_ref * (1.0 + tempco * (T - T_ref))


# =============================================================================
# REGISTRY
# =============================================================================

EQUATIONS: Dict[str, EquationSpec] = {

    # ---- Channel: Lambertian LOS gain ----
    "channel.los_gain": EquationSpec(
        key="channel.los_gain",
        title="Lambertian line-of-sight channel gain",
        latex=r"H_{\mathrm{LOS}}(0) = \frac{(m+1)\,A_{rx}}{2\pi\,d^{2}}\,\cos^{m}(\varphi)\,\cos(\psi)",
        plain="H_LOS(0) = ((m+1)*A_rx) / (2*pi*d^2) * cos^m(phi) * cos(psi)",
        symbols=[
            SymbolSpec("m", "Lambertian order", "1", "= -ln(2)/ln(cos(alpha_half))"),
            SymbolSpec("A_{rx}", "Receiver area", "m^2", "config: sc_area_cm2 * 1e-4"),
            SymbolSpec("d", "TX-RX distance", "m", "config: distance_m"),
            SymbolSpec(r"\varphi", "TX emission angle", "rad", "config: tx_angle_deg"),
            SymbolSpec(r"\psi", "RX incidence angle", "rad", "config: rx_tilt_deg"),
            SymbolSpec("H_{\\mathrm{LOS}}(0)", "LOS channel gain", "1", "derived"),
        ],
        references=["Kahn & Barry, Proc. IEEE 1997", "Komine & Nakagawa, IEEE TCE 2004"],
        compute_inputs=lambda cfg: {
            "m": _channel(cfg).m_lambertian,
            "A_{rx}": float(cfg.sc_area_cm2) * 1e-4,
            "d": float(cfg.distance_m),
            r"\varphi": float(np.radians(cfg.tx_angle_deg)),
            r"\psi": float(np.radians(cfg.rx_tilt_deg)),
            "H_{\\mathrm{LOS}}(0)": _channel(cfg).los_gain(),
        },
    ),

    "channel.diffuse_gain": EquationSpec(
        key="channel.diffuse_gain",
        title="First-order diffuse (multipath) gain",
        latex=r"H_{\mathrm{diff}} \approx \frac{\rho\,(m+1)\,A_{rx}}{\pi\,A_{\mathrm{room}}}",
        plain="H_diff = rho * (m+1) * A_rx / (pi * A_room)",
        symbols=[
            SymbolSpec(r"\rho", "Wall reflectivity", "1", "config: wall_reflectivity"),
            SymbolSpec("m", "Lambertian order", "1", "derived from led_half_angle_deg"),
            SymbolSpec("A_{rx}", "Receiver area", "m^2", "config: sc_area_cm2 * 1e-4"),
            SymbolSpec("A_{\\mathrm{room}}", "Room interior surface area", "m^2",
                       "= 2(L+W)H + LW (walls + ceiling)"),
            SymbolSpec("H_{\\mathrm{diff}}", "Diffuse gain", "1", "derived"),
        ],
        references=["Komine & Nakagawa, IEEE TCE 2004"],
        compute_inputs=lambda cfg: {
            r"\rho": float(cfg.wall_reflectivity),
            "m": _channel(cfg).m_lambertian,
            "A_{rx}": float(cfg.sc_area_cm2) * 1e-4,
            "A_{\\mathrm{room}}": (
                2 * (float(cfg.room_length_m) + float(cfg.room_width_m)) * float(cfg.room_height_m)
                + float(cfg.room_length_m) * float(cfg.room_width_m)
            ),
            "H_{\\mathrm{diff}}": _channel(cfg).diffuse_gain(),
        },
    ),

    "channel.beer_lambert": EquationSpec(
        key="channel.beer_lambert",
        title="Beer-Lambert atmospheric attenuation",
        latex=r"\tau_{\mathrm{atm}} = e^{-\alpha\,d}, \quad \alpha = \alpha_{0} + 4\,(\mathrm{RH}-0.3)^{1.5}",
        plain="tau = exp(-alpha*d) with alpha = alpha_base + 4*(RH-0.3)^1.5",
        symbols=[
            SymbolSpec(r"\alpha", "Attenuation coefficient", "1/m", "humidity-driven"),
            SymbolSpec("d", "Path length", "m", "config: distance_m"),
            SymbolSpec(r"\mathrm{RH}", "Relative humidity", "1", "config: humidity_rh"),
            SymbolSpec(r"\tau_{\mathrm{atm}}", "Transmission factor", "1", "derived"),
        ],
        references=["Correa et al. 2025"],
        compute_inputs=lambda cfg: {
            r"\alpha": _channel(cfg)._alpha_attenuation,
            "d": float(cfg.distance_m),
            r"\mathrm{RH}": float(cfg.humidity_rh) if cfg.humidity_rh is not None else 0.0,
            r"\tau_{\mathrm{atm}}": _channel(cfg).beer_lambert_factor(),
        },
    ),

    "channel.total_gain": EquationSpec(
        key="channel.total_gain",
        title="Total optical channel gain",
        latex=r"H_{\mathrm{tot}} = (H_{\mathrm{LOS}} + H_{\mathrm{diff}})\,\tau_{\mathrm{atm}}",
        plain="H_tot = (H_LOS + H_diff) * tau_atm",
        symbols=[
            SymbolSpec("H_{\\mathrm{LOS}}", "LOS gain", "1", "see channel.los_gain"),
            SymbolSpec("H_{\\mathrm{diff}}", "Diffuse gain", "1", "see channel.diffuse_gain"),
            SymbolSpec(r"\tau_{\mathrm{atm}}", "Atm. transmission", "1", "see channel.beer_lambert"),
            SymbolSpec("H_{\\mathrm{tot}}", "Total channel gain", "1", "derived"),
        ],
        references=[],
        compute_inputs=lambda cfg: {
            "H_{\\mathrm{LOS}}": _channel(cfg).los_gain(),
            "H_{\\mathrm{diff}}": _channel(cfg).diffuse_gain(),
            r"\tau_{\mathrm{atm}}": _channel(cfg).beer_lambert_factor(),
            "H_{\\mathrm{tot}}": _channel(cfg).channel_gain(),
        },
    ),

    # ---- PV responsivity ----
    "pv.responsivity": EquationSpec(
        key="pv.responsivity",
        title="Temperature-compensated PV responsivity",
        latex=r"R(T) = R_{\mathrm{ref}}\,\bigl[1 + \kappa_{R}(T - T_{\mathrm{ref}})\bigr],\quad I_{ph}=R(T)\,P_{rx}",
        plain="R(T) = R_ref * (1 + kappa_R*(T - T_ref)); I_ph = R(T)*P_rx",
        symbols=[
            SymbolSpec("R_{\\mathrm{ref}}", "Responsivity at T_ref", "A/W", "config: sc_responsivity"),
            SymbolSpec(r"\kappa_{R}", "Tempco", "1/K", "config: sc_responsivity_tempco_per_K"),
            SymbolSpec("T", "Operating temperature", "K", "config: temperature_K"),
            SymbolSpec("T_{\\mathrm{ref}}", "Reference temperature", "K",
                       "config: sc_responsivity_ref_T_K"),
            SymbolSpec("R(T)", "Effective responsivity", "A/W", "derived"),
        ],
        references=["PV cell datasheet"],
        compute_inputs=lambda cfg: {
            "R_{\\mathrm{ref}}": float(cfg.sc_responsivity),
            r"\kappa_{R}": float(getattr(cfg, "sc_responsivity_tempco_per_K", 4e-4)),
            "T": float(cfg.temperature_K),
            "T_{\\mathrm{ref}}": float(getattr(cfg, "sc_responsivity_ref_T_K", 300.0)),
            "R(T)": _pv_responsivity_at_T(cfg),
        },
    ),

    # ---- Noise: 6 sources ----
    "noise.shot": EquationSpec(
        key="noise.shot",
        title="Shot noise (Poisson statistics of photocurrent)",
        latex=r"\sigma^{2}_{\mathrm{shot}} = 2q\,(I_{ph}+I_{\mathrm{dark}})\,B_{n}",
        plain="sigma^2_shot = 2*q*(I_ph + I_dark)*Bn",
        symbols=[
            SymbolSpec("q", "Electron charge", "C", "constant: 1.602176634e-19"),
            SymbolSpec("I_{ph}", "Photocurrent", "A", "derived"),
            SymbolSpec("I_{\\mathrm{dark}}", "Dark current", "A", "scaled by temperature"),
            SymbolSpec("B_{n}", "Noise bandwidth", "Hz", "= data_rate_bps / 2"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{shot}}", "Shot noise variance", "A^2", "derived"),
        ],
        references=["Kahn & Barry, Proc. IEEE 1997"],
        compute_inputs=lambda cfg: {
            "q": 1.602176634e-19,
            "I_{ph}": float(cfg.photocurrent_A()),
            "I_{\\mathrm{dark}}": _noise_model(cfg).dark_current_at_T(),
            "B_{n}": _bandwidth_hz(cfg),
            r"\sigma^{2}_{\mathrm{shot}}": _noise_model(cfg).shot_noise_variance(
                float(cfg.photocurrent_A()), _bandwidth_hz(cfg),
                _noise_model(cfg).dark_current_at_T(),
            ),
        },
    ),

    "noise.thermal": EquationSpec(
        key="noise.thermal",
        title="Thermal (Johnson-Nyquist) noise across sense resistor",
        latex=r"\sigma^{2}_{\mathrm{th}} = \frac{4k_{B}T\,B_{n}}{R_{\mathrm{load}}}",
        plain="sigma^2_th = 4*kB*T*Bn / R_load",
        symbols=[
            SymbolSpec("k_{B}", "Boltzmann constant", "J/K", "constant: 1.380649e-23"),
            SymbolSpec("T", "Temperature", "K", "config: temperature_K"),
            SymbolSpec("B_{n}", "Noise bandwidth", "Hz", "= data_rate_bps / 2"),
            SymbolSpec("R_{\\mathrm{load}}", "Sense resistance", "Ohm", "config: r_sense_ohm"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{th}}", "Thermal noise variance", "A^2", "derived"),
        ],
        references=["Nyquist 1928", "Johnson 1928"],
        compute_inputs=lambda cfg: {
            "k_{B}": 1.380649e-23,
            "T": float(cfg.temperature_K),
            "B_{n}": _bandwidth_hz(cfg),
            "R_{\\mathrm{load}}": float(cfg.r_sense_ohm),
            r"\sigma^{2}_{\mathrm{th}}": _noise_model(cfg).thermal_noise_variance(_bandwidth_hz(cfg)),
        },
    ),

    "noise.ambient": EquationSpec(
        key="noise.ambient",
        title="Ambient-light induced shot noise",
        latex=r"\sigma^{2}_{\mathrm{amb}} = 2q\,I_{\mathrm{amb}}\,B_{n},\quad I_{\mathrm{amb}} = R\cdot(\mathrm{lux}\cdot 1.46\!\times\!10^{-6}\cdot A_{rx})",
        plain="sigma^2_amb = 2q * I_amb * Bn; I_amb = R * (lux * 1.46e-6 * A_rx)",
        symbols=[
            SymbolSpec("\\mathrm{lux}", "Ambient illuminance", "lx",
                       "config: ambient_illuminance_lux"),
            SymbolSpec("A_{rx}", "Receiver area", "cm^2", "config: sc_area_cm2"),
            SymbolSpec("R", "Responsivity", "A/W", "config: sc_responsivity"),
            SymbolSpec("I_{\\mathrm{amb}}", "Ambient photocurrent", "A", "derived"),
            SymbolSpec("B_{n}", "Noise bandwidth", "Hz", "= data_rate_bps / 2"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{amb}}", "Ambient noise variance", "A^2", "derived"),
        ],
        references=["Kahn & Barry, Proc. IEEE 1997"],
        compute_inputs=lambda cfg: {
            "\\mathrm{lux}": float(cfg.ambient_illuminance_lux),
            "A_{rx}": float(cfg.sc_area_cm2),
            "R": float(cfg.sc_responsivity),
            "I_{\\mathrm{amb}}": (
                float(cfg.sc_responsivity) * float(cfg.ambient_illuminance_lux)
                * 1.46e-6 * float(cfg.sc_area_cm2)
            ),
            "B_{n}": _bandwidth_hz(cfg),
            r"\sigma^{2}_{\mathrm{amb}}": _noise_model(cfg).ambient_noise_variance(_bandwidth_hz(cfg)),
        },
    ),

    "noise.amplifier": EquationSpec(
        key="noise.amplifier",
        title="Input-referred amplifier noise (voltage + current)",
        latex=r"\sigma^{2}_{\mathrm{amp}} = \left(\frac{e_{n}^{2}}{R_{\mathrm{load}}^{2}} + i_{n}^{2}\right) B_{n}",
        plain="sigma^2_amp = (e_n^2/R_load^2 + i_n^2) * Bn",
        symbols=[
            SymbolSpec("e_{n}", "Voltage noise density", "V/sqrt(Hz)",
                       "config: ina_noise_nV_rtHz * 1e-9"),
            SymbolSpec("i_{n}", "Current noise density", "A/sqrt(Hz)",
                       "config: ina_noise_current_pA_rtHz * 1e-12"),
            SymbolSpec("R_{\\mathrm{load}}", "Sense resistance", "Ohm", "config: r_sense_ohm"),
            SymbolSpec("B_{n}", "Noise bandwidth", "Hz", "= data_rate_bps / 2"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{amp}}", "Amplifier noise variance", "A^2", "derived"),
        ],
        references=["INA322 datasheet (SBOS163)"],
        compute_inputs=lambda cfg: {
            "e_{n}": float(cfg.ina_noise_nV_rtHz) * 1e-9,
            "i_{n}": float(cfg.ina_noise_current_pA_rtHz) * 1e-12,
            "R_{\\mathrm{load}}": float(cfg.r_sense_ohm),
            "B_{n}": _bandwidth_hz(cfg),
            r"\sigma^{2}_{\mathrm{amp}}": _noise_model(cfg).amplifier_noise_variance(_bandwidth_hz(cfg)),
        },
    ),

    "noise.adc": EquationSpec(
        key="noise.adc",
        title="ADC quantization noise (input-referred)",
        latex=r"\sigma^{2}_{\mathrm{adc}} = \frac{V_{\mathrm{LSB}}^{2}/12}{(R_{\mathrm{load}}\,G_{\mathrm{amp}})^{2}},\quad V_{\mathrm{LSB}} = \frac{V_{\mathrm{ref}}}{2^{N}}",
        plain="sigma^2_adc = (V_LSB^2/12) / (R_load*G_amp)^2; V_LSB = V_ref/2^N",
        symbols=[
            SymbolSpec("N", "ADC bits", "1", "config: adc_bits"),
            SymbolSpec("V_{\\mathrm{ref}}", "ADC reference", "V", "config: adc_vref"),
            SymbolSpec("V_{\\mathrm{LSB}}", "LSB voltage", "V", "= V_ref / 2^N"),
            SymbolSpec("G_{\\mathrm{amp}}", "Amplifier gain (linear)", "1",
                       "= 10^(ina_gain_dB/20)"),
            SymbolSpec("R_{\\mathrm{load}}", "Sense resistance", "Ohm", "config: r_sense_ohm"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{adc}}", "ADC noise variance (current)", "A^2", "derived"),
        ],
        references=["Widrow 1956"],
        compute_inputs=lambda cfg: {
            "N": int(cfg.adc_bits),
            "V_{\\mathrm{ref}}": float(cfg.adc_vref),
            "V_{\\mathrm{LSB}}": float(cfg.adc_vref) / (2 ** int(cfg.adc_bits)) if cfg.adc_bits else 0.0,
            "G_{\\mathrm{amp}}": 10 ** (float(cfg.ina_gain_dB) / 20.0),
            "R_{\\mathrm{load}}": float(cfg.r_sense_ohm),
            r"\sigma^{2}_{\mathrm{adc}}": _noise_model(cfg).adc_quantization_variance(),
        },
    ),

    "noise.processing": EquationSpec(
        key="noise.processing",
        title="Comparator offset and timing-jitter noise",
        latex=r"\sigma^{2}_{\mathrm{proc}} = \sigma^{2}_{\mathrm{offset}} + \sigma^{2}_{\mathrm{jitter}}",
        plain="sigma^2_proc = sigma^2_offset + sigma^2_jitter",
        symbols=[
            SymbolSpec(r"\sigma_{\mathrm{offset}}", "Comparator offset", "V",
                       "config: comparator_offset_mV * 1e-3"),
            SymbolSpec(r"\sigma_{\mathrm{jitter}}", "Timing jitter (1-sigma)", "s",
                       "config: comparator_jitter_ns * 1e-9"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{proc}}", "Processing variance (current)", "A^2", "derived"),
        ],
        references=["TLV7011 datasheet (SBOS819)"],
        compute_inputs=lambda cfg: {
            r"\sigma_{\mathrm{offset}}": float(cfg.comparator_offset_mV) * 1e-3,
            r"\sigma_{\mathrm{jitter}}": float(cfg.comparator_jitter_ns) * 1e-9,
            r"\sigma^{2}_{\mathrm{proc}}": _noise_model(cfg).processing_noise_variance(),
        },
    ),

    # ---- Non-AWGN real-world sources ----
    "noise.mains_flicker": EquationSpec(
        key="noise.mains_flicker",
        title="AC mains flicker (deterministic interferer)",
        latex=r"i_{\mathrm{fl}}(t) = d_{\mathrm{fl}}\,I_{\mathrm{amb}}\sum_{k=1}^{3} a_{k}\cos(2\pi k f_{\mathrm{mains}} t + \varphi_{k})",
        plain="i_fl(t) = d * I_amb * sum_k a_k * cos(2*pi*k*f_mains*t + phi_k); a1=1.0, a2=0.30, a3=0.10",
        symbols=[
            SymbolSpec("f_{\\mathrm{mains}}", "Flicker fundamental", "Hz",
                       "config: mains_flicker_freq_hz — 100 (50 Hz mains) or 120 (60 Hz)"),
            SymbolSpec("d_{\\mathrm{fl}}", "Flicker depth", "1",
                       "config: mains_flicker_depth — fraction of I_amb at the fundamental"),
            SymbolSpec("I_{\\mathrm{amb}}", "Ambient photocurrent", "A",
                       "= R · (lux · 1.46e-6 · A_rx)"),
            SymbolSpec("a_1, a_2, a_3", "Harmonic weights", "1",
                       "1.00 / 0.30 / 0.10 — typical fluorescent/LED ballast spectrum"),
        ],
        references=["Wilkins, Veitch, Lehman, IEEE TIA 2010"],
        compute_inputs=lambda cfg: {
            "f_{\\mathrm{mains}}": float(getattr(cfg, 'mains_flicker_freq_hz', 100.0)),
            "d_{\\mathrm{fl}}": float(getattr(cfg, 'mains_flicker_depth', 0.05)),
            "I_{\\mathrm{amb}}": (
                float(cfg.sc_responsivity) * float(cfg.ambient_illuminance_lux)
                * 1.46e-6 * float(cfg.sc_area_cm2)
            ),
            "a_1, a_2, a_3": 1.00,
        },
    ),

    "noise.pink": EquationSpec(
        key="noise.pink",
        title="Amplifier 1/f (pink) noise",
        latex=r"e_{n}(f) = e_{n,\mathrm{wb}}\sqrt{1+\frac{f_{c}}{f}},\quad \sigma^{2}_{\mathrm{pink}} = \frac{e_{n,\mathrm{wb}}^{2}\,f_{c}\,\ln(B_{n}/f_{\mathrm{low}})}{R_{\mathrm{load}}^{2}}",
        plain="sigma^2_pink = e_n^2 * f_c * ln(Bn / f_low) / R_load^2  (input-referred to current)",
        symbols=[
            SymbolSpec("e_{n,\\mathrm{wb}}", "Wideband voltage-noise density", "V/sqrt(Hz)",
                       "config: ina_noise_nV_rtHz * 1e-9"),
            SymbolSpec("f_{c}", "Flicker corner frequency", "Hz",
                       "config: amp_flicker_corner_hz — TL072 ~100, INA322 ~10"),
            SymbolSpec("B_{n}", "Noise bandwidth (Nyquist proxy)", "Hz", "= 0.5 / dt"),
            SymbolSpec("f_{\\mathrm{low}}", "Low-frequency floor", "Hz",
                       "= 1 / T_total — set by run length"),
            SymbolSpec("R_{\\mathrm{load}}", "Sense resistance", "Ohm", "config: r_sense_ohm"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{pink}}", "Pink-noise variance (current)", "A^2", "derived"),
        ],
        references=["OpAmp datasheet flicker-corner spec", "Voss & Clarke, J. Acoust. Soc. Am. 1975"],
        compute_inputs=lambda cfg: {
            "e_{n,\\mathrm{wb}}": float(cfg.ina_noise_nV_rtHz) * 1e-9,
            "f_{c}": float(getattr(cfg, 'amp_flicker_corner_hz', 100.0)),
            "B_{n}": 0.5 * float(cfg.data_rate_bps) * 100.0,   # ≈ Nyquist for 100 samples/bit
            "f_{\\mathrm{low}}": 1.0 / max(float(cfg.t_stop_s), 1e-6),
            "R_{\\mathrm{load}}": float(cfg.r_sense_ohm),
        },
    ),

    # ---- Signal-to-noise quantities (see cosim/snr.py for the science) ----
    "snr.link_budget": EquationSpec(
        key="snr.link_budget",
        title="Link-budget SNR (closed-form, all modelled sources)",
        latex=r"\mathrm{SNR}_{\mathrm{lb}} = \frac{\mathrm{Var}\!\bigl[I_{ph}\bigr]}{\sigma^{2}_{\mathrm{AWGN}} + \sigma^{2}_{\mathrm{flicker}} + \sigma^{2}_{\mathrm{pink}}}",
        plain="SNR_lb = var(I_ph) / (sigma_AWGN^2 + sigma_flicker^2 + sigma_pink^2)",
        symbols=[
            SymbolSpec(r"\mathrm{Var}[I_{ph}]", "Signal photocurrent variance", "A^2",
                       "from the clean modulated I_ph(t)"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{AWGN}}", "Sum of 6 AWGN variances",
                       "A^2", "shot + thermal + ambient + amp + adc + processing"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{flicker}}", "Mains-flicker variance",
                       "A^2", "= (depth · I_amb)² · Σ a_k²/2 — closed-form for sum-of-cosines"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{pink}}", "Amp 1/f integrated variance",
                       "A^2", "= e_n² · f_c · ln(B/f_low) / R_load²"),
            SymbolSpec(r"\mathrm{SNR}_{\mathrm{lb}}", "Link-budget SNR", "ratio",
                       "10·log10 → dB; closed-form, does NOT see chain filters (notch/BPF)"),
        ],
        references=[
            "Sklar, *Digital Communications*, 2nd ed., §1.5",
            "Wilkins, Veitch, Lehman, IEEE TIA 2010 — flicker variance model",
        ],
        compute_inputs=lambda cfg: {},  # values come from the run, not closed-form
    ),

    "snr.measured": EquationSpec(
        key="snr.measured",
        title="Measured SNR at the demodulator input (two-run technique)",
        latex=r"\mathrm{SNR}_{\mathrm{meas}}\Big|_{\mathrm{node}} = \frac{\mathrm{Var}\!\bigl[V_{\mathrm{node}}^{\,\text{signal-only}}\bigr]}{\mathrm{Var}\!\bigl[V_{\mathrm{node}}^{\,\text{noise-only}}\bigr]}",
        plain="SNR_meas = var(reference) / var(full - reference)  — two-run technique",
        symbols=[
            SymbolSpec(r"V_{\mathrm{node}}^{\,\text{signal-only}}",
                       "Reference run with all noise off (same seed)", "V", "captured probe"),
            SymbolSpec(r"V_{\mathrm{node}}^{\,\text{noise-only}}",
                       "full − reference (the actual injected noise)", "V", "computed"),
            SymbolSpec(r"\mathrm{SNR}_{\mathrm{meas}}", "Measured SNR at the node",
                       "ratio", "10·log10 → dB; THIS is the SNR that physically drives BER"),
        ],
        references=[
            "Keysight, *VSA Vector Signal Analyzer*, application note 150-3",
            "GNU Radio block: digital.probe_signal_*",
        ],
        compute_inputs=lambda cfg: {},
    ),

    "snr.eb_n0": EquationSpec(
        key="snr.eb_n0",
        title="Eb/N0 against the AWGN noise floor",
        latex=r"\frac{E_{b}}{N_{0}} = \frac{\mathrm{Var}\!\bigl[I_{ph}\bigr]\,T_{b}\,B_{n}}{\sigma^{2}_{\mathrm{AWGN,total}}}",
        plain="Eb/N0 = (var(I_ph) * T_bit * Bn) / sigma_total^2   — valid only for AWGN",
        symbols=[
            SymbolSpec("T_{b}", "Bit period", "s", "= 1 / data_rate_bps"),
            SymbolSpec("B_{n}", "Noise bandwidth", "Hz", "= data_rate_bps / 2"),
            SymbolSpec(r"\mathrm{Var}[I_{ph}]", "Signal photocurrent variance", "A^2",
                       "from the clean modulated I_ph(t)"),
            SymbolSpec(r"\sigma^{2}_{\mathrm{AWGN,total}}", "AWGN variance only",
                       "A^2", "mains flicker + 1/f excluded — N0 isn't flat for them"),
            SymbolSpec("E_{b}/N_{0}", "Energy-per-bit / noise PSD", "ratio",
                       "Comparable to textbook BER curves; AWGN-floor only"),
        ],
        references=[
            "Sklar, *Digital Communications*, 2nd ed., §3.1",
            "Proakis & Salehi, *Digital Communications*, 5th ed., §4.3",
        ],
        compute_inputs=lambda cfg: {
            "T_{b}": 1.0 / max(float(cfg.data_rate_bps), 1.0),
            "B_{n}": _bandwidth_hz(cfg),
        },
    ),

    # ---- Modulation thresholds ----
    "modulation.ook": EquationSpec(
        key="modulation.ook",
        title="OOK modulation with bias and modulation depth",
        latex=r"P_{tx}(t) = P_{0}\bigl[1 + \mu\,(2b(t)-1)\bigr]",
        plain="P_tx(t) = P0 * (1 + mu*(2*b(t)-1))",
        symbols=[
            SymbolSpec("P_{0}", "DC LED power", "W", "config: led_radiated_power_mW * 1e-3"),
            SymbolSpec(r"\mu", "Modulation depth", "1", "config: modulation_depth"),
            SymbolSpec("b(t)", "Bit stream", "{0,1}", "PRBS or user-provided"),
        ],
        references=[],
        compute_inputs=lambda cfg: {
            "P_{0}": float(cfg.led_radiated_power_mW) * 1e-3,
            r"\mu": float(cfg.modulation_depth),
        },
    ),
}


def equation_keys() -> List[str]:
    """Stable, ordered list of all equation keys."""
    return list(EQUATIONS.keys())


def get_equation(key: str) -> Optional[EquationSpec]:
    """Return a spec by key, or None if unknown."""
    return EQUATIONS.get(key)


def resolve_for_config(key: str, cfg) -> Optional[Dict[str, Any]]:
    """
    Evaluate an equation's symbols for a given SystemConfig and return a
    JSON-serializable payload for the /api/equations endpoint.

    Returns None if the key is unknown. If a symbol resolver fails, that
    symbol's value is reported as null (so the UI can still render the LaTeX).
    """
    spec = EQUATIONS.get(key)
    if spec is None:
        return None

    try:
        values = spec.compute_inputs(cfg)
    except Exception:  # noqa: BLE001 - report partial result rather than 500
        values = {}

    symbols_out = []
    for sym in spec.symbols:
        raw = values.get(sym.symbol)
        try:
            value = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            value = None
        symbols_out.append({
            "symbol": sym.symbol,
            "name": sym.name,
            "units": sym.units,
            "source": sym.source,
            "value": value,
        })

    return {
        "key": spec.key,
        "title": spec.title,
        "latex": spec.latex,
        "plain": spec.plain,
        "symbols": symbols_out,
        "references": list(spec.references),
    }

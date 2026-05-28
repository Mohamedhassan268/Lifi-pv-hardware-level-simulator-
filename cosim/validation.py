# cosim/validation.py
"""
Parameter validation & guardrails for SystemConfig.

One authoritative validator that returns a structured list of issues.
All UIs (CLI, GUI, frontend) render from the same source.

Severity levels:
    ERROR   - non-physical / will crash. Refuse to simulate.
    WARNING - exceeds datasheet limits or link-budget red flag. Run + flag.
    INFO    - advisory (e.g. unusual but legal values).

Rule families:
    _rule_physical_sanity  - non-negative, range, enum membership (existing checks)
    _rule_nyquist          - sampling vs BPF / OFDM bandwidth
    _rule_datasheet        - configured values vs component datasheet limits
    _rule_link_budget      - received optical power vs noise floor
    _rule_statistical      - n_bits vs target_ber resolution
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List


class IssueLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    level: IssueLevel
    field: str           # dotted path, e.g. "tx.bias_current_A"
    message: str         # human-readable
    suggestion: str = "" # optional "try X" hint
    rule_id: str = ""    # stable ID, e.g. "tx.led_overcurrent"

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "field": self.field,
            "message": self.message,
            "suggestion": self.suggestion,
            "rule_id": self.rule_id,
        }


_VALID_MODULATIONS = frozenset({
    'OOK', 'OOK_Manchester', 'OFDM', 'BFSK', 'PWM_ASK',
})
_VALID_TOPOLOGIES = frozenset({
    'ina_bpf_comp', 'amp_slicer', 'direct', 'auto',
})


# -----------------------------------------------------------------------------
# Rule: physical sanity (refactored from system_config._post_init)
# -----------------------------------------------------------------------------

def _rule_physical_sanity(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    E = IssueLevel.ERROR

    if cfg.channel.distance_m <= 0:
        issues.append(ValidationIssue(
            E, "channel.distance_m",
            f"distance_m must be > 0, got {cfg.channel.distance_m}",
            rule_id="physical.distance_nonpositive",
        ))
    if cfg.sim.data_rate_bps <= 0:
        issues.append(ValidationIssue(
            E, "sim.data_rate_bps",
            f"data_rate_bps must be > 0, got {cfg.sim.data_rate_bps}",
            rule_id="physical.data_rate_nonpositive",
        ))
    if cfg.sim.modulation not in _VALID_MODULATIONS:
        issues.append(ValidationIssue(
            E, "sim.modulation",
            f"modulation must be one of {sorted(_VALID_MODULATIONS)}, "
            f"got '{cfg.sim.modulation}'",
            rule_id="physical.modulation_unknown",
        ))
    if cfg.rx.rx_topology not in _VALID_TOPOLOGIES:
        issues.append(ValidationIssue(
            E, "rx.rx_topology",
            f"rx_topology must be one of {sorted(_VALID_TOPOLOGIES)}, "
            f"got '{cfg.rx.rx_topology}'",
            rule_id="physical.topology_unknown",
        ))
    if cfg.rx.bpf_f_low_Hz >= cfg.rx.bpf_f_high_Hz and cfg.rx.bpf_f_low_Hz > 0:
        issues.append(ValidationIssue(
            E, "rx.bpf_f_low_Hz",
            f"bpf_f_low_Hz ({cfg.rx.bpf_f_low_Hz}) must be < "
            f"bpf_f_high_Hz ({cfg.rx.bpf_f_high_Hz})",
            rule_id="physical.bpf_inverted",
        ))
    if cfg.channel.fov_half_angle_deg <= 0 or cfg.channel.fov_half_angle_deg > 90:
        issues.append(ValidationIssue(
            E, "channel.fov_half_angle_deg",
            f"fov_half_angle_deg must be in (0, 90], got {cfg.channel.fov_half_angle_deg}",
            rule_id="physical.fov_out_of_range",
        ))
    if cfg.channel.n_reflections < 0:
        issues.append(ValidationIssue(
            E, "channel.n_reflections",
            f"n_reflections must be >= 0, got {cfg.channel.n_reflections}",
            rule_id="physical.n_reflections_negative",
        ))
    if cfg.channel.wall_reflectivity < 0 or cfg.channel.wall_reflectivity > 1:
        issues.append(ValidationIssue(
            E, "channel.wall_reflectivity",
            f"wall_reflectivity must be in [0, 1], got {cfg.channel.wall_reflectivity}",
            rule_id="physical.wall_reflectivity_out_of_range",
        ))
    if cfg.noise.ambient_illuminance_lux < 0:
        issues.append(ValidationIssue(
            E, "noise.ambient_illuminance_lux",
            f"ambient_illuminance_lux must be >= 0, got {cfg.noise.ambient_illuminance_lux}",
            rule_id="physical.ambient_negative",
        ))

    # Non-negative scalars on RX
    for field_name in ('sc_area_cm2', 'sc_responsivity', 'sc_cj_nF',
                       'r_sense_ohm', 'vcc_volts'):
        val = getattr(cfg.rx, field_name)
        if val < 0:
            issues.append(ValidationIssue(
                E, f"rx.{field_name}",
                f"{field_name} must be >= 0, got {val}",
                rule_id=f"physical.{field_name}_negative",
            ))

    if cfg.tx.led_radiated_power_mW < 0:
        issues.append(ValidationIssue(
            E, "tx.led_radiated_power_mW",
            f"led_radiated_power_mW must be >= 0, got {cfg.tx.led_radiated_power_mW}",
            rule_id="physical.led_power_negative",
        ))
    if cfg.tx.bias_current_A < 0:
        issues.append(ValidationIssue(
            E, "tx.bias_current_A",
            f"bias_current_A must be >= 0, got {cfg.tx.bias_current_A}",
            rule_id="physical.bias_current_negative",
        ))

    return issues


# -----------------------------------------------------------------------------
# Rule: Nyquist / sampling
# -----------------------------------------------------------------------------

def _rule_nyquist(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # Sim window vs requested bits. The SPICE engine uses t_stop_s for the
    # transient run; the Python BER engine sends n_bits independently. So
    # this is a WARNING (informative, not gating) - only relevant when the
    # SPICE engine is also expected to capture every bit.
    if cfg.sim.data_rate_bps > 0 and cfg.sim.t_stop_s > 0:
        max_bits = cfg.sim.t_stop_s * cfg.sim.data_rate_bps
        if cfg.sim.n_bits > max_bits + 1:  # +1 for rounding tolerance
            issues.append(ValidationIssue(
                IssueLevel.WARNING, "sim.t_stop_s",
                f"t_stop_s={cfg.sim.t_stop_s}s only holds ~{int(max_bits)} bits "
                f"at data_rate_bps={cfg.sim.data_rate_bps} (n_bits={cfg.sim.n_bits}). "
                f"SPICE waveform will be truncated; Python BER engine still sends all bits.",
                suggestion=f"For full SPICE coverage set t_stop_s >= "
                           f"{cfg.sim.n_bits / cfg.sim.data_rate_bps:.4g}s",
                rule_id="nyquist.sim_time_short",
            ))

    # Signal frequency vs BPF high cutoff. OOK signal first harmonic is ~data_rate/2.
    if (cfg.rx.bpf_f_high_Hz > 0
            and cfg.rx.bpf_stages > 0
            and cfg.sim.data_rate_bps > 0):
        signal_hz = cfg.sim.data_rate_bps / 2.0
        if signal_hz > cfg.rx.bpf_f_high_Hz:
            issues.append(ValidationIssue(
                IssueLevel.WARNING, "rx.bpf_f_high_Hz",
                f"signal fundamental ~{signal_hz:.1f} Hz exceeds "
                f"BPF high cutoff {cfg.rx.bpf_f_high_Hz:.1f} Hz "
                f"- signal will be attenuated by the filter",
                suggestion=f"Raise bpf_f_high_Hz above {signal_hz:.0f} Hz",
                rule_id="nyquist.signal_above_bpf",
            ))

    # OFDM: total occupied bandwidth vs BPF high cutoff
    if (cfg.sim.modulation == 'OFDM'
            and cfg.rx.bpf_f_high_Hz > 0
            and cfg.rx.bpf_stages > 0):
        n_used = getattr(cfg.modulation_config, 'ofdm_subcarriers_used', 0)
        spacing = getattr(cfg.modulation_config, 'ofdm_subcarrier_spacing_Hz', 0.0)
        occupied = n_used * spacing
        if occupied > 0 and occupied > cfg.rx.bpf_f_high_Hz:
            issues.append(ValidationIssue(
                IssueLevel.WARNING, "modulation_config.ofdm_subcarriers_used",
                f"OFDM occupied bandwidth ~{occupied:.0f} Hz exceeds "
                f"BPF high cutoff {cfg.rx.bpf_f_high_Hz:.0f} Hz",
                rule_id="nyquist.ofdm_above_bpf",
            ))

    # OFDM through a BPF cascade — phase 14 routes OFDM through the actual
    # RX chain, so a narrowband BPF cascade (designed for kHz OOK) will
    # crush the MHz-class OFDM signal. All current OFDM presets use
    # rx_topology='direct' so this is preventive.
    if cfg.sim.modulation == 'OFDM' and cfg.rx.bpf_stages > 0:
        issues.append(ValidationIssue(
            IssueLevel.WARNING, "rx.bpf_stages",
            f"OFDM with bpf_stages={cfg.rx.bpf_stages} routes a wideband "
            f"signal through a narrowband BPF cascade (designed for kHz "
            f"OOK). The chain equalizer can only undo so much attenuation "
            f"before subcarrier SNR collapses.",
            suggestion="Use rx_topology='direct' for OFDM presets, "
                       "or set bpf_stages=0.",
            rule_id="nyquist.ofdm_bpf_incompatible",
        ))

    return issues


# -----------------------------------------------------------------------------
# Rule: datasheet limits (queries COMPONENT_REGISTRY)
# -----------------------------------------------------------------------------

def _get_component(name: str):
    """Look up a component instance by part name, or return None."""
    if not name or name == 'N/A':
        return None
    try:
        from components import get_component  # local import to avoid cycle
        return get_component(name)
    except (KeyError, ImportError, TypeError):
        return None


def _rule_datasheet(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # --- LED forward current ---
    led = _get_component(cfg.tx.led_part)
    if led is not None and hasattr(led, 'max_drive_current_A'):
        try:
            i_max = float(led.max_drive_current_A)
        except (TypeError, ValueError):
            i_max = 0.0
        if i_max > 0 and cfg.tx.bias_current_A > i_max:
            issues.append(ValidationIssue(
                IssueLevel.WARNING, "tx.bias_current_A",
                f"bias_current_A={cfg.tx.bias_current_A:.3g}A exceeds "
                f"{cfg.tx.led_part} max drive current {i_max:.3g}A "
                f"- would damage the LED in hardware",
                suggestion=f"Set bias_current_A <= {i_max:.3g}",
                rule_id="datasheet.led_overcurrent",
            ))

    # --- Amplifier GBW vs signal frequency ---
    amp = _get_component(cfg.rx.ina_part)
    if (amp is not None
            and hasattr(amp, 'gain_bandwidth_product')
            and cfg.sim.data_rate_bps > 0):
        try:
            gbw = float(amp.gain_bandwidth_product)
        except (TypeError, ValueError):
            gbw = 0.0
        if gbw > 0:
            signal_hz = cfg.sim.data_rate_bps / 2.0
            # Closed-loop BW = GBW / gain_linear. Linear gain from dB.
            gain_db = max(cfg.rx.ina_gain_dB, 0.0)
            gain_linear = 10.0 ** (gain_db / 20.0) if gain_db > 0 else 1.0
            closed_loop_bw = gbw / gain_linear if gain_linear > 0 else gbw
            if signal_hz > closed_loop_bw:
                issues.append(ValidationIssue(
                    IssueLevel.WARNING, "sim.data_rate_bps",
                    f"signal frequency ~{signal_hz:.1f} Hz exceeds "
                    f"{cfg.rx.ina_part} closed-loop bandwidth "
                    f"~{closed_loop_bw:.1f} Hz (GBW={gbw:.0f}Hz, "
                    f"gain={gain_db:.1f}dB) - amp will roll off the signal",
                    rule_id="datasheet.amp_above_bandwidth",
                ))

    return issues


# -----------------------------------------------------------------------------
# Rule: link budget (optical power vs noise floor)
# -----------------------------------------------------------------------------

def _rule_link_budget(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # Conservative back-of-envelope: compare received irradiance to shot-noise
    # floor at the configured bandwidth. Avoid heavy imports for fast validation.
    if (cfg.tx.led_radiated_power_mW <= 0
            or cfg.channel.distance_m <= 0
            or cfg.rx.sc_responsivity <= 0
            or cfg.rx.sc_area_cm2 <= 0):
        return issues

    # Lambertian: I(0) = (m+1)/(2π) * P_total; on-axis irradiance at d:
    # E = I(0) / d^2.  m from half-angle.
    half_angle_rad = math.radians(max(cfg.tx.led_half_angle_deg, 0.001))
    cos_half = math.cos(half_angle_rad)
    if cos_half <= 0 or cos_half >= 1:
        return issues
    m = -math.log(2.0) / math.log(cos_half)
    p_total_W = cfg.tx.led_radiated_power_mW * 1e-3 * cfg.tx.lens_transmittance
    I0 = (m + 1.0) / (2.0 * math.pi) * p_total_W   # W/sr
    irradiance = I0 / (cfg.channel.distance_m ** 2)  # W/m^2

    area_m2 = cfg.rx.sc_area_cm2 * 1e-4
    p_rx_W = irradiance * area_m2
    i_signal_A = p_rx_W * cfg.rx.sc_responsivity

    # Shot-noise floor at BPF bandwidth (or data rate if no BPF).
    if cfg.rx.bpf_stages > 0 and cfg.rx.bpf_f_high_Hz > 0:
        bw_Hz = cfg.rx.bpf_f_high_Hz - cfg.rx.bpf_f_low_Hz
    else:
        bw_Hz = max(cfg.sim.data_rate_bps, 1.0)
    Q_E = 1.602e-19
    if i_signal_A <= 0 or bw_Hz <= 0:
        return issues
    i_shot_A = math.sqrt(2.0 * Q_E * i_signal_A * bw_Hz)
    if i_shot_A <= 0:
        return issues

    snr_db = 20.0 * math.log10(i_signal_A / i_shot_A)
    if snr_db < 3.0:
        issues.append(ValidationIssue(
            IssueLevel.WARNING, "channel.distance_m",
            f"link budget SNR margin {snr_db:.1f} dB < 3 dB "
            f"(P_rx={p_rx_W*1e9:.2f} nW, BW={bw_Hz:.0f} Hz). "
            f"BER will be dominated by shot noise.",
            suggestion="Reduce distance_m, raise led_radiated_power_mW, or narrow BPF",
            rule_id="link_budget.snr_margin_low",
        ))
    elif snr_db < 10.0:
        issues.append(ValidationIssue(
            IssueLevel.INFO, "channel.distance_m",
            f"link budget SNR margin {snr_db:.1f} dB is tight "
            f"(<10 dB). Expect noticeable BER from shot noise.",
            rule_id="link_budget.snr_margin_tight",
        ))

    return issues


# -----------------------------------------------------------------------------
# Rule: statistical resolution (kept from system_config)
# -----------------------------------------------------------------------------

def _rule_statistical(cfg) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    target_ber = getattr(cfg.validation, 'target_ber', 0.0) or 0.0
    if target_ber > 0:
        min_bits = int(math.ceil(10.0 / target_ber))
        if cfg.sim.n_bits < min_bits:
            issues.append(ValidationIssue(
                IssueLevel.WARNING, "sim.n_bits",
                f"n_bits={cfg.sim.n_bits} too small to resolve "
                f"target_ber={target_ber:.0e} - need >= {min_bits} bits "
                f"(~10 errors expected). BER may read 0 by chance.",
                suggestion=f"Raise n_bits to >= {min_bits}",
                rule_id="statistical.n_bits_under_target_ber",
            ))
    return issues


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

_ALL_RULES = (
    _rule_physical_sanity,
    _rule_nyquist,
    _rule_datasheet,
    _rule_link_budget,
    _rule_statistical,
)

_LEVEL_ORDER = {
    IssueLevel.ERROR: 0,
    IssueLevel.WARNING: 1,
    IssueLevel.INFO: 2,
}


def validate_config(cfg) -> List[ValidationIssue]:
    """Run every rule and return issues sorted error-first.

    A rule that raises is recorded as a WARNING with rule_id
    'rule.internal_error' instead of crashing the validator.
    """
    issues: List[ValidationIssue] = []
    for rule in _ALL_RULES:
        try:
            issues.extend(rule(cfg))
        except Exception as exc:
            issues.append(ValidationIssue(
                IssueLevel.WARNING,
                f"rule:{rule.__name__}",
                f"validator failed: {exc!r}",
                rule_id="rule.internal_error",
            ))
    issues.sort(key=lambda i: (_LEVEL_ORDER[i.level], i.field, i.rule_id))
    return issues


def has_errors(issues: List[ValidationIssue]) -> bool:
    return any(i.level == IssueLevel.ERROR for i in issues)


def format_issues(issues: List[ValidationIssue]) -> str:
    """One-line-per-issue text rendering for CLI."""
    if not issues:
        return "OK - no issues."
    lines = []
    for i in issues:
        lines.append(f"{i.level.value.upper():7s} {i.field:42s} {i.message}")
        if i.suggestion:
            lines.append(f"        -> {i.suggestion}")
    return "\n".join(lines)

# ai/parameter_validator.py
"""
Physics-Based Validation of Extracted LiFi-PV Parameters.

Checks extracted values against physically reasonable bounds
for optical wireless communication and photovoltaic systems.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Physics-based bounds for each parameter: (min, max, unit)
PARAMETER_BOUNDS = {
    # Transmitter
    'bias_current_A':          (0.001, 5.0,      'A'),
    'modulation_depth':        (0.01,  1.0,      ''),
    'led_radiated_power_mW':   (0.01,  5000.0,   'mW'),
    'led_half_angle_deg':      (1.0,   90.0,     'deg'),
    'led_driver_re':           (0.1,   1000.0,   'Ohm'),
    'led_gled':                (0.01,  1.0,      ''),
    'lens_transmittance':      (0.1,   1.0,      ''),

    # Channel
    'distance_m':              (0.01,  100.0,    'm'),
    'tx_angle_deg':            (0.0,   89.0,     'deg'),
    'rx_tilt_deg':             (0.0,   89.0,     'deg'),

    # Receiver - Solar Cell
    'n_cells_series':          (1,     100,      ''),
    'n_cells_parallel':        (1,     100,      ''),
    'sc_area_cm2':             (0.01,  1000.0,   'cm2'),
    'sc_responsivity':         (0.01,  1.2,      'A/W'),
    'sc_cj_nF':                (0.001, 100000.0, 'nF'),
    'sc_rsh_kOhm':             (0.001, 10000.0,  'kOhm'),
    'sc_iph_uA':               (0.01,  1e6,      'uA'),
    'sc_vmpp_mV':              (100.0, 5000.0,   'mV'),
    'sc_impp_uA':              (0.01,  1e6,      'uA'),
    'sc_pmpp_uW':              (0.01,  1e6,      'uW'),

    # Receiver - Signal Chain
    'r_sense_ohm':             (0.001, 10000.0,  'Ohm'),
    'ina_gain_dB':             (0.0,   80.0,     'dB'),
    'ina_gbw_kHz':             (1.0,   1e6,      'kHz'),
    'bpf_stages':              (1,     6,        ''),
    'bpf_f_low_Hz':            (0.1,   1e6,      'Hz'),
    'bpf_f_high_Hz':           (10.0,  1e9,      'Hz'),

    # DC-DC
    'dcdc_fsw_kHz':            (1.0,   10000.0,  'kHz'),
    'dcdc_l_uH':               (0.1,   10000.0,  'uH'),
    'dcdc_cp_uF':              (0.01,  10000.0,  'uF'),
    'dcdc_cl_uF':              (0.01,  10000.0,  'uF'),
    'r_load_ohm':              (1.0,   1e7,      'Ohm'),

    # Simulation
    'data_rate_bps':           (1.0,   1e10,     'bps'),
    'n_bits':                  (10,    1e6,      ''),

    # Targets
    'target_harvested_power_uW': (0.001, 1e6,    'uW'),
    'target_ber':              (0.0,   0.5,      ''),
    'target_noise_rms_mV':     (0.0,   1000.0,   'mV'),
}

VALID_MODULATIONS = {'OOK', 'OOK_Manchester', 'OFDM', 'BFSK', 'PWM_ASK'}

# Parameters that are critical for simulation accuracy
CRITICAL_PARAMS = {
    'distance_m', 'sc_area_cm2', 'sc_responsivity', 'sc_cj_nF',
    'led_radiated_power_mW', 'data_rate_bps', 'modulation',
    'led_half_angle_deg', 'ina_gain_dB', 'r_sense_ohm',
}

# Parameters that are important but not critical
IMPORTANT_PARAMS = {
    'sc_rsh_kOhm', 'sc_iph_uA', 'sc_vmpp_mV', 'sc_impp_uA', 'sc_pmpp_uW',
    'modulation_depth', 'bias_current_A', 'bpf_f_low_Hz', 'bpf_f_high_Hz',
    'r_load_ohm', 'n_cells_series',
}


def validate_parameters(params: dict, confidence: Optional[dict] = None) -> dict:
    """
    Validate extracted parameters against physics bounds.

    Args:
        params: Dict of parameter name -> value (from extraction).
        confidence: Optional dict of parameter name -> confidence level.

    Returns:
        Dict with:
            'valid': list of (name, value, status) tuples
            'warnings': list of (name, value, message) tuples
            'errors': list of (name, value, message) tuples
            'missing': list of field names with null values
            'missing_critical': list of critical fields that are null
            'missing_important': list of important fields that are null
            'score': float 0-100 overall quality score
    """
    valid = []
    warnings = []
    errors = []
    missing = []
    confidence = confidence or {}

    for name, value in params.items():
        # Skip metadata fields
        if name in ('preset_name', 'paper_reference', 'simulation_engine',
                     'noise_enable', 'shot_noise_enable', 'thermal_noise_enable',
                     'prbs_order', 't_stop_s', 'temperature_K', 'humidity_rh',
                     'amp_gain_linear', 'notch_freq_hz', 'notch_Q',
                     'ofdm_nfft', 'ofdm_qam_order', 'ofdm_n_subcarriers',
                     'ofdm_cp_len', 'ofdm_sample_rate_hz',
                     'bfsk_f0_hz', 'bfsk_f1_hz', 'pwm_freq_hz',
                     'carrier_freq_hz', 'target_data_rate_mbps',
                     'target_fec_threshold'):
            if value is not None:
                valid.append((name, value, 'ok'))
            continue

        # Handle null/missing
        if value is None:
            missing.append(name)
            continue

        # Check modulation scheme
        if name == 'modulation':
            if value in VALID_MODULATIONS:
                valid.append((name, value, 'ok'))
            else:
                warnings.append((name, value,
                    f"Unknown modulation '{value}'. "
                    f"Valid: {', '.join(sorted(VALID_MODULATIONS))}"))
            continue

        # String fields (part numbers)
        if name in ('led_part', 'driver_part', 'pv_part', 'ina_part',
                     'comparator_part'):
            if isinstance(value, str) and len(value) > 0:
                valid.append((name, value, 'ok'))
            else:
                warnings.append((name, value, "Empty or non-string part number"))
            continue

        # Numeric bounds check
        if name in PARAMETER_BOUNDS:
            lo, hi, unit = PARAMETER_BOUNDS[name]
            try:
                val = float(value)
            except (TypeError, ValueError):
                errors.append((name, value, f"Not a number: {value}"))
                continue

            if lo <= val <= hi:
                valid.append((name, val, 'ok'))
            elif val < lo:
                errors.append((name, val,
                    f"Below minimum: {val} < {lo} {unit}"))
            else:
                errors.append((name, val,
                    f"Above maximum: {val} > {hi} {unit}"))
        else:
            # Unknown field — pass through with warning
            valid.append((name, value, 'unknown_field'))

    # Cross-checks
    cross_warnings = _cross_validate(params)
    warnings.extend(cross_warnings)

    # Categorize missing params
    missing_critical = [m for m in missing if m in CRITICAL_PARAMS]
    missing_important = [m for m in missing if m in IMPORTANT_PARAMS]

    # Calculate quality score (weighted)
    total_checkable = len(valid) + len(warnings) + len(errors) + len(missing)
    if total_checkable > 0:
        # Base score from valid/total ratio
        base_score = (len(valid) / total_checkable) * 100

        # Penalty for missing critical params (each costs 5 points)
        critical_penalty = len(missing_critical) * 5
        # Penalty for missing important params (each costs 2 points)
        important_penalty = len(missing_important) * 2
        # Penalty for errors (each costs 3 points)
        error_penalty = len(errors) * 3

        score = max(0, base_score - critical_penalty - important_penalty - error_penalty)
    else:
        score = 0.0

    # Penalize for high proportion of "estimated" confidence
    if confidence:
        n_estimated = sum(1 for v in confidence.values() if v == 'estimated')
        n_total = len(confidence)
        if n_total > 0:
            estimated_ratio = n_estimated / n_total
            score *= (1.0 - 0.3 * estimated_ratio)

    result = {
        'valid': valid,
        'valid_count': len(valid),
        'warnings': warnings,
        'errors': errors,
        'missing': missing,
        'missing_critical': missing_critical,
        'missing_important': missing_important,
        'score': round(score, 1),
    }

    logger.info("Validation: %d valid, %d warnings, %d errors, %d missing "
                "(critical=%d, important=%d, score=%.1f%%)",
                len(valid), len(warnings), len(errors), len(missing),
                len(missing_critical), len(missing_important), score)
    return result


def _cross_validate(params: dict) -> list:
    """Cross-check parameter relationships."""
    warnings = []

    # BPF: low freq should be less than high freq
    f_lo = params.get('bpf_f_low_Hz')
    f_hi = params.get('bpf_f_high_Hz')
    if f_lo is not None and f_hi is not None:
        try:
            if float(f_lo) >= float(f_hi):
                warnings.append(('bpf_f_low_Hz', f_lo,
                    f"BPF low freq ({f_lo} Hz) >= high freq ({f_hi} Hz)"))
        except (TypeError, ValueError):
            pass

    # BPF: data rate should be within passband
    rate = params.get('data_rate_bps')
    if rate is not None and f_lo is not None and f_hi is not None:
        try:
            r, fl, fh = float(rate), float(f_lo), float(f_hi)
            # Fundamental frequency is half the data rate for OOK
            f_data = r / 2
            if f_data < fl * 0.5:
                warnings.append(('bpf_f_low_Hz', f_lo,
                    f"BPF low cutoff ({fl} Hz) too high for data rate "
                    f"({r} bps, fundamental ≈ {f_data:.0f} Hz)"))
            if f_data > fh * 2:
                warnings.append(('bpf_f_high_Hz', f_hi,
                    f"BPF high cutoff ({fh} Hz) too low for data rate "
                    f"({r} bps, fundamental ≈ {f_data:.0f} Hz)"))
        except (TypeError, ValueError):
            pass

    # Solar cell: Vmpp should produce reasonable Pmpp
    vmpp = params.get('sc_vmpp_mV')
    impp = params.get('sc_impp_uA')
    pmpp = params.get('sc_pmpp_uW')
    if vmpp is not None and impp is not None and pmpp is not None:
        try:
            calc_pmpp = float(vmpp) * float(impp) / 1000.0  # mV * uA / 1000 = uW
            actual = float(pmpp)
            if actual > 0 and abs(calc_pmpp - actual) / actual > 0.5:
                warnings.append(('sc_pmpp_uW', pmpp,
                    f"Pmpp ({actual} uW) doesn't match Vmpp*Impp "
                    f"({calc_pmpp:.1f} uW) within 50%"))
        except (TypeError, ValueError):
            pass

    # LED half-angle: flag if using bare LED angle with a lens
    half_angle = params.get('led_half_angle_deg')
    lens_t = params.get('lens_transmittance')
    if half_angle is not None and lens_t is not None:
        try:
            ha = float(half_angle)
            lt = float(lens_t)
            if ha > 60 and lt < 1.0:
                warnings.append(('led_half_angle_deg', half_angle,
                    f"Half-angle {ha}° is wide for a system with a lens "
                    f"(transmittance={lt}). Check if lens half-angle should be used."))
        except (TypeError, ValueError):
            pass

    # Data rate vs modulation sanity
    mod = params.get('modulation')
    if rate is not None and mod is not None:
        try:
            r = float(rate)
            if mod == 'OOK' and r > 100e6:
                warnings.append(('data_rate_bps', rate,
                    "OOK at >100 Mbps is unusual — verify modulation type"))
            if mod == 'OFDM' and r < 1000:
                warnings.append(('data_rate_bps', rate,
                    "OFDM at <1 kbps is unusual — verify modulation type"))
        except (TypeError, ValueError):
            pass

    # INA gain sanity: 40 dB = 100x, 60 dB = 1000x
    ina_gain = params.get('ina_gain_dB')
    if ina_gain is not None:
        try:
            g = float(ina_gain)
            if g > 60:
                warnings.append(('ina_gain_dB', ina_gain,
                    f"INA gain {g} dB is very high (={10**(g/20):.0f}x linear). Verify this is dB not linear."))
        except (TypeError, ValueError):
            pass

    return warnings


def format_validation_report(result: dict) -> str:
    """Format validation result into a readable text report."""
    lines = []
    lines.append(f"=== Parameter Validation Report (Score: {result['score']:.1f}%) ===\n")

    if result['errors']:
        lines.append(f"ERRORS ({len(result['errors'])}):")
        for name, val, msg in result['errors']:
            lines.append(f"  [X] {name} = {val} -- {msg}")
        lines.append("")

    if result['warnings']:
        lines.append(f"WARNINGS ({len(result['warnings'])}):")
        for name, val, msg in result['warnings']:
            lines.append(f"  [!] {name} = {val} -- {msg}")
        lines.append("")

    if result.get('missing_critical'):
        lines.append(f"MISSING CRITICAL ({len(result['missing_critical'])}):")
        for name in result['missing_critical']:
            lines.append(f"  [!!] {name}")
        lines.append("")

    if result.get('missing_important'):
        lines.append(f"MISSING IMPORTANT ({len(result['missing_important'])}):")
        for name in result['missing_important']:
            lines.append(f"  [!] {name}")
        lines.append("")

    if result['missing']:
        other_missing = [m for m in result['missing']
                         if m not in result.get('missing_critical', [])
                         and m not in result.get('missing_important', [])]
        if other_missing:
            lines.append(f"MISSING OPTIONAL ({len(other_missing)}):")
            for name in other_missing:
                lines.append(f"  [ ] {name}")
            lines.append("")

    lines.append(f"VALID: {result.get('valid_count', len(result['valid']))} parameters passed bounds check")
    return "\n".join(lines)

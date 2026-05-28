"""
Tests for cosim.validation — parameter guardrails.

Covers:
- All shipped presets load with zero ERROR-level issues (regression guard)
- Each rule_id can be triggered by a hand-crafted bad config
- validate() returns a deterministic, error-first ordering
- has_errors() / format_issues() helpers behave as documented
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from cosim.system_config import SystemConfig
from cosim.validation import (
    IssueLevel,
    ValidationIssue,
    format_issues,
    has_errors,
    validate_config,
)


# ---------------------------------------------------------------------------
# Regression: every shipped preset must be ERROR-free
# ---------------------------------------------------------------------------

_PRESET_DIR = _PROJECT_ROOT / 'presets'
_PRESET_NAMES = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(str(_PRESET_DIR / '*.json'))
)


@pytest.mark.parametrize("preset_name", _PRESET_NAMES)
def test_shipped_preset_has_no_errors(preset_name):
    cfg = SystemConfig.from_preset(preset_name)
    issues = cfg.validate()
    errors = [i for i in issues if i.level == IssueLevel.ERROR]
    assert not errors, (
        f"preset {preset_name!r} has unexpected ERROR-level issues:\n"
        + "\n".join(f"  {e.rule_id}: {e.message}" for e in errors)
    )


# ---------------------------------------------------------------------------
# Rule-by-rule triggers. One test per rule_id — confirms each fires for the
# input it's meant to catch.
# ---------------------------------------------------------------------------

def _baseline_kwargs() -> dict:
    """Minimal kwargs that produce a valid SystemConfig."""
    return {}  # all defaults are valid


def _validate_raw(**overrides) -> list:
    """Build via _post_init bypass: call validate_config on a config that
    we've patched after construction. Construction with bad values would
    raise ValueError before we get a chance to inspect issues."""
    cfg = SystemConfig()
    for dotted, value in overrides.items():
        sub, _, field = dotted.partition('.')
        setattr(getattr(cfg, sub), field, value)
    return validate_config(cfg)


def _has_rule(issues, rule_id) -> bool:
    return any(i.rule_id == rule_id for i in issues)


# --- physical sanity ---

def test_rule_distance_nonpositive():
    issues = _validate_raw(**{'channel.distance_m': -1.0})
    assert _has_rule(issues, "physical.distance_nonpositive")


def test_rule_data_rate_nonpositive():
    issues = _validate_raw(**{'sim.data_rate_bps': 0.0})
    assert _has_rule(issues, "physical.data_rate_nonpositive")


def test_rule_modulation_unknown():
    issues = _validate_raw(**{'sim.modulation': 'QPSK'})
    assert _has_rule(issues, "physical.modulation_unknown")


def test_rule_topology_unknown():
    issues = _validate_raw(**{'rx.rx_topology': 'banana'})
    assert _has_rule(issues, "physical.topology_unknown")


def test_rule_bpf_inverted():
    issues = _validate_raw(**{
        'rx.bpf_f_low_Hz': 1e5,
        'rx.bpf_f_high_Hz': 1e3,
    })
    assert _has_rule(issues, "physical.bpf_inverted")


def test_rule_fov_out_of_range():
    issues = _validate_raw(**{'channel.fov_half_angle_deg': 120.0})
    assert _has_rule(issues, "physical.fov_out_of_range")


def test_rule_wall_reflectivity_out_of_range():
    issues = _validate_raw(**{'channel.wall_reflectivity': 1.5})
    assert _has_rule(issues, "physical.wall_reflectivity_out_of_range")


def test_rule_n_reflections_negative():
    issues = _validate_raw(**{'channel.n_reflections': -2})
    assert _has_rule(issues, "physical.n_reflections_negative")


def test_rule_ambient_negative():
    issues = _validate_raw(**{'noise.ambient_illuminance_lux': -50.0})
    assert _has_rule(issues, "physical.ambient_negative")


def test_rule_led_power_negative():
    issues = _validate_raw(**{'tx.led_radiated_power_mW': -1.0})
    assert _has_rule(issues, "physical.led_power_negative")


def test_rule_bias_current_negative():
    issues = _validate_raw(**{'tx.bias_current_A': -0.1})
    assert _has_rule(issues, "physical.bias_current_negative")


# --- Nyquist / sampling ---

def test_rule_signal_above_bpf():
    issues = _validate_raw(**{
        'sim.data_rate_bps': 5e6,
        'rx.bpf_stages': 2,
        'rx.bpf_f_low_Hz': 1e3,
        'rx.bpf_f_high_Hz': 1e4,
    })
    assert _has_rule(issues, "nyquist.signal_above_bpf")


def test_rule_sim_time_short():
    issues = _validate_raw(**{
        'sim.n_bits': 10_000,
        'sim.data_rate_bps': 1000.0,
        'sim.t_stop_s': 0.001,  # holds 1 bit
    })
    assert _has_rule(issues, "nyquist.sim_time_short")


# --- datasheet ---

def test_rule_led_overcurrent():
    # LXM5-PD01 max drive is 0.700 A.
    issues = _validate_raw(**{
        'tx.led_part': 'LXM5-PD01',
        'tx.bias_current_A': 2.0,
    })
    assert _has_rule(issues, "datasheet.led_overcurrent")


def test_rule_amp_above_bandwidth():
    # INA322 GBW ~700 kHz. data_rate 5 Mbps signal ~2.5 MHz far above.
    issues = _validate_raw(**{
        'rx.ina_part': 'INA322',
        'rx.ina_gain_dB': 40.0,
        'sim.data_rate_bps': 5e6,
    })
    assert _has_rule(issues, "datasheet.amp_above_bandwidth")


# --- link budget ---

def test_rule_link_budget_snr_margin_low():
    # Push distance huge to crater the link budget.
    issues = _validate_raw(**{
        'channel.distance_m': 1000.0,
        'tx.led_radiated_power_mW': 1.0,
        'rx.sc_area_cm2': 0.01,
        'rx.sc_responsivity': 0.1,
        'rx.bpf_stages': 0,
        'sim.data_rate_bps': 1e6,
    })
    assert _has_rule(issues, "link_budget.snr_margin_low")


# --- statistical ---

def test_rule_n_bits_under_target_ber():
    issues = _validate_raw(**{
        'sim.n_bits': 100,
        'validation.target_ber': 1e-6,
    })
    assert _has_rule(issues, "statistical.n_bits_under_target_ber")


# ---------------------------------------------------------------------------
# Determinism / ordering
# ---------------------------------------------------------------------------

def test_issues_sorted_errors_first():
    # Create a config with mixed levels.
    issues = _validate_raw(**{
        'sim.n_bits': 100,
        'validation.target_ber': 1e-6,         # WARNING
        'channel.distance_m': -1.0,            # ERROR
        'channel.wall_reflectivity': 2.0,      # ERROR
    })
    levels = [i.level for i in issues]
    # All ERROR must come before any WARNING/INFO.
    last_error_idx = max(
        (idx for idx, lv in enumerate(levels) if lv == IssueLevel.ERROR),
        default=-1,
    )
    first_warn_idx = next(
        (idx for idx, lv in enumerate(levels) if lv == IssueLevel.WARNING),
        len(levels),
    )
    assert last_error_idx < first_warn_idx


def test_validate_is_deterministic():
    a = _validate_raw(**{'channel.distance_m': -1.0, 'tx.bias_current_A': -1.0})
    b = _validate_raw(**{'channel.distance_m': -1.0, 'tx.bias_current_A': -1.0})
    assert [i.rule_id for i in a] == [i.rule_id for i in b]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_has_errors_helper():
    assert not has_errors([])
    assert not has_errors([ValidationIssue(IssueLevel.WARNING, "x", "y")])
    assert has_errors([ValidationIssue(IssueLevel.ERROR, "x", "y")])


def test_format_issues_empty():
    assert "OK" in format_issues([])


def test_format_issues_renders_suggestion():
    out = format_issues([
        ValidationIssue(IssueLevel.ERROR, "f", "bad", suggestion="try X"),
    ])
    assert "ERROR" in out
    assert "try X" in out


def test_validate_method_on_systemconfig():
    cfg = SystemConfig()
    issues = cfg.validate()
    assert isinstance(issues, list)
    # default config must be ERROR-free
    assert not has_errors(issues)

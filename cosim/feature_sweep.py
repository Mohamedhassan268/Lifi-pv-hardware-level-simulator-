"""cosim/feature_sweep.py — build an ML-ready feature dataset across a sweep.

`run_feature_sweep(cfg, param, values)` patches one numeric config field over a
list of values, runs the Python engine at each point, and returns one
`extract_features` record per point (with the swept value attached). A single
run is just the 1-element case. `to_csv` / `to_jsonl` serialize the rows for
offline analysis / model training.
"""

from __future__ import annotations

import io
import json
from typing import Iterable, Optional

from cosim.features import FEATURES, TAG_KEYS, extract_features
from cosim.python_engine import run_python_simulation
from cosim.system_config import SystemConfig

# Numeric config fields we allow sweeping (guards against arbitrary attribute
# injection and gives the UI a known axis list).
SWEEPABLE = (
    "distance_m", "data_rate_bps", "modulation_depth", "ambient_illuminance_lux",
    "led_radiated_power_mW", "mcu_sample_rate_hz", "adc_bits", "n_reflections",
    "fov_half_angle_deg", "tx_angle_deg", "rx_tilt_deg",
)

# Stable column order for exports: the swept axis, identifier tags, then every
# registered feature in declaration order.
_COLUMNS = ("sweep_param", "sweep_value", *TAG_KEYS, *FEATURES.keys())


def run_feature_sweep(
    cfg: SystemConfig,
    param: str,
    values: Iterable[float],
    n_bits: Optional[int] = None,
    seed: int = 0,
) -> list[dict]:
    """Run the Python engine at each `param=value` and return feature rows.

    Each row is `extract_features(...)` plus `sweep_param` / `sweep_value`.
    Rows are row-complete even if a point degenerates (features become NaN).
    """
    if param not in SWEEPABLE:
        raise ValueError(f"param {param!r} not sweepable; choose from {SWEEPABLE}")

    base = cfg.to_dict()
    base["simulation_engine"] = "python"
    if n_bits is not None:
        base["n_bits"] = int(n_bits)
    base["random_seed"] = seed

    rows: list[dict] = []
    for v in values:
        d = dict(base)
        d[param] = v
        point_cfg = SystemConfig(**d)
        try:
            result = run_python_simulation(point_cfg)
        except Exception:  # noqa: BLE001 — never abort the whole sweep on one point
            result = {}
        rec = extract_features(result, point_cfg)
        row = {"sweep_param": param, "sweep_value": float(v), **rec}
        rows.append(row)
    return rows


def to_csv(rows: list[dict]) -> str:
    """Serialize feature rows to CSV with a stable, fully-populated header."""
    buf = io.StringIO()
    buf.write(",".join(_COLUMNS) + "\n")
    for r in rows:
        buf.write(",".join(_csv_cell(r.get(c)) for c in _COLUMNS) + "\n")
    return buf.getvalue()


def to_jsonl(rows: list[dict]) -> str:
    """Serialize feature rows to JSON Lines (one JSON object per row)."""
    return "\n".join(json.dumps(_json_row(r)) for r in rows)


def _csv_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else repr(v)  # NaN -> empty cell
    s = str(v)
    return f'"{s}"' if ("," in s or '"' in s) else s


def _json_row(r: dict) -> dict:
    out = {}
    for c in _COLUMNS:
        v = r.get(c)
        if isinstance(v, float) and v != v:  # NaN -> null
            out[c] = None
        else:
            out[c] = v
    return out

# cosim/probes.py
"""
Probe Registry — addressable intermediate signals for glass-box observability.

Every signal computed at a block boundary in the pipeline (P_tx, P_rx, I_ph,
per-source noise, V_sense, V_ina, V_bpf, V_comp, bits_rx, etc.) has a stable
string ID, a parent stage, units, and an associated equation key. The frontend
schematic edges reference these IDs to open scope views; the equations module
indexes the same keys to surface live math.

The registry is the single source of truth. Backend pipeline code calls
`ProbeCapture.capture(probe_id, arr)`; the WebSocket layer asks the capture
which probes are available for a given stage and serves them on demand as
binary frames.

Design choices:
- Flat ID namespace (`"<stage>.<name>"`) — no nesting, easy to mirror to TS.
- Spec is data, not behaviour. The capture object holds arrays; specs hold
  metadata (dtype, units, equation key, citation).
- Two probe kinds: "array" (time-series sample data) and "scalar" (a derived
  number for a single run — channel gain, SNR estimate, etc.).
- `config_hash()` tags every capture so the frontend can mark probes stale
  when the user edits the config without re-running.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# PROBE SPEC
# =============================================================================

@dataclass(frozen=True)
class ProbeSpec:
    """Metadata for a single named probe point."""
    id: str                       # e.g. "rx.V_ina"
    stage: str                    # "TX" | "Channel" | "RX"
    kind: str                     # "array" | "scalar"
    units: str                    # SI units string, e.g. "V", "A", "W", "dB", "1"
    dtype: str                    # numpy dtype name; "float64" | "uint8" | etc.
    label: str                    # short human label, e.g. "Photocurrent (clean)"
    description: str              # one-line explanation
    equation_key: Optional[str] = None   # links to cosim.equations.EQUATIONS
    citation: Optional[str] = None       # paper/datasheet reference
    edge_id: Optional[str] = None        # react-flow edge id this probe attaches to


# =============================================================================
# REGISTRY
# =============================================================================

# Edge IDs match SchematicCanvas.tsx edges: "tx-channel", "channel-rx".
# Probes that don't sit on an edge (per-source noise, intermediate RX-chain
# voltages) leave edge_id=None and are reached via the RX node's Inspect panel.

PROBE_REGISTRY: Dict[str, ProbeSpec] = {
    # ---- TX block ----
    "tx.bits": ProbeSpec(
        id="tx.bits", stage="TX", kind="array", units="1", dtype="uint8",
        label="TX bit stream", description="Raw user/PRBS bits fed into the modulator.",
        equation_key=None, citation=None, edge_id=None,
    ),
    "tx.P_tx": ProbeSpec(
        id="tx.P_tx", stage="TX", kind="array", units="W", dtype="float64",
        label="Transmitted optical power", description="P_tx(t) emitted by the LED after modulation.",
        equation_key="modulation.ook", citation="Kahn & Barry, Proc. IEEE 1997",
        edge_id="tx-channel",
    ),

    # ---- Channel block (scalars + array) ----
    "channel.G_los": ProbeSpec(
        id="channel.G_los", stage="Channel", kind="scalar", units="1", dtype="float64",
        label="LOS channel gain", description="Lambertian line-of-sight gain H_LOS(0).",
        equation_key="channel.los_gain", citation="Komine & Nakagawa, IEEE TCE 2004",
        edge_id=None,
    ),
    "channel.G_diffuse": ProbeSpec(
        id="channel.G_diffuse", stage="Channel", kind="scalar", units="1", dtype="float64",
        label="Diffuse channel gain", description="First-order wall-reflection contribution.",
        equation_key="channel.diffuse_gain", citation="Komine & Nakagawa, IEEE TCE 2004",
        edge_id=None,
    ),
    "channel.beer_lambert": ProbeSpec(
        id="channel.beer_lambert", stage="Channel", kind="scalar", units="1", dtype="float64",
        label="Beer-Lambert factor", description="Atmospheric transmission exp(-alpha*d).",
        equation_key="channel.beer_lambert", citation="Correa et al. 2025",
        edge_id=None,
    ),
    "channel.G_total": ProbeSpec(
        id="channel.G_total", stage="Channel", kind="scalar", units="1", dtype="float64",
        label="Total channel gain", description="(G_LOS + G_diffuse) * BL factor.",
        equation_key="channel.total_gain", citation=None, edge_id=None,
    ),
    "channel.P_rx": ProbeSpec(
        id="channel.P_rx", stage="Channel", kind="array", units="W", dtype="float64",
        label="Received optical power", description="P_rx(t) at the PV-cell aperture.",
        equation_key="channel.total_gain", citation=None, edge_id="channel-rx",
    ),

    # ---- RX block: photodetection ----
    "rx.I_ph": ProbeSpec(
        id="rx.I_ph", stage="RX", kind="array", units="A", dtype="float64",
        label="Photocurrent (clean)", description="I_ph = R(T) * P_rx, before noise injection.",
        equation_key="pv.responsivity", citation="PV cell datasheet",
        edge_id=None,
    ),

    # ---- RX block: per-source noise arrays ----
    "rx.noise.shot": ProbeSpec(
        id="rx.noise.shot", stage="RX", kind="array", units="A", dtype="float64",
        label="Shot noise contribution", description="2q(I_ph+I_dark)Bn — Poisson statistics of photocurrent.",
        equation_key="noise.shot", citation="Kahn & Barry, Proc. IEEE 1997",
        edge_id=None,
    ),
    "rx.noise.thermal": ProbeSpec(
        id="rx.noise.thermal", stage="RX", kind="array", units="A", dtype="float64",
        label="Thermal noise contribution", description="4kT*Bn/R_load — Johnson-Nyquist across the sense resistor.",
        equation_key="noise.thermal", citation="Nyquist 1928",
        edge_id=None,
    ),
    "rx.noise.ambient": ProbeSpec(
        id="rx.noise.ambient", stage="RX", kind="array", units="A", dtype="float64",
        label="Ambient-light noise", description="2q*I_ambient*Bn from background illumination.",
        equation_key="noise.ambient", citation="Kahn & Barry, Proc. IEEE 1997",
        edge_id=None,
    ),
    "rx.noise.amplifier": ProbeSpec(
        id="rx.noise.amplifier", stage="RX", kind="array", units="A", dtype="float64",
        label="Amplifier input-referred noise", description="(e_n^2/R^2 + i_n^2)*Bn — INA322 spec.",
        equation_key="noise.amplifier", citation="INA322 datasheet (SBOS163)",
        edge_id=None,
    ),
    "rx.noise.adc": ProbeSpec(
        id="rx.noise.adc", stage="RX", kind="array", units="A", dtype="float64",
        label="ADC quantization noise", description="V_LSB^2/12 referred to the input current domain.",
        equation_key="noise.adc", citation="Widrow 1956",
        edge_id=None,
    ),
    "rx.noise.processing": ProbeSpec(
        id="rx.noise.processing", stage="RX", kind="array", units="A", dtype="float64",
        label="Processing/threshold noise", description="Comparator offset and jitter, input-referred.",
        equation_key="noise.processing", citation="TLV7011 datasheet (SBOS819)",
        edge_id=None,
    ),
    "rx.noise.mains_flicker": ProbeSpec(
        id="rx.noise.mains_flicker", stage="RX", kind="array", units="A", dtype="float64",
        label="AC mains flicker", description="Deterministic 100/120 Hz interferer + harmonics scaled to ambient.",
        equation_key="noise.mains_flicker", citation="Wilkins et al., IEEE TIA 2010",
        edge_id=None,
    ),
    "rx.noise.pink": ProbeSpec(
        id="rx.noise.pink", stage="RX", kind="array", units="A", dtype="float64",
        label="Amplifier 1/f noise", description="Pink noise below the amplifier flicker corner.",
        equation_key="noise.pink", citation="OpAmp datasheet flicker corner spec",
        edge_id=None,
    ),

    # ---- RX block: signal chain ----
    "rx.I_ph_noisy": ProbeSpec(
        id="rx.I_ph_noisy", stage="RX", kind="array", units="A", dtype="float64",
        label="Photocurrent (noisy)", description="I_ph + summed noise samples.",
        equation_key=None, citation=None, edge_id=None,
    ),
    "rx.V_sense": ProbeSpec(
        id="rx.V_sense", stage="RX", kind="array", units="V", dtype="float64",
        label="Sense voltage", description="V_sense = I_ph_noisy * R_sense.",
        equation_key=None, citation=None, edge_id=None,
    ),
    "rx.V_ina": ProbeSpec(
        id="rx.V_ina", stage="RX", kind="array", units="V", dtype="float64",
        label="INA output", description="Voltage after instrumentation amplifier (rail-clamped).",
        equation_key=None, citation="INA322 datasheet (SBOS163)", edge_id=None,
    ),
    "rx.V_bpf": ProbeSpec(
        id="rx.V_bpf", stage="RX", kind="array", units="V", dtype="float64",
        label="BPF output", description="Voltage after the band-pass filter stages.",
        equation_key=None, citation=None, edge_id=None,
    ),
    "rx.V_comp": ProbeSpec(
        id="rx.V_comp", stage="RX", kind="array", units="V", dtype="float64",
        label="Comparator output", description="Digital edges (0 / V_cc).",
        equation_key=None, citation="TLV7011 datasheet (SBOS819)", edge_id=None,
    ),

    # ---- RX block: demod & metrics ----
    "rx.bits_rx": ProbeSpec(
        id="rx.bits_rx", stage="RX", kind="array", units="1", dtype="uint8",
        label="Recovered bits", description="Demodulator output, ready for BER compare.",
        equation_key=None, citation=None, edge_id=None,
    ),
    "rx.snr_db": ProbeSpec(
        id="rx.snr_db", stage="RX", kind="scalar", units="dB", dtype="float64",
        label="SNR estimate", description="Electrical SNR estimate from signal/noise powers.",
        equation_key=None, citation=None, edge_id=None,
    ),
}


def all_probe_ids() -> List[str]:
    """Stable, ordered list of all known probe IDs."""
    return list(PROBE_REGISTRY.keys())


def probes_for_stage(stage: str) -> List[str]:
    """All probe IDs belonging to a given pipeline stage."""
    return [pid for pid, spec in PROBE_REGISTRY.items() if spec.stage == stage]


def probes_for_edge(edge_id: str) -> List[str]:
    """All probe IDs attached to a specific react-flow edge."""
    return [pid for pid, spec in PROBE_REGISTRY.items() if spec.edge_id == edge_id]


def registry_export() -> List[Dict[str, Any]]:
    """Serializable view of the full registry (for /api endpoints + TS mirror)."""
    return [
        {
            "id": spec.id,
            "stage": spec.stage,
            "kind": spec.kind,
            "units": spec.units,
            "dtype": spec.dtype,
            "label": spec.label,
            "description": spec.description,
            "equation_key": spec.equation_key,
            "citation": spec.citation,
            "edge_id": spec.edge_id,
        }
        for spec in PROBE_REGISTRY.values()
    ]


# =============================================================================
# CONFIG HASH
# =============================================================================

def config_hash(cfg) -> str:
    """Deterministic short hash of a SystemConfig (or any JSON-able dict).

    Used to tag captured probes so the frontend can mark them stale when the
    user edits the config without re-running. Truncated to 16 hex chars —
    plenty for staleness detection, not a security primitive.
    """
    if hasattr(cfg, "to_dict"):
        payload = cfg.to_dict()
    elif isinstance(cfg, dict):
        payload = cfg
    else:
        payload = {"_repr": repr(cfg)}
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# CAPTURE CONTEXT
# =============================================================================

@dataclass
class ProbeCapture:
    """
    Holds the captured probe values for a single pipeline run.

    Both `cosim.pipeline.SimulationPipeline` and `cosim.python_engine.
    run_python_simulation` accept an optional `ProbeCapture` and call
    `.capture(probe_id, value)` at each block boundary. If no capture is
    provided, the pipeline runs as before with no per-probe overhead.
    """
    session_dir: Optional[Path] = None
    config_hash: str = ""
    enabled_ids: Optional[List[str]] = None    # None = capture every known probe
    _arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    _scalars: Dict[str, float] = field(default_factory=dict)
    _saved: Dict[str, dict] = field(default_factory=dict)   # per-probe metadata snapshot

    def is_enabled(self, probe_id: str) -> bool:
        if probe_id not in PROBE_REGISTRY:
            return False
        if self.enabled_ids is None:
            return True
        return probe_id in self.enabled_ids

    def capture(self, probe_id: str, value: Any, *, units: Optional[str] = None) -> None:
        """Record one probe value. Unknown or disabled IDs are silently ignored."""
        spec = PROBE_REGISTRY.get(probe_id)
        if spec is None:
            logger.debug("probe.capture: unknown id %s (ignored)", probe_id)
            return
        if not self.is_enabled(probe_id):
            return

        try:
            if spec.kind == "array":
                arr = np.asarray(value)
                if spec.dtype:
                    arr = arr.astype(spec.dtype, copy=False)
                self._arrays[probe_id] = arr
            else:
                self._scalars[probe_id] = float(np.asarray(value).reshape(-1)[0]) \
                    if hasattr(value, "__len__") else float(value)
        except (TypeError, ValueError) as e:
            logger.warning("probe.capture(%s) failed: %s", probe_id, e)
            return

        self._saved[probe_id] = {
            "units": units or spec.units,
            "stage": spec.stage,
            "kind": spec.kind,
            "dtype": spec.dtype,
            "config_hash": self.config_hash,
        }

    # -- accessors -----------------------------------------------------------

    def get(self, probe_id: str):
        """Return the captured value (np.ndarray for arrays, float for scalars)."""
        if probe_id in self._arrays:
            return self._arrays[probe_id]
        return self._scalars.get(probe_id)

    def meta(self, probe_id: str) -> Optional[dict]:
        return self._saved.get(probe_id)

    def available_ids(self, stage: Optional[str] = None) -> List[str]:
        ids = list(self._arrays.keys()) + list(self._scalars.keys())
        if stage is None:
            return ids
        return [pid for pid in ids if PROBE_REGISTRY[pid].stage == stage]

    def has(self, probe_id: str) -> bool:
        return probe_id in self._arrays or probe_id in self._scalars

    # -- persistence ---------------------------------------------------------

    def save(self) -> Optional[Path]:
        """Persist captured probes to session_dir/probes/ (npz + scalars.json).

        Returns the probes directory path, or None if no session_dir was set.
        """
        if not self.session_dir:
            return None
        out = Path(self.session_dir) / "probes"
        out.mkdir(parents=True, exist_ok=True)

        if self._arrays:
            # numpy savez doesn't accept dots in keys
            np.savez(
                out / "arrays.npz",
                **{pid.replace(".", "__"): arr for pid, arr in self._arrays.items()},
            )
        meta_payload = {
            "config_hash": self.config_hash,
            "scalars": self._scalars,
            "meta": self._saved,
        }
        (out / "manifest.json").write_text(
            json.dumps(meta_payload, indent=2, default=str), encoding="utf-8"
        )
        return out

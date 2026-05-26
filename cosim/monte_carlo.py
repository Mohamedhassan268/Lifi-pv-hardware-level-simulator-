# cosim/monte_carlo.py
"""
Monte Carlo tolerance analysis for production yield estimation.

Given a baseline preset and a tolerance specification (fractional ranges
per parameter), sample N perturbed configs, run the pipeline, and report
the BER/SNR distribution.

Tolerance spec format (all values are fractional, e.g. 0.05 = +/-5%):
    {
        "responsivity": 0.05,            # +/-5% around nominal
        "pv_dark_current_A": 0.20,       # +/-20%
        "rx_load_resistance": 0.01,      # +/-1% (precision resistor)
        "distance_m": 0.02,              # +/-2% placement error
    }

Distributions:
    Each parameter is drawn uniformly from [nominal*(1-tol), nominal*(1+tol)].
    Uniform (not Gaussian) because component tolerances are typically
    specified as hard bounds, not sigma.

The July 2026 PCB bring-up in Cairo will use 500 units; this module
feeds the yield target (currently: 95% of units meet BER < 1e-3).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cosim.system_config import SystemConfig
from cosim.python_engine import run_python_simulation


@dataclass
class MonteCarloResult:
    """Aggregate result from a Monte Carlo sweep."""
    preset: str
    n_runs: int
    tolerance_spec: Dict[str, float]
    ber_samples: List[float] = field(default_factory=list)
    snr_samples: List[float] = field(default_factory=list)
    p_rx_samples: List[float] = field(default_factory=list)
    ber_threshold: float = 1e-3

    @property
    def yield_fraction(self) -> float:
        """Fraction of runs meeting BER < threshold."""
        if not self.ber_samples:
            return 0.0
        passing = sum(1 for b in self.ber_samples if b < self.ber_threshold)
        return passing / len(self.ber_samples)

    @property
    def ber_median(self) -> float:
        return float(np.median(self.ber_samples)) if self.ber_samples else 0.0

    @property
    def ber_p95(self) -> float:
        return float(np.percentile(self.ber_samples, 95)) if self.ber_samples else 0.0

    @property
    def snr_median_dB(self) -> float:
        return float(np.median(self.snr_samples)) if self.snr_samples else 0.0


def _sample_param(nominal: float, tolerance: float, rng: np.random.Generator) -> float:
    """Uniformly sample a parameter within +/-tolerance of its nominal value."""
    lo = nominal * (1.0 - tolerance)
    hi = nominal * (1.0 + tolerance)
    return float(rng.uniform(lo, hi))


def run_monte_carlo(preset: str,
                    tolerance_spec: Dict[str, float],
                    n_runs: int = 100,
                    ber_threshold: float = 1e-3,
                    seed: Optional[int] = None,
                    output_dir: Optional[str] = None,
                    verbose: bool = True) -> MonteCarloResult:
    """Run Monte Carlo tolerance sweep.

    Args:
        preset: Preset name (e.g. 'kadirvelu2021')
        tolerance_spec: {field_name: fractional_tol} dict
        n_runs: Number of MC trials
        ber_threshold: BER threshold for yield calculation
        seed: RNG seed for reproducibility
        output_dir: If set, save histogram figure here

    Returns:
        MonteCarloResult with BER/SNR samples and yield statistics.
    """
    rng = np.random.default_rng(seed)
    baseline = SystemConfig.from_preset(preset)

    # Validate tolerance spec references real fields
    for fname in tolerance_spec:
        if not hasattr(baseline, fname):
            raise ValueError(
                f"Tolerance spec references unknown field '{fname}'. "
                f"Check spelling against SystemConfig.")

    # Capture nominal values up-front
    nominals: Dict[str, float] = {}
    for fname in tolerance_spec:
        val = getattr(baseline, fname)
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"Tolerance spec field '{fname}' must be numeric, "
                f"got {type(val).__name__}")
        nominals[fname] = float(val)

    result = MonteCarloResult(
        preset=preset,
        n_runs=n_runs,
        tolerance_spec=dict(tolerance_spec),
        ber_threshold=ber_threshold,
    )

    for i in range(n_runs):
        overrides = {
            fname: _sample_param(nominals[fname], tol, rng)
            for fname, tol in tolerance_spec.items()
        }
        cfg = baseline.replace(**overrides)

        try:
            res = run_python_simulation(cfg)
            ber = float(res.get('ber', 1.0))
            snr = float(res.get('snr_est_dB', 0.0))
            p_rx = float(res.get('P_rx_avg_uW', 0.0))
        except Exception as e:
            if verbose:
                print(f"  run {i+1}: FAILED ({e!r})")
            ber, snr, p_rx = 1.0, 0.0, 0.0

        result.ber_samples.append(ber)
        result.snr_samples.append(snr)
        result.p_rx_samples.append(p_rx)

        if verbose and (i + 1) % max(1, n_runs // 10) == 0:
            print(f"  run {i+1}/{n_runs}  "
                  f"median BER = {result.ber_median:.3e}  "
                  f"yield = {result.yield_fraction*100:.1f}%")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _plot_monte_carlo(result, out / f'monte_carlo_{preset}.png')

    return result


def _plot_monte_carlo(result: MonteCarloResult, path: Path):
    ber = np.array([max(b, 1e-8) for b in result.ber_samples])
    snr = np.array(result.snr_samples)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        f'Monte Carlo: {result.preset}  '
        f'(N={result.n_runs}, yield={result.yield_fraction*100:.1f}% '
        f'at BER<{result.ber_threshold:.0e})'
    )

    axes[0].hist(np.log10(ber), bins=30, color='steelblue', edgecolor='black')
    axes[0].axvline(np.log10(result.ber_threshold), color='red',
                    linestyle='--', label=f'BER threshold')
    axes[0].set_xlabel('log10(BER)')
    axes[0].set_ylabel('Runs')
    axes[0].set_title('BER distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(snr, bins=30, color='darkgreen', edgecolor='black')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('Runs')
    axes[1].set_title('SNR distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

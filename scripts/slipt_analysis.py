"""SLIPT data<->energy analysis for the greenhouse node.

Generates two figures + summary tables from the `slipt_node` preset, reusing the
existing feature-sweep machinery (no new physics):

  1. Pareto  — data quality (BER) vs harvested power across the PV load
               operating point (r_sense_ohm: short-circuit -> MPP -> open).
  2. Greenhouse conditions — how BER and harvest respond to ambient light
               (night -> full sun) and to link distance, at a fixed operating
               point near the data<->energy knee.

Run:
    python scripts/slipt_analysis.py [-o OUTDIR] [--n-bits N]

Figures land in OUTDIR (default workspace/slipt/). Kept out of git on purpose;
docs/greenhouse_system.md records the numbers + how to reproduce.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosim.system_config import SystemConfig  # noqa: E402
from cosim.feature_sweep import run_feature_sweep  # noqa: E402

PRESET = "slipt_node"


def _col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r.get(key, float("nan")) or float("nan")) for r in rows])


def _cfg_with(base: SystemConfig, **overrides) -> SystemConfig:
    d = base.to_dict()
    d.update(overrides)
    return SystemConfig(**d)


def pareto(base: SystemConfig, n_bits: int, outdir: Path) -> None:
    R = [50, 100, 200, 300, 500, 1000, 1500, 2200, 3300, 6800, 15000, 33000]
    rows = run_feature_sweep(base, "r_sense_ohm", R, n_bits=n_bits, seed=0)
    harvest = _col(rows, "harvested_power_W") * 1e6  # uW
    ber = _col(rows, "ber")

    mpp_i = int(np.nanargmax(harvest))
    # Low loads can degenerate (signal too weak to demod -> NaN BER); the
    # "best data" reference is the lowest *finite* BER point.
    finite = [i for i in range(len(R)) if np.isfinite(ber[i])]
    best_i = min(finite, key=lambda i: ber[i]) if finite else mpp_i

    print("\n=== Pareto: PV load sweep ===")
    print(f"{'R (ohm)':>9} {'harvest_uW':>11} {'BER':>10}")
    for Ri, h, b in zip(R, harvest, ber):
        print(f"{Ri:>9} {h:>11.3f} {b:>10.2e}")
    print(f"MPP @ {R[mpp_i]} ohm ({harvest[mpp_i]:.1f} uW); "
          f"best data @ {R[best_i]} ohm (BER {ber[best_i]:.1e}, {harvest[best_i]:.1f} uW)")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(harvest, ber, "o-", color="#1f77b4", zorder=2)
    for Ri, h, b in zip(R, harvest, ber):
        ax.annotate(f"{Ri}Ω", (h, b), textcoords="offset points",
                    xytext=(6, 5), fontsize=8, color="#555")
    ax.scatter([harvest[mpp_i]], [ber[mpp_i]], s=140, facecolors="none",
               edgecolors="#d62728", linewidths=2, zorder=3,
               label=f"MPP ({R[mpp_i]}Ω, {harvest[mpp_i]:.0f}µW)")
    ax.scatter([harvest[best_i]], [ber[best_i]], s=140, facecolors="none",
               edgecolors="#2ca02c", linewidths=2, zorder=3,
               label=f"best data ({R[best_i]}Ω, BER={ber[best_i]:.1e})")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Harvested power (µW)")
    ax.set_ylabel("BER (symlog; 0 at axis)")
    ax.set_title(f"SLIPT data↔energy Pareto — {PRESET}\n"
                 "(PV load operating point; labels = r_sense)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = outdir / "slipt_pareto.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  figure -> {out}")


def greenhouse(base: SystemConfig, n_bits: int, outdir: Path,
               knee_R: float = 1000.0) -> None:
    cfg = _cfg_with(base, r_sense_ohm=knee_R)

    lux = [0.0, 100.0, 1000.0, 10000.0, 50000.0, 100000.0]  # night -> full sun
    rows_a = run_feature_sweep(cfg, "ambient_illuminance_lux", lux, n_bits=n_bits, seed=0)
    ber_a = _col(rows_a, "ber")
    har_a = _col(rows_a, "harvested_power_W") * 1e6

    dist = [0.1, 0.2, 0.325, 0.5, 0.75, 1.0]
    rows_d = run_feature_sweep(cfg, "distance_m", dist, n_bits=n_bits, seed=0)
    ber_d = _col(rows_d, "ber")
    har_d = _col(rows_d, "harvested_power_W") * 1e6

    print(f"\n=== Greenhouse conditions (R={knee_R:.0f} ohm) ===")
    print("ambient lux ->", dict(zip(lux, zip(np.round(ber_a, 5), np.round(har_a, 2)))))
    print("distance m  ->", dict(zip(dist, zip(np.round(ber_d, 5), np.round(har_d, 2)))))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].semilogx(np.maximum(lux, 1), ber_a, "o-", color="#d62728")
    axes[0, 0].set_xlabel("Ambient illuminance (lux)"); axes[0, 0].set_ylabel("BER")
    axes[0, 0].set_title("Data vs ambient light (night → sun)")
    axes[0, 1].semilogx(np.maximum(lux, 1), har_a, "s-", color="#ff7f0e")
    axes[0, 1].set_xlabel("Ambient illuminance (lux)"); axes[0, 1].set_ylabel("Harvest (µW)")
    axes[0, 1].set_title("Harvest vs ambient light")
    axes[1, 0].plot(dist, ber_d, "o-", color="#d62728")
    axes[1, 0].set_xlabel("Distance (m)"); axes[1, 0].set_ylabel("BER")
    axes[1, 0].set_title("Data vs distance")
    axes[1, 1].plot(dist, har_d, "s-", color="#ff7f0e")
    axes[1, 1].set_xlabel("Distance (m)"); axes[1, 1].set_ylabel("Harvest (µW)")
    axes[1, 1].set_title("Harvest vs distance")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"SLIPT greenhouse-condition sensitivity — {PRESET} @ {knee_R:.0f}Ω",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = outdir / "slipt_greenhouse.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  figure -> {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SLIPT data<->energy analysis")
    p.add_argument("-o", "--output", default=str(ROOT / "workspace" / "slipt"))
    p.add_argument("--n-bits", type=int, default=2500, dest="n_bits")
    args = p.parse_args(argv)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    base = SystemConfig.from_preset(PRESET)

    pareto(base, args.n_bits, outdir)
    greenhouse(base, args.n_bits, outdir)
    print(f"\nDone. Figures in: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

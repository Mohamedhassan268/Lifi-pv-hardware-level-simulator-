"""Path C / M2 sizing chart — deployable coverage vs luminaire power.

Operationalizes the M2 finding ("self-powered region is a central disc"): for a
range of ceiling-luminaire optical powers, compute what fraction of the
greenhouse floor a self-powered node can be deployed on, at several per-node
power budgets. Answers "how bright a luminaire to power-cover X% of the floor?"

Reuses `coverage_grid` from coverage_map.py: a node budget is just a threshold on
the harvest field, so all budgets are evaluated from one grid per power point.

Run:
    python scripts/coverage_sizing.py [--grid 7] [--n-bits 600]
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
from scripts.coverage_map import coverage_grid  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Deployable-coverage sizing chart (Path C / M2)")
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--n-bits", type=int, default=600)
    p.add_argument("--height", type=float, default=2.5)
    p.add_argument("--half-extent", type=float, default=2.0)
    p.add_argument("--half-angle", type=float, default=40.0)
    p.add_argument("--r-sense", type=float, default=500.0)
    p.add_argument("--ber-thresh", type=float, default=1e-2)
    p.add_argument("--dcdc-eff", type=float, default=0.75)
    p.add_argument("-o", "--output", default=str(ROOT / "workspace" / "slipt"))
    args = p.parse_args(argv)

    powers = [1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0]  # mW optical
    budgets = [1.0, 10.0, 50.0]  # uW per-node draw

    base = SystemConfig.from_preset("slipt_node")
    # rows: deployable % per (power, budget); also data-coverage % per power.
    dep = np.zeros((len(powers), len(budgets)))
    data_cov = np.zeros(len(powers))

    print(f"Sizing sweep: {len(powers)} powers x {args.grid}x{args.grid} grid "
          f"(node R={args.r_sense:g} ohm, h={args.height} m)")
    for ip, pw in enumerate(powers):
        g = coverage_grid(base, args.height, args.half_extent, args.grid,
                          args.n_bits, args.half_angle, pw, 0.0, args.r_sense)
        ber = g["BER"]; har = g["harvest_uW"]
        closes = np.isfinite(ber) & (ber <= args.ber_thresh)
        usable = np.where(np.isfinite(har), har * args.dcdc_eff, 0.0)
        data_cov[ip] = 100.0 * np.count_nonzero(closes) / closes.size
        for ib, b in enumerate(budgets):
            dep[ip, ib] = 100.0 * np.count_nonzero(closes & (usable >= b)) / closes.size
        print(f"  {pw/1000:>5.1f} W  data={data_cov[ip]:3.0f}%  " +
              "  ".join(f"{b:g}uW={dep[ip, ib]:3.0f}%" for ib, b in enumerate(budgets)))

    fig, ax = plt.subplots(figsize=(8, 5.4))
    pw_W = np.array(powers) / 1000.0
    ax.plot(pw_W, data_cov, "k--", marker="o", label="data link closes")
    for ib, b in enumerate(budgets):
        ax.plot(pw_W, dep[:, ib], marker="s", label=f"self-powered (≥{b:g}µW node)")
    ax.set_xscale("log")
    ax.set_xlabel("Luminaire optical power (W)")
    ax.set_ylabel("% of greenhouse floor")
    ax.set_ylim(-3, 103)
    ax.set_title(f"Deployable coverage vs luminaire power\n"
                 f"{2*args.half_extent:.0f}×{2*args.half_extent:.0f} m floor, "
                 f"h={args.height} m, node R={args.r_sense:g}Ω")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fpath = out / "coverage_sizing.png"
    fig.savefig(fpath, dpi=160)
    plt.close(fig)
    print(f"  figure -> {fpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Path C — multi-luminaire greenhouse coverage.

Follows the sizing finding ("single-luminaire power-coverage is expensive"): does
splitting the SAME total optical power across more ceiling luminaires cover more
floor for a self-powered node? Compares L = 1, 4, 9 luminaires at fixed total
power.

Model (reuses the engine + the ambient→harvest path):
  - per floor point, compute each luminaire's channel gain (geometry);
  - the DATA link uses the strongest luminaire (its distance/angle drive the
    modulated signal);
  - every other luminaire's light is summed and injected as equivalent ambient
    DC illuminance, so it adds to harvest (and to shot noise) — a first-cut that
    treats co-luminaires as DC light, not co-channel data interference.

Run:
    python scripts/coverage_multi.py [--grid 7] [--total-power-W 8] [--budget-uW 10]
"""

from __future__ import annotations

import argparse
import math
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
from cosim.channel import OpticalChannel  # noqa: E402
from cosim.python_engine import run_python_simulation  # noqa: E402

LUX_PER_W_CM2 = 1.0 / 1.46e-6  # inverse of noise.py's 1 lux ≈ 1.46 µW/cm²


def lumin_positions(L: int, span: float) -> list[tuple[float, float]]:
    """L luminaires on a k×k ceiling grid spanning ±span (L must be a square)."""
    k = int(round(math.sqrt(L)))
    coords = [0.0] if k == 1 else list(np.linspace(-span, span, k))
    return [(x, y) for y in coords for x in coords]


def coverage(base: SystemConfig, lumins, total_power_W, height, half_extent,
             n, n_bits, half_angle, area_cm2, fov):
    xs = np.linspace(-half_extent, half_extent, n)
    BER = np.full((n, n), np.nan)
    HAR = np.full((n, n), np.nan)
    p_each_W = total_power_W / len(lumins)

    d0 = base.to_dict()
    d0.update(simulation_engine="python", n_bits=int(n_bits),
              led_half_angle_deg=float(half_angle),
              led_radiated_power_mW=float(p_each_W * 1e3), r_sense_ohm=500.0)

    for iy, y in enumerate(xs):
        for ix, x in enumerate(xs):
            # Per-luminaire received power via channel geometry.
            prx = []
            geo = []
            for (lx, ly) in lumins:
                r = math.hypot(x - lx, y - ly)
                d = math.hypot(r, height)
                ang = math.degrees(math.atan2(r, height))
                g = OpticalChannel(distance_m=d, led_half_angle_deg=half_angle,
                                   rx_area_cm2=area_cm2, tx_angle_deg=ang,
                                   rx_tilt_deg=ang, fov_half_angle_deg=fov).channel_gain()
                prx.append(p_each_W * g)
                geo.append((d, ang))
            prx = np.array(prx)
            s = int(np.argmax(prx))                  # strongest = data link
            others_W = float(prx.sum() - prx[s])     # rest -> DC harvest light
            extra_lux = others_W * LUX_PER_W_CM2 / area_cm2
            d_s, ang_s = geo[s]
            cfg = SystemConfig(**{**d0, "distance_m": d_s, "tx_angle_deg": ang_s,
                                  "rx_tilt_deg": ang_s,
                                  "ambient_illuminance_lux": extra_lux})
            try:
                res = run_python_simulation(cfg)
                BER[iy, ix] = float(res.get("ber", np.nan))
                vc, ic = res.get("V_cell"), res.get("I_cell")
                if vc is not None and ic is not None:
                    HAR[iy, ix] = float(np.mean(np.asarray(vc) * np.asarray(ic)) * 1e6)
            except Exception:  # noqa: BLE001
                pass
    return xs, BER, HAR


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Multi-luminaire greenhouse coverage")
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--n-bits", type=int, default=600)
    p.add_argument("--height", type=float, default=2.5)
    p.add_argument("--half-extent", type=float, default=2.0)
    p.add_argument("--half-angle", type=float, default=40.0)
    p.add_argument("--total-power-W", type=float, default=8.0)
    p.add_argument("--budget-uW", type=float, default=10.0)
    p.add_argument("--dcdc-eff", type=float, default=0.75)
    p.add_argument("--ber-thresh", type=float, default=1e-2)
    p.add_argument("-o", "--output", default=str(ROOT / "workspace" / "slipt"))
    args = p.parse_args(argv)

    base = SystemConfig.from_preset("slipt_node")
    area = base.sc_area_cm2
    fov = getattr(base, "fov_half_angle_deg", 90.0)
    layouts = [1, 4, 9]
    span = args.half_extent * 0.6

    fig, axes = plt.subplots(1, len(layouts), figsize=(6 * len(layouts), 5.4))
    cmap = matplotlib.colors.ListedColormap(["#3b0f3f", "#f0a73a", "#1a9850"])
    print(f"Multi-luminaire: total {args.total_power_W} W split across L, "
          f"node budget {args.budget_uW} uW, {args.grid}x{args.grid} grid")
    for ax, L in zip(axes, layouts):
        lumins = lumin_positions(L, span)
        xs, ber, har = coverage(base, lumins, args.total_power_W, args.height,
                                args.half_extent, args.grid, args.n_bits,
                                args.half_angle, area, fov)
        closes = np.isfinite(ber) & (ber <= args.ber_thresh)
        usable = np.where(np.isfinite(har), har * args.dcdc_eff, 0.0)
        deployable = closes & (usable >= args.budget_uW)
        dep = 100.0 * np.count_nonzero(deployable) / deployable.size
        print(f"  L={L}  ({args.total_power_W/L:.2f} W each)  deployable={dep:3.0f}%  "
              f"harvest {np.nanmin(har):.1f}-{np.nanmax(har):.1f} uW")

        cat = np.zeros_like(ber); cat[closes] = 1.0; cat[deployable] = 2.0
        ext = [xs[0], xs[-1], xs[0], xs[-1]]
        ax.imshow(cat, extent=ext, origin="lower", cmap=cmap, vmin=-0.5, vmax=2.5,
                  aspect="equal")
        ax.scatter([p[0] for p in lumins], [p[1] for p in lumins],
                   marker="*", c="white", s=130, edgecolors="k")
        ax.set_title(f"L={L} luminaires ({args.total_power_W/L:.2f} W each)\n"
                     f"deployable {dep:.0f}% of floor")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    fig.suptitle(f"Same {args.total_power_W} W total, split across more luminaires "
                 f"→ more deployable floor (node {args.budget_uW:g}µW)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fpath = out / "coverage_multi.png"
    fig.savefig(fpath, dpi=160); plt.close(fig)
    print(f"  figure -> {fpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

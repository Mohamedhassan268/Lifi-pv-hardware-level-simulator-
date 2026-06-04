"""Path C / M1 — multi-node greenhouse coverage map.

Foundation for the multi-node canvas: evaluate the real per-node PHY (link
closure + PV harvest) across a greenhouse floor lit by one ceiling luminaire.
A node at horizontal offset r under a luminaire at height h sees emission angle
phi = incidence angle psi = atan(r/h) and distance d = sqrt(h^2 + r^2) — exactly
the channel's tx_angle/rx_tilt/distance, with the FOV gate dropping nodes
outside the cone. So coverage is a 2-D position sweep over the existing engine.

`coverage_grid(...)` returns the per-position fields; the canvas/backend can call
it directly later. `python scripts/coverage_map.py` renders the figure.

Run:
    python scripts/coverage_map.py [--height 2.5] [--half-extent 2.0] [--grid 9]
                                   [--half-angle 40] [--power-mW 2000]
                                   [--ambient-lux 0] [--n-bits 1000]
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
from cosim.python_engine import run_python_simulation  # noqa: E402


def coverage_grid(base: SystemConfig, height_m: float, half_extent_m: float,
                  n: int, n_bits: int, half_angle_deg: float,
                  power_mW: float, ambient_lux: float,
                  r_sense_ohm: float) -> dict:
    """Run the per-node PHY across an n×n floor grid under one ceiling luminaire.

    Returns dict of (n,n) arrays: BER, harvest_uW, dist_m, angle_deg, plus the
    1-D axis `xs` (m). LED is above the floor centre at (0,0,height).
    """
    xs = np.linspace(-half_extent_m, half_extent_m, n)
    BER = np.full((n, n), np.nan)
    HAR = np.full((n, n), np.nan)
    DIST = np.zeros((n, n))
    ANG = np.zeros((n, n))

    d0 = base.to_dict()
    d0["simulation_engine"] = "python"
    d0["n_bits"] = int(n_bits)
    d0["led_half_angle_deg"] = float(half_angle_deg)
    d0["led_radiated_power_mW"] = float(power_mW)
    d0["ambient_illuminance_lux"] = float(ambient_lux)
    d0["r_sense_ohm"] = float(r_sense_ohm)  # deployment operating point

    for iy, y in enumerate(xs):
        for ix, x in enumerate(xs):
            r = math.hypot(x, y)
            d = math.hypot(r, height_m)
            ang = math.degrees(math.atan2(r, height_m))
            DIST[iy, ix] = d
            ANG[iy, ix] = ang
            cfg = SystemConfig(**{**d0, "distance_m": d,
                                  "tx_angle_deg": ang, "rx_tilt_deg": ang})
            try:
                res = run_python_simulation(cfg)
                BER[iy, ix] = float(res.get("ber", np.nan))
                vc = res.get("V_cell"); ic = res.get("I_cell")
                if vc is not None and ic is not None:
                    HAR[iy, ix] = float(np.mean(np.asarray(vc) * np.asarray(ic)) * 1e6)
            except Exception:  # noqa: BLE001 — one bad point shouldn't kill the map
                pass
    return {"xs": xs, "BER": BER, "harvest_uW": HAR, "dist_m": DIST,
            "angle_deg": ANG}


def plot(grid: dict, ber_thresh: float, outdir: Path,
         height_m: float, half_angle_deg: float,
         node_budget_uW: float, dcdc_eff: float) -> None:
    xs = grid["xs"]
    extent = [xs[0], xs[-1], xs[0], xs[-1]]
    ber = grid["BER"]; har = grid["harvest_uW"]
    closes = np.isfinite(ber) & (ber <= ber_thresh)
    usable = np.where(np.isfinite(har), har * dcdc_eff, 0.0)   # after DC-DC
    powered = usable >= node_budget_uW
    deployable = closes & powered
    frac = 100.0 * np.count_nonzero(closes) / closes.size
    dep_frac = 100.0 * np.count_nonzero(deployable) / deployable.size

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17, 5.3))

    # 1) BER coverage (log) + link-closure contour.
    berp = np.where(np.isfinite(ber), ber, 1.0)
    im0 = a0.imshow(berp, extent=extent, origin="lower", cmap="viridis_r",
                    norm=matplotlib.colors.LogNorm(vmin=max(berp[berp > 0].min(), 1e-6),
                                                   vmax=0.5), aspect="equal")
    a0.contour(xs, xs, closes.astype(float), levels=[0.5], colors="w", linewidths=2)
    a0.plot(0, 0, "r*", ms=13)
    a0.set_title(f"Link BER — closes on {frac:.0f}% of floor")
    a0.set_xlabel("x (m)"); a0.set_ylabel("y (m)")
    fig.colorbar(im0, ax=a0, label="BER")

    # 2) Harvest + the energy-budget contour.
    im1 = a1.imshow(har, extent=extent, origin="lower", cmap="inferno", aspect="equal")
    thr_raw = node_budget_uW / dcdc_eff  # raw harvest needed to meet the budget
    if np.nanmin(har) < thr_raw < np.nanmax(har):
        a1.contour(xs, xs, np.where(np.isfinite(har), har, 0.0),
                   levels=[thr_raw], colors="c", linewidths=2)
    a1.plot(0, 0, "c*", ms=13)
    a1.set_title(f"PV harvest (µW)\ncyan = {node_budget_uW:g}µW budget @ {dcdc_eff:.0%} DC-DC")
    a1.set_xlabel("x (m)"); a1.set_ylabel("y (m)")
    fig.colorbar(im1, ax=a1, label="harvest (µW)")

    # 3) Deployability zones: no link / linked-but-starved / deployable.
    cat = np.zeros_like(ber)
    cat[closes] = 1.0
    cat[deployable] = 2.0
    cmap = matplotlib.colors.ListedColormap(["#3b0f3f", "#f0a73a", "#1a9850"])
    im2 = a2.imshow(cat, extent=extent, origin="lower", cmap=cmap,
                    vmin=-0.5, vmax=2.5, aspect="equal")
    a2.plot(0, 0, "w*", ms=13)
    a2.set_title(f"Deployable region — {dep_frac:.0f}% of floor\n(link + self-powered)")
    a2.set_xlabel("x (m)"); a2.set_ylabel("y (m)")
    cb = fig.colorbar(im2, ax=a2, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["no link", "linked,\nstarved", "deployable"])

    fig.suptitle(f"Greenhouse coverage + autonomy — luminaire h={height_m} m, "
                 f"half-angle {half_angle_deg}°, node budget {node_budget_uW:g}µW",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = outdir / "coverage_map.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    print(f"\nLink closure:        {frac:.0f}% of floor (BER<={ber_thresh:g})")
    print(f"Self-powered region: {dep_frac:.0f}% of floor (>= {node_budget_uW:g}uW usable)")
    print(f"harvest range: {np.nanmin(har):.1f} - {np.nanmax(har):.1f} uW")
    print(f"  figure -> {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Greenhouse coverage map (Path C / M1)")
    p.add_argument("--height", type=float, default=2.5, help="luminaire height (m)")
    p.add_argument("--half-extent", type=float, default=2.0, help="floor half-size (m)")
    p.add_argument("--grid", type=int, default=9, help="grid points per axis")
    p.add_argument("--half-angle", type=float, default=40.0, help="LED half-angle (deg)")
    p.add_argument("--power-mW", type=float, default=8000.0, help="LED optical power (mW)")
    p.add_argument("--ambient-lux", type=float, default=0.0, help="ambient illuminance (lux)")
    p.add_argument("--r-sense", type=float, default=500.0,
                   help="node PV load operating point (ohm); 500 ~ best-data point")
    p.add_argument("--n-bits", type=int, default=1000)
    p.add_argument("--ber-thresh", type=float, default=1e-2)
    p.add_argument("--node-budget-uW", type=float, default=10.0,
                   help="node average power draw (uW) — MCU+radio+sensor")
    p.add_argument("--dcdc-eff", type=float, default=0.75, help="harvester DC-DC efficiency")
    p.add_argument("-o", "--output", default=str(ROOT / "workspace" / "slipt"))
    args = p.parse_args(argv)

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    base = SystemConfig.from_preset("slipt_node")
    print(f"Coverage grid {args.grid}x{args.grid} over {2*args.half_extent:.0f}x"
          f"{2*args.half_extent:.0f} m, luminaire h={args.height} m...")
    grid = coverage_grid(base, args.height, args.half_extent, args.grid,
                         args.n_bits, args.half_angle, args.power_mW, args.ambient_lux,
                         args.r_sense)
    plot(grid, args.ber_thresh, outdir, args.height, args.half_angle,
         args.node_budget_uW, args.dcdc_eff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Path C / M3 — end-to-end greenhouse latency & throughput budget.

Chains the fast optical edge tier (our PHY) onto the slow LoRa backhaul to
deliver a sensor reading to the dashboard, and sweeps the LoRa spreading factor
(SF7..SF12 — the range-vs-airtime knob). Shows where the bottleneck is.

  sensor --[ LiFi edge PHY ]--> aggregator --[ LoRa backhaul ]--> gateway/dashboard

End-to-end latency  = edge frame time + LoRa time-on-air (+ propagation).
Aggregate throughput = min(edge goodput, LoRa bit rate)  (the bottleneck tier).

Run:
    python scripts/e2e_budget.py [--payload-bytes 20] [--backhaul-m 300] [--tx-dBm 14]
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
from cosim.python_engine import run_python_simulation  # noqa: E402
from cosim.features import extract_features  # noqa: E402
from cosim.lora_budget import LoRaLink  # noqa: E402


def edge_tier(preset: str, payload_bytes: int) -> dict:
    """Run the optical edge PHY once; return goodput + per-payload frame latency."""
    cfg = SystemConfig.from_preset(preset)
    d = cfg.to_dict(); d["simulation_engine"] = "python"
    res = run_python_simulation(SystemConfig(**d))
    feat = extract_features(res, SystemConfig(**d))
    goodput = feat.get("goodput_bps") or feat.get("payload_rate_bps") or float("nan")
    dsp = feat.get("latency_dsp_s") or 0.0
    prop = feat.get("latency_propagation_s") or 0.0
    frame = (payload_bytes * 8) / goodput + dsp + prop
    return {"goodput_bps": goodput, "frame_s": frame, "dsp_s": dsp, "prop_s": prop}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="End-to-end greenhouse budget (Path C / M3)")
    p.add_argument("--edge-preset", default="lifi_compare", help="optical edge PHY preset")
    p.add_argument("--payload-bytes", type=int, default=20, help="sensor payload (bytes)")
    p.add_argument("--backhaul-m", type=float, default=300.0, help="LoRa range (m)")
    p.add_argument("--tx-dBm", type=float, default=14.0, help="LoRa TX power (dBm)")
    p.add_argument("--bw-khz", type=float, default=125.0, help="LoRa bandwidth (kHz)")
    p.add_argument("-o", "--output", default=str(ROOT / "workspace" / "slipt"))
    args = p.parse_args(argv)

    edge = edge_tier(args.edge_preset, args.payload_bytes)
    print(f"Edge tier ({args.edge_preset}): goodput {edge['goodput_bps']/1e6:.2f} Mbps, "
          f"frame {edge['frame_s']*1e3:.3f} ms for {args.payload_bytes} B")

    sfs = [7, 8, 9, 10, 11, 12]
    airtime, bitrate, margin, e2e_lat, bottleneck = [], [], [], [], []
    print(f"\nLoRa backhaul @ {args.backhaul_m:.0f} m, {args.tx_dBm:.0f} dBm, "
          f"BW {args.bw_khz:.0f} kHz, payload {args.payload_bytes} B:")
    print(f"{'SF':>3} {'airtime_ms':>10} {'bitrate_bps':>11} {'sens_dBm':>9} "
          f"{'margin_dB':>9} {'e2e_lat_ms':>10} {'bottleneck_bps':>14}")
    for sf in sfs:
        link = LoRaLink(sf=sf, bw_hz=args.bw_khz * 1e3)
        toa = link.airtime_s(args.payload_bytes)
        br = link.bitrate_bps()
        mg = link.link_margin_dB(args.tx_dBm, args.backhaul_m)
        lat = edge["frame_s"] + toa
        bn = min(edge["goodput_bps"], br)
        airtime.append(toa * 1e3); bitrate.append(br); margin.append(mg)
        e2e_lat.append(lat * 1e3); bottleneck.append(bn)
        print(f"{sf:>3} {toa*1e3:>10.1f} {br:>11.1f} {link.sensitivity_dBm():>9.1f} "
              f"{mg:>9.1f} {lat*1e3:>10.1f} {bn:>14.1f}")

    # ---- figure ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.2))
    sf_arr = np.array(sfs)
    closes = np.array(margin) > 0

    a0.bar(sf_arr, e2e_lat, color=np.where(closes, "#1a9850", "#c0392b"))
    a0.axhline(edge["frame_s"] * 1e3, color="#1f77b4", ls="--",
               label=f"edge frame ({edge['frame_s']*1e3:.2g} ms)")
    a0.set_yscale("log")
    a0.set_xlabel("LoRa spreading factor"); a0.set_ylabel("end-to-end latency (ms)")
    a0.set_title(f"E2E latency per reading ({args.payload_bytes} B)\n"
                 f"green = backhaul closes @ {args.backhaul_m:.0f} m")
    a0.legend(fontsize=8); a0.grid(True, which="both", axis="y", alpha=0.3)

    a1.plot(sf_arr, [edge["goodput_bps"]] * len(sfs), "b--o", label="LiFi edge goodput")
    a1.plot(sf_arr, bitrate, "s-", color="#e67e22", label="LoRa backhaul rate")
    a1.plot(sf_arr, bottleneck, "k*-", ms=11, label="end-to-end bottleneck")
    a1.set_yscale("log")
    a1.set_xlabel("LoRa spreading factor"); a1.set_ylabel("throughput (bit/s)")
    a1.set_title("Throughput per tier — LoRa is the bottleneck")
    a1.legend(fontsize=8); a1.grid(True, which="both", axis="y", alpha=0.3)

    fig.suptitle("End-to-end greenhouse budget: LiFi edge + LoRa backhaul", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fpath = out / "e2e_budget.png"
    fig.savefig(fpath, dpi=160); plt.close(fig)
    max_sf = sfs[max([i for i, c in enumerate(closes)], default=0)] if closes.any() else None
    print(f"\nBackhaul closes up to SF{max_sf} at {args.backhaul_m:.0f} m" if max_sf
          else "\nBackhaul does NOT close at this range/power")
    print(f"  figure -> {fpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

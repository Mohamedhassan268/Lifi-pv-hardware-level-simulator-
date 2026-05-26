"""
WebSocket /ws/sweep — stream parameter-sweep results live as each point completes.

Protocol
========
Client → server:
    {"type": "start", "kind": "thermal", "preset": "kadirvelu2021",
     "params": {"temps_C": [-10, 0, 25, 45, 65, 85]}}

    {"type": "start", "kind": "montecarlo", "preset": "kadirvelu2021",
     "params": {"n_runs": 50, "tolerance_spec": {"sc_responsivity": 0.05}, "seed": 42}}

    {"type": "cancel"}

Server → client:
    {"type": "started", "kind": "...", "total": 50}
    {"type": "point", "i": 0, "total": 50, "data": {...one row...}}
    {"type": "done", "kind": "...", "count": 50}
    {"type": "error", "message": "..."}

Why generators
==============
The existing run_thermal_sweep() / run_monte_carlo() return a full list at
the end — they can't drive a live UI. We replicate their inner loop here as
a Python generator that yields one ThermalPoint / one MC sample per iteration.
The same simulation calls are reused; only the loop body is mirrored so we
can stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from typing import Any, Iterator

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from cosim.ber_sweep import stream_ber_vs_distance, stream_ber_vs_snr
from cosim.noise import NoiseModel
from cosim.python_engine import run_python_simulation
from cosim.system_config import SystemConfig
from cosim.tolerance import derive_tolerance_spec, describe_tolerance_spec

router = APIRouter()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Streaming generators (mirror cosim/thermal_sweep.py + cosim/monte_carlo.py)
# -----------------------------------------------------------------------------

def stream_thermal(preset: str, temps_C: list[float],
                   cancel: threading.Event) -> Iterator[dict[str, Any]]:
    """Yield one dict per temperature. Mirrors run_thermal_sweep()."""
    for T_C in temps_C:
        if cancel.is_set():
            return
        T_K = float(T_C) + 273.15
        cfg = SystemConfig.from_preset(preset)
        cfg.temperature_K = T_K

        nm = NoiseModel.from_config(cfg)
        I_dark = nm.dark_current_at_T()

        res = run_python_simulation(cfg)
        P_rx_avg_W = res.get("P_rx_avg_uW", 0.0) * 1e-6
        I_ph_avg_A = res.get("I_ph_avg_uA", 0.0) * 1e-6
        R_eff = I_ph_avg_A / P_rx_avg_W if P_rx_avg_W > 0 else 0.0

        yield {
            "temperature_C": float(T_C),
            "temperature_K": T_K,
            "dark_current_A": float(I_dark),
            "responsivity_A_per_W": float(R_eff),
            "I_ph_avg_uA": float(res.get("I_ph_avg_uA", 0.0)),
            "snr_dB": float(res.get("snr_est_dB", 0.0)),
            "ber": float(res.get("ber", 1.0)),
        }


def _sample_param(nominal: float, tol: float, rng: np.random.Generator) -> float:
    lo = nominal * (1.0 - tol)
    hi = nominal * (1.0 + tol)
    return float(rng.uniform(lo, hi))


def stream_monte_carlo(preset: str, tolerance_spec: dict[str, float],
                       n_runs: int, seed: int | None,
                       cancel: threading.Event) -> Iterator[dict[str, Any]]:
    """Yield one dict per Monte Carlo trial. Mirrors run_monte_carlo()."""
    rng = np.random.default_rng(seed)
    baseline = SystemConfig.from_preset(preset)

    for fname in tolerance_spec:
        if not hasattr(baseline, fname):
            raise ValueError(f"Unknown field '{fname}'")

    nominals = {fname: float(getattr(baseline, fname)) for fname in tolerance_spec}

    for i in range(n_runs):
        if cancel.is_set():
            return
        overrides = {
            fname: _sample_param(nominals[fname], tol, rng)
            for fname, tol in tolerance_spec.items()
        }
        cfg = baseline.replace(**overrides)
        try:
            res = run_python_simulation(cfg)
            ber = float(res.get("ber", 1.0))
            snr = float(res.get("snr_est_dB", 0.0))
            p_rx = float(res.get("P_rx_avg_uW", 0.0))
        except Exception:  # noqa: BLE001
            ber, snr, p_rx = 1.0, 0.0, 0.0

        yield {
            "i": i,
            "ber": ber,
            "snr_dB": snr,
            "P_rx_avg_uW": p_rx,
            "overrides": overrides,
        }


# -----------------------------------------------------------------------------
# WebSocket endpoint
# -----------------------------------------------------------------------------

def _clean(v: Any) -> Any:
    """Strip NaN/inf so json.dumps emits valid JSON."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


@router.websocket("/ws/sweep")
async def sweep_ws(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    cancel = threading.Event()

    async def send(payload: dict) -> None:
        await ws.send_text(json.dumps(_clean(payload)))

    def send_from_thread(payload: dict) -> None:
        fut = asyncio.run_coroutine_threadsafe(send(payload), loop)
        try:
            fut.result(timeout=2.0)
        except Exception as e:  # noqa: BLE001
            logger.warning("sweep send failed: %s", e)

    async def listen_for_cancel() -> None:
        """Parallel coroutine: watch for cancel frames while the sweep runs."""
        try:
            while not cancel.is_set():
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "cancel":
                    cancel.set()
                    return
        except WebSocketDisconnect:
            cancel.set()

    try:
        # First message must be `start`.
        raw = await ws.receive_text()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await send({"type": "error", "message": "invalid json"})
            return

        if msg.get("type") != "start":
            await send({"type": "error", "message": "first message must be start"})
            return

        kind = msg.get("kind")
        preset = msg.get("preset") or "kadirvelu2021"
        params = msg.get("params") or {}

        if kind == "thermal":
            temps = list(params.get("temps_C") or [-10, 0, 25, 45, 65, 85])
            total = len(temps)
            await send({"type": "started", "kind": "thermal", "total": total})

            def run_thermal() -> None:
                try:
                    for i, point in enumerate(stream_thermal(preset, temps, cancel)):
                        send_from_thread({"type": "point", "i": i, "total": total, "data": point})
                    send_from_thread({"type": "done", "kind": "thermal", "count": total})
                except Exception as e:  # noqa: BLE001
                    send_from_thread({"type": "error", "message": str(e)})

            task = asyncio.gather(
                listen_for_cancel(),
                loop.run_in_executor(None, run_thermal),
            )
            await task

        elif kind == "montecarlo":
            n_runs = int(params.get("n_runs", 50))
            auto_tol = bool(params.get("auto_tolerance", False))
            seed = params.get("seed")
            if auto_tol:
                baseline = SystemConfig.from_preset(preset)
                tolerance_spec = derive_tolerance_spec(baseline)
                # Send the resolved table to the client so the UI can display
                # which fields are being varied and from which component.
                await send({
                    "type": "tolerance_resolved",
                    "rows": describe_tolerance_spec(baseline),
                })
            else:
                tolerance_spec = dict(
                    params.get("tolerance_spec") or {"sc_responsivity": 0.05}
                )
            await send({"type": "started", "kind": "montecarlo", "total": n_runs})

            def run_mc() -> None:
                try:
                    for i, sample in enumerate(
                        stream_monte_carlo(preset, tolerance_spec, n_runs, seed, cancel)
                    ):
                        send_from_thread({"type": "point", "i": i, "total": n_runs, "data": sample})
                    send_from_thread({"type": "done", "kind": "montecarlo", "count": n_runs})
                except Exception as e:  # noqa: BLE001
                    send_from_thread({"type": "error", "message": str(e)})

            task = asyncio.gather(
                listen_for_cancel(),
                loop.run_in_executor(None, run_mc),
            )
            await task

        elif kind in ("ber_snr", "ber_distance"):
            n_pts = int(params.get("n_points", 12))
            if kind == "ber_snr":
                lo = float(params.get("ambient_lux_lo", 1.0))
                hi = float(params.get("ambient_lux_hi", 10000.0))
                gen = lambda: stream_ber_vs_snr(preset, lo, hi, n_pts, cancel)  # noqa: E731
            else:
                lo = float(params.get("distance_lo_m", 0.1))
                hi = float(params.get("distance_hi_m", 10.0))
                gen = lambda: stream_ber_vs_distance(preset, lo, hi, n_pts, cancel)  # noqa: E731
            await send({"type": "started", "kind": kind, "total": n_pts})

            def run_ber() -> None:
                try:
                    for i, sample in enumerate(gen()):
                        send_from_thread({"type": "point", "i": i, "total": n_pts, "data": sample})
                    send_from_thread({"type": "done", "kind": kind, "count": n_pts})
                except Exception as e:  # noqa: BLE001
                    send_from_thread({"type": "error", "message": str(e)})

            task = asyncio.gather(
                listen_for_cancel(),
                loop.run_in_executor(None, run_ber),
            )
            await task

        else:
            await send({"type": "error", "message": f"unknown kind: {kind}"})

    except WebSocketDisconnect:
        cancel.set()
        logger.info("sweep_ws: client disconnected")
    except Exception as e:  # noqa: BLE001
        logger.exception("sweep_ws: unexpected error")
        try:
            await send({"type": "error", "message": str(e)})
        except Exception:  # noqa: BLE001
            pass

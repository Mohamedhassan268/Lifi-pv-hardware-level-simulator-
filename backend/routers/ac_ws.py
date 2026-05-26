"""
WebSocket /ws/ac — small-signal frequency response of the RX chain.

Protocol
========
Client → server:
    {"type": "start", "config": {...flat SystemConfig dict...},
     "params": {"f_start_hz": 1, "f_stop_hz": 1e7, "n_points": 400}}

Server → client:
    {"type": "started", "n_points": 400}
    {"type": "ac_result",
        "frequency_hz": [...], "magnitude_db": [...], "phase_deg": [...],
        "gain_margin_db": 18.2, "phase_margin_deg": 62.0,
        "unity_gain_freq_hz": 14e3, "crossover_180_freq_hz": 35e3,
        "f_low_3dB_hz": 700, "f_high_3dB_hz": 9.5e3,
        "midband_gain_db": 41.2}
    {"type": "done"}
    {"type": "error", "message": "..."}

The run is fast (<50 ms for N=400) so we send one batch payload, not
per-point streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from cosim.ac_analysis import run_ac_analysis
from cosim.system_config import SystemConfig

router = APIRouter()
logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ac")


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


@router.websocket("/ws/ac")
async def ac_ws(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()

    async def send(payload: dict) -> None:
        await ws.send_text(json.dumps(_clean(payload)))

    try:
        raw = await ws.receive_text()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await send({"type": "error", "message": "invalid json"})
            return

        if msg.get("type") != "start":
            await send({"type": "error", "message": "first message must be start"})
            return

        config_dict = msg.get("config") or {}
        params = msg.get("params") or {}
        try:
            cfg = SystemConfig._from_flat_dict(config_dict)
        except ValueError as e:
            await send({"type": "error", "message": f"invalid config: {e}"})
            return

        f_start = float(params.get("f_start_hz", 1.0))
        f_stop = float(params.get("f_stop_hz", 1e7))
        n_points = int(params.get("n_points", 400))
        await send({"type": "started", "n_points": n_points})

        def run_blocking() -> dict:
            return run_ac_analysis(cfg, f_start, f_stop, n_points).as_dict()

        try:
            result = await loop.run_in_executor(_EXECUTOR, run_blocking)
        except Exception as e:  # noqa: BLE001
            await send({"type": "error", "message": f"ac analysis crashed: {e}"})
            return

        payload = {"type": "ac_result", **result}
        await send(payload)
        await send({"type": "done"})

    except WebSocketDisconnect:
        logger.info("ac_ws: client disconnected")
    except Exception as e:  # noqa: BLE001
        logger.exception("ac_ws: unexpected error")
        try:
            await send({"type": "error", "message": str(e)})
        except Exception:  # noqa: BLE001
            pass

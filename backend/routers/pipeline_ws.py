"""
WebSocket /ws/pipeline — run a TX→Channel→RX simulation and stream progress.

Protocol (Phase 8: glass-box observability)
===========================================
Client → server (JSON text frames):
    {"type": "start", "mode": "continuous" | "stepwise",
     "config": {... flat SystemConfig dict ...}, "profile_id": "...",
     "capture_probes": "all" | [probe_id, ...] | null}
    {"type": "step"}                 # only valid while paused in stepwise mode
    {"type": "resume"}               # alias for step
    {"type": "probe_fetch", "id": "rx.V_ina", "request_id": 17}
    {"type": "cancel"}

Server → client (JSON text frames except probe_data which is BINARY):
    {"type": "step", "name": "TX"|"Channel"|"RX",
     "status": "running"|"done"|"error", "message": "...", "duration_s": 0.12}
    {"type": "paused", "after": "TX"|"Channel"|"RX",
     "available_probes": ["tx.bits","tx.P_tx",...]}
    {"type": "probes_available", "stage": "...", "probe_ids": [...]}
    {"type": "metrics", "ber": ..., "snr_est_dB": ..., "noise_breakdown": {...}}
    {"type": "waveforms", ...}      # backward-compat subsampled JSON view
    {"type": "compliance", ...}
    {"type": "done", "engine": "...", "session_dir": "..."}
    {"type": "error", "message": "..."}
    <binary> probe_data frame       # see backend.transport.binary_frame

Notes
=====
The existing SimulationPipeline.on_progress callback is synchronous and runs
on the executor thread; events are marshalled back onto the FastAPI event
loop via `asyncio.run_coroutine_threadsafe`. The stepwise pause is implemented
by a `stage_hook` plumbed all the way down into `run_python_simulation`; the
hook blocks the executor thread on a `threading.Event` that is set when the
event loop receives `{"type":"step"}`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.spice_waveforms import build_waveforms_from_spice
from backend.transport.binary_frame import pack_probe
from cosim.noise import NoiseModel
from cosim.pipeline import SimulationPipeline
from cosim.probes import (
    PROBE_REGISTRY,
    ProbeCapture,
    config_hash,
    probes_for_stage,
)
from cosim.session import SessionManager
from cosim.standards import evaluate as evaluate_compliance
from cosim.standards import get_profile
from cosim.system_config import SystemConfig

router = APIRouter()
logger = logging.getLogger(__name__)

# One simulation per WebSocket; a small pool covers concurrent clients without
# letting them queue indefinitely against each other.
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")

# Hard cap for the legacy "waveforms" JSON payload (preserved for the
# existing nine Plotly result panels). Binary probe_data frames are not
# subsampled — they ship the full Float64Array.
_MAX_WAVEFORM_POINTS = 4000

# Stages we know how to pause after. Order matters for protocol clarity.
_STAGES_ORDER = ("TX", "Channel", "RX")


def _subsample(arr, max_points: int = _MAX_WAVEFORM_POINTS) -> list[float]:
    """Uniform stride subsample for the legacy JSON waveforms payload."""
    if arr is None:
        return []
    a = np.asarray(arr).ravel()
    if a.size == 0:
        return []
    if a.size <= max_points:
        return [float(x) for x in a.tolist()]
    stride = math.ceil(a.size / max_points)
    return [float(x) for x in a[::stride].tolist()]


def _clean_for_json(obj: Any) -> Any:
    """Strip NaN/inf so json.dumps doesn't produce invalid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    return obj


@router.websocket("/ws/pipeline")
async def pipeline_ws(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    cancel_flag = {"cancelled": False}

    async def send_json(payload: dict) -> None:
        await ws.send_text(json.dumps(_clean_for_json(payload)))

    async def send_bytes(frame: bytes) -> None:
        await ws.send_bytes(frame)

    def send_from_thread(payload: dict) -> None:
        """Thread-safe bridge for control/JSON messages from the executor."""
        fut = asyncio.run_coroutine_threadsafe(send_json(payload), loop)
        try:
            fut.result(timeout=2.0)
        except Exception as e:  # noqa: BLE001 - non-fatal; pipeline keeps going
            logger.warning("send_from_thread failed: %s", e)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send_json({"type": "error", "message": "invalid json"})
                continue

            mtype = msg.get("type")

            if mtype == "cancel":
                cancel_flag["cancelled"] = True
                await send_json({"type": "step", "name": "RX", "status": "error",
                                 "message": "cancelled by client"})
                continue

            if mtype != "start":
                await send_json({"type": "error", "message": f"unknown type: {mtype}"})
                continue

            config_dict = msg.get("config") or {}
            profile_id = msg.get("profile_id")
            mode = msg.get("mode", "continuous")
            capture_probes = msg.get("capture_probes")  # None | "all" | list

            try:
                cfg = SystemConfig._from_flat_dict(config_dict)
            except ValueError as e:
                await send_json({"type": "error", "message": f"invalid config: {e}"})
                continue

            # ---- Set up probe capture for this run ----
            run_hash = config_hash(cfg)
            if capture_probes == "all":
                enabled_ids: Optional[list] = None
            elif isinstance(capture_probes, list):
                enabled_ids = list(capture_probes)
            else:
                # Stepwise mode implies probe capture is useful even if the
                # client didn't ask — pause-without-probes is useless.
                enabled_ids = None if mode == "stepwise" else []

            session_dir = SessionManager().create_session(label="ws")
            probes = ProbeCapture(
                session_dir=session_dir,
                config_hash=run_hash,
                enabled_ids=enabled_ids,
            )

            # ---- Stepwise pause coordination ----
            step_event = threading.Event()
            stepwise = (mode == "stepwise")
            pause_state = {"after": None}  # which stage we're paused after

            def stage_hook(stage_name: str) -> None:
                """Called from the executor thread after each block completes."""
                if not stepwise or cancel_flag["cancelled"]:
                    return
                # Emit "paused" event listing available probes for the stage.
                available = sorted(probes.available_ids(stage=stage_name))
                pause_state["after"] = stage_name
                send_from_thread({
                    "type": "paused",
                    "after": stage_name,
                    "available_probes": available,
                })
                step_event.clear()
                # Block until the WS handler signals resume (or disconnect).
                step_event.wait()
                pause_state["after"] = None

            # ---- Per-stage progress callback (unchanged) ----
            def on_progress(step: str, status: str, message: str) -> None:
                send_from_thread({
                    "type": "step",
                    "name": step,
                    "status": status,
                    "message": message,
                })
                # When a block completes, also publish its probe inventory so
                # the frontend can light up the edge.
                if status == "done":
                    pids = sorted(probes.available_ids(stage=step))
                    if pids:
                        send_from_thread({
                            "type": "probes_available",
                            "stage": step,
                            "probe_ids": pids,
                        })

            pipeline = SimulationPipeline(
                cfg, session_dir,
                on_progress=on_progress,
                probes=probes,
                stage_hook=stage_hook,
            )

            def run_blocking() -> dict:
                pipeline.run_all()
                return {
                    "tx": pipeline.step_tx,
                    "channel": pipeline.step_channel,
                    "rx": pipeline.step_rx,
                }

            # ---- Drive the simulation while handling step/probe_fetch messages ----
            run_task = loop.run_in_executor(_EXECUTOR, run_blocking)

            async def control_loop() -> None:
                """While the pipeline runs, handle inbound control messages."""
                while not run_task.done():
                    try:
                        raw_msg = await asyncio.wait_for(ws.receive_text(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    except WebSocketDisconnect:
                        cancel_flag["cancelled"] = True
                        step_event.set()  # unblock executor on disconnect
                        return

                    try:
                        m = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        await send_json({"type": "error", "message": "invalid json"})
                        continue

                    t = m.get("type")
                    if t in ("step", "resume"):
                        step_event.set()
                    elif t == "cancel":
                        cancel_flag["cancelled"] = True
                        step_event.set()
                    elif t == "probe_fetch":
                        await _serve_probe_fetch(m, probes, run_hash, send_bytes, send_json)
                    else:
                        await send_json({"type": "error",
                                         "message": f"unknown msg while running: {t}"})

            control_task = asyncio.create_task(control_loop())

            try:
                results = await run_task
            except Exception as e:  # noqa: BLE001
                control_task.cancel()
                await send_json({"type": "error", "message": f"pipeline crashed: {e}"})
                continue
            finally:
                control_task.cancel()
                try:
                    await control_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

            # Persist captured probes to disk for the verify step / replay.
            try:
                probes.save()
            except Exception as e:  # noqa: BLE001 - non-fatal
                logger.warning("probe save failed: %s", e)

            rx = results["rx"]
            if rx.status != "done":
                await send_json({"type": "error", "message": rx.message or "pipeline failed"})
                continue

            # ---- Metrics ----
            metrics: dict[str, Any] = {
                "type": "metrics",
                "engine": rx.outputs.get("engine"),
                "ber": rx.outputs.get("ber"),
                "ber_n_errors": rx.outputs.get("ber_n_errors"),
                "ber_n_bits": rx.outputs.get("ber_n_bits"),
                "snr_est_dB": rx.outputs.get("snr_est_dB"),
                "P_rx_avg_uW": results["channel"].outputs.get("P_rx_avg_uW"),
                "I_ph_avg_uA": results["channel"].outputs.get("I_ph_avg_uA"),
                "G_ch": results["channel"].outputs.get("G_ch"),
                "config_hash": run_hash,
            }

            # ---- Per-source noise breakdown (Phase 1A) ----
            try:
                I_ph_uA = results["channel"].outputs.get("I_ph_avg_uA") or 0.0
                I_ph_A = float(I_ph_uA) * 1e-6
                bandwidth_hz = float(getattr(cfg, "bpf_f_high_Hz", 1.0) or 1.0)
                breakdown = NoiseModel.from_config(cfg).compute_noise(
                    I_ph=I_ph_A, bandwidth=bandwidth_hz,
                )
                metrics["noise_breakdown"] = breakdown.as_dict()
            except Exception as e:  # noqa: BLE001 — non-fatal, keep streaming
                logger.warning("noise breakdown failed: %s", e)

            await send_json(metrics)

            # ---- Compliance ----
            if profile_id:
                try:
                    profile = get_profile(profile_id)
                    metrics_for_eval: dict[str, Any] = {
                        "ber": metrics.get("ber"),
                        "snr_est_dB": metrics.get("snr_est_dB"),
                        "modulation": getattr(cfg, "modulation", None),
                    }
                    comp_results = evaluate_compliance(profile, metrics_for_eval)
                    await send_json({
                        "type": "compliance",
                        "profile_id": profile.id,
                        "profile_label": profile.label,
                        "spec_section": profile.spec_section,
                        "passed": all(r.ok for r in comp_results),
                        "results": [
                            {
                                "label": r.label, "ok": r.ok, "actual": r.actual,
                                "bound": r.bound, "detail": r.detail,
                            }
                            for r in comp_results
                        ],
                    })
                except KeyError as e:
                    await send_json({"type": "error", "message": str(e)})

            # ---- Legacy waveforms (kept for existing Plotly panels) ----
            py_result = rx.outputs.get("python_result")
            if not py_result:
                raw_path = rx.outputs.get("raw_file")
                if raw_path:
                    py_result = build_waveforms_from_spice(
                        raw_path,
                        bits_tx=getattr(pipeline, "_tx_bits", None),
                        bits_rx=rx.outputs.get("bits_rx"),
                        cfg=cfg,
                    )

            if py_result:
                payload = {
                    "type": "waveforms",
                    "time": _subsample(py_result.get("time")),
                    "P_tx": _subsample(py_result.get("P_tx")),
                    "P_rx": _subsample(py_result.get("P_rx")),
                    "I_ph": _subsample(py_result.get("I_ph")),
                    "V_rx": _subsample(py_result.get("V_rx")),
                    "bits_tx": _subsample(py_result.get("bits_tx"), max_points=512),
                    "bits_rx": _subsample(py_result.get("bits_rx"), max_points=512),
                    "modulation": py_result.get("modulation"),
                }
                # Constellation payload (ConstellationPanel.tsx renders this).
                # OFDM: complex post-FFT symbols -> ship as [real[], imag[]].
                # BFSK: (E0, E1) Goertzel-energy pairs per bit.
                # Cap point count for SVG-scatter perf. Ceil-div so e.g.
                # 4000 -> stride 2 (gives 2000), not stride 1 (no-op).
                _CONSTELL_MAX = 2048
                ofdm_syms = py_result.get("ofdm_symbols") or []
                if ofdm_syms:
                    if len(ofdm_syms) > _CONSTELL_MAX:
                        stride = (len(ofdm_syms) + _CONSTELL_MAX - 1) // _CONSTELL_MAX
                        ofdm_syms = ofdm_syms[::stride]
                    payload["ofdm_iq_real"] = [float(s.real) for s in ofdm_syms]
                    payload["ofdm_iq_imag"] = [float(s.imag) for s in ofdm_syms]
                bfsk_e = py_result.get("bfsk_energies") or []
                if bfsk_e:
                    if len(bfsk_e) > _CONSTELL_MAX:
                        stride = (len(bfsk_e) + _CONSTELL_MAX - 1) // _CONSTELL_MAX
                        bfsk_e = bfsk_e[::stride]
                    payload["bfsk_e0"] = [float(p[0]) for p in bfsk_e]
                    payload["bfsk_e1"] = [float(p[1]) for p in bfsk_e]
                await send_json(payload)

            await send_json({
                "type": "done",
                "engine": rx.outputs.get("engine"),
                "session_dir": str(session_dir),
                "config_hash": run_hash,
                "available_probes": sorted(probes.available_ids()),
            })

    except WebSocketDisconnect:
        logger.info("pipeline_ws: client disconnected")
    except Exception as e:  # noqa: BLE001
        logger.exception("pipeline_ws: unexpected error")
        try:
            await send_json({"type": "error", "message": str(e)})
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# probe_fetch handler
# =============================================================================

async def _serve_probe_fetch(msg: dict, probes: ProbeCapture, run_hash: str,
                             send_bytes, send_json) -> None:
    """Serve a {type: probe_fetch} request with a binary probe_data frame."""
    probe_id = msg.get("id")
    request_id = int(msg.get("request_id", 0))

    if not probe_id or probe_id not in PROBE_REGISTRY:
        await send_json({
            "type": "error",
            "message": f"unknown probe id: {probe_id}",
            "request_id": request_id,
        })
        return

    if not probes.has(probe_id):
        await send_json({
            "type": "error",
            "message": f"probe not captured: {probe_id}",
            "request_id": request_id,
        })
        return

    spec = PROBE_REGISTRY[probe_id]
    value = probes.get(probe_id)

    if spec.kind == "scalar":
        # Scalars stay JSON — no point with a binary frame.
        await send_json({
            "type": "probe_data",
            "id": probe_id,
            "kind": "scalar",
            "value": float(value),
            "units": spec.units,
            "stage": spec.stage,
            "request_id": request_id,
            "config_hash": run_hash,
        })
        return

    frame = pack_probe(
        probe_id,
        np.asarray(value),
        units=spec.units,
        config_hash=run_hash,
        request_id=request_id,
        stage=spec.stage,
    )
    await send_bytes(frame)

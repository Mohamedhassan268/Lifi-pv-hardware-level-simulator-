"""POST /api/link-budget — scalar link-budget calc for the Setup view sliders."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException

from backend.schemas import LinkBudgetRequest, LinkBudgetResponse
from cosim.channel import OpticalChannel

router = APIRouter()


@router.post("", response_model=LinkBudgetResponse)
def compute(req: LinkBudgetRequest) -> LinkBudgetResponse:
    try:
        channel = OpticalChannel(
            distance_m=req.distance_m,
            led_half_angle_deg=req.led_half_angle_deg,
            rx_area_cm2=req.sc_area_cm2,
            tx_angle_deg=req.tx_angle_deg,
            rx_tilt_deg=req.rx_tilt_deg,
            fov_half_angle_deg=req.fov_half_angle_deg,
            beer_lambert_enabled=req.beer_lambert_enabled,
            humidity_rh=req.humidity_rh,
            temperature_K=req.temperature_K,
            n_reflections=req.n_reflections,
            room_dims_m=(req.room_length_m, req.room_width_m, req.room_height_m),
            wall_reflectivity=req.wall_reflectivity,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(e)) from e

    # OpticalChannel exposes a single-sample propagate(P_tx). Use it on the LED
    # radiated power to get the steady-state P_rx, which is the standard
    # link-budget answer.
    p_tx_W = req.led_radiated_power_mW * 1e-3
    p_rx_W = float(channel.propagate(p_tx_W))
    gain = p_rx_W / p_tx_W if p_tx_W > 0 else 0.0
    i_ph = req.sc_responsivity * p_rx_W

    # SNR estimate (mirrors SystemConfig.snr_estimate_dB so the GUI matches).
    q = 1.602e-19
    k_T = 1.38e-23 * 300.0
    bw = req.data_rate_bps / 2.0
    i_signal = i_ph * req.modulation_depth
    n_shot = math.sqrt(2 * q * max(i_ph, 0.0) * bw)
    n_thermal = math.sqrt(4 * k_T * bw / max(req.r_sense_ohm, 1e-6))
    n_total = math.sqrt(n_shot * n_shot + n_thermal * n_thermal)
    snr_dB = 20.0 * math.log10(i_signal / n_total) if n_total > 0 and i_signal > 0 else 200.0

    return LinkBudgetResponse(
        channel_gain=gain,
        p_rx_W=p_rx_W,
        p_rx_uW=p_rx_W * 1e6,
        i_ph_A=i_ph,
        i_ph_uA=i_ph * 1e6,
        lambertian_order=float(channel.m_lambertian),
        snr_estimate_dB=snr_dB,
    )

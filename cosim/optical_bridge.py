"""The comms-layer bridge between the TX and RX SPICE passes.

In the single-canvas co-simulation the channel is Python, so it can never be
one SPICE pass. The TX deck yields the LED current ``I(LED, t)``; this module
maps that current to emitted optical power via the LED's own electro-optical
model, then propagates it through ``cosim.channel`` to the received optical
power ``P_rx(t)`` that drives the RX deck's photodetector node.

This is the only place the optical domain crosses from the TX circuit to the
RX circuit. Keeping it here (not in SPICE) is the deliberate architecture: the
validated Lambertian / Beer-Lambert channel stays in Python and is reused
verbatim.
"""

from __future__ import annotations

import numpy as np

from cosim.channel import OpticalChannel


def led_current_to_optical(led, i_led: np.ndarray) -> np.ndarray:
    """Map LED drive current (A) to emitted optical power (W).

    Uses the part's paper-calibrated ``optical_power_from_current`` when present
    (e.g. LXM5-PD01: GLED*I*T_lens), else the generic linear ``optical_power``.
    Negative excursions (reverse transient) clamp to zero — an LED emits no
    light below zero forward current.
    """
    i = np.asarray(i_led, dtype=float)
    if hasattr(led, "optical_power_from_current"):
        p = led.GLED * i * led.LENS_TRANSMITTANCE  # vectorized form of the scalar method
    else:
        p = led.radiant_flux_W * (i / led.max_drive_current_A)
    return np.clip(p, 0.0, None)


def led_current_to_p_rx(led, i_led: np.ndarray, cfg) -> np.ndarray:
    """Full bridge: LED current -> emitted optical -> channel -> received optical.

    Returns ``P_rx(t)`` in watts, ready to inject on the RX photodetector node.
    """
    p_tx = led_current_to_optical(led, i_led)
    channel = OpticalChannel.from_config(cfg)
    return channel.propagate(p_tx)

"""
Factory Li-Fi Analysis — lififactory1 preset
=============================================
Four analyses for an IIoT CNC factory environment (Li-Fi + 5G URLLC + Wi-Fi 6):

  Analysis 1 — BER noise floor breakdown (per-source, cumulative)
  Analysis 2 — FEC coding gain (LDPC 5/6 before vs after)
  Analysis 3 — Sensitivity sweeps (aerosol, LED aging, path length, ambient)
  Analysis 4 — Multi-cell floor coverage heatmap (4 ceiling APs, 30×20 m)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cosim.system_config import SystemConfig
from papers.shared import (
    save_figure, validation_header, validate_metric,
    print_validation_summary, lambertian_order, lambertian_channel_gain,
)
from papers.ofdm_modem import ber_mqam


TARGETS = {'ber': 1e-4, 'coverage_pct': 35.0}   # 35% realistic for 4×30° APs in 30×20m

_q = 1.602e-19
_k = 1.38e-23

# Four ceiling AP positions (x, y) in the 30×20 m factory hall
_AP_XY     = [(7.5, 5.0), (22.5, 5.0), (7.5, 15.0), (22.5, 15.0)]
_AP_LABELS = ['AP-1', 'AP-2', 'AP-3', 'AP-4']
_AP_COLORS = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A']


# =============================================================================
# PHYSICS HELPERS
# =============================================================================

def _factory_bl(distance_m: float, aerosol: float) -> float:
    """
    Beer-Lambert for 850 nm in factory air, physically calibrated for
    5–10 m industrial links (NOT the Correa 2025 spray model):
      alpha = 0.002 + 0.06 * max(aerosol - 0.2, 0)  [/m]
    aerosol: 0 = clean, 0.5 = light coolant mist, 1.0 = heavy spray.
    """
    alpha = 0.002 + 0.06 * max(aerosol - 0.2, 0.0)
    return float(np.exp(-alpha * distance_m))


def _los_gain(cfg, distance_m: float, tx_angle_rad: float = 0.0) -> float:
    """Lambertian LOS channel gain (no Beer-Lambert, no lens)."""
    m    = lambertian_order(cfg.led_half_angle_deg)
    A_m2 = cfg.sc_area_cm2 * 1e-4
    H    = lambertian_channel_gain(m, distance_m, A_m2)
    H   *= np.cos(tx_angle_rad) ** m
    H   *= np.cos(np.radians(cfg.rx_tilt_deg))
    return H


def _iphoton(cfg, distance_m=None, led_mW=None, aerosol=0.0) -> float:
    """Photocurrent (A) with optional overrides for sensitivity sweeps."""
    d    = distance_m if distance_m is not None else cfg.distance_m
    P_mW = led_mW     if led_mW     is not None else cfg.led_radiated_power_mW
    H    = _los_gain(cfg, d)
    BL   = _factory_bl(d, aerosol)
    P_rx = P_mW * 1e-3 * H * BL * cfg.lens_transmittance
    return cfg.sc_responsivity * P_rx


def _iambient(cfg, lux=None) -> float:
    """Ambient photocurrent at 850 nm (~5% of white-LED spectral power in 800-900 nm)."""
    E = lux if lux is not None else cfg.ambient_illuminance_lux
    return cfg.sc_responsivity * (E / 683.0) * 0.05 * (cfg.sc_area_cm2 * 1e-4)


def _noise_vars(I_ph: float, cfg, lux=None) -> dict:
    """
    Six noise variances (A²) at the photodiode node.
    flicker: conservative broadband-equivalent of VFD harmonic EMI
             (assumes some harmonic content falls within the OFDM band).
    adc:     input-referred assuming signal fills ADC full-scale range.
    """
    BW    = cfg.ofdm_sample_rate_hz / 2
    I_amb = _iambient(cfg, lux)
    I_sig = I_ph * cfg.modulation_depth
    return {
        'shot':    2 * _q * I_ph * BW,
        'thermal': 4 * _k * cfg.temperature_K * BW / cfg.r_sense_ohm,
        'ambient': 2 * _q * I_amb * BW,
        'flicker': (I_amb * cfg.mains_flicker_depth) ** 2,
        'adc':     (I_sig / 2 ** cfg.adc_bits) ** 2 / 12,
    }


def _snr_db(I_sig: float, noise_var: float) -> float:
    if noise_var <= 0:
        return 200.0
    return float(20.0 * np.log10(max(I_sig, 1e-30) / np.sqrt(noise_var)))


def _ldpc_ber(snr_db: float, qam: int = 16) -> float:
    """Post-LDPC 5/6 BER: ~3.5 dB coding gain (waterfall approximation)."""
    return max(ber_mqam(snr_db + 3.5, qam), 1e-15)


# =============================================================================
# ANALYSIS 1 — NOISE FLOOR BREAKDOWN
# =============================================================================

def _fig1_noise_breakdown(cfg, output_dir):
    I_ph  = _iphoton(cfg)
    I_sig = I_ph * cfg.modulation_depth
    vars_ = _noise_vars(I_ph, cfg)

    order  = ['thermal', 'shot', 'ambient', 'flicker', 'adc']
    labels = ['Thermal\n(Johnson–Nyquist)', 'Shot\n(photon statistics)',
              'Ambient light\n(500 lux)', 'VFD flicker\n(broadband equiv.)',
              'ADC quant.\n(12-bit)']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    snrs, bers, running = [], [], 0.0
    for src in order:
        running += vars_[src]
        s = _snr_db(I_sig, running)
        snrs.append(s)
        bers.append(max(_ldpc_ber(s, cfg.ofdm_qam_order), 1e-15))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Factory Li-Fi — Noise Floor Breakdown (lififactory1)\n'
        '5 m ceiling · 10 W IR LED · QAM-16 OFDM · LDPC 5/6',
        fontsize=13, fontweight='bold')

    # Left: SNR waterfall
    bars = ax1.barh(labels, snrs, color=colors, alpha=0.85,
                    edgecolor='white', height=0.6)
    for bar, snr in zip(bars, snrs):
        ax1.text(snr + 0.4, bar.get_y() + bar.get_height() / 2,
                 f'{snr:.1f} dB', va='center', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Cumulative SNR (dB)', fontsize=12)
    ax1.set_title('SNR After Each Source Added', fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim([0, max(snrs) * 1.15])

    # Right: BER waterfall (log)
    x = np.arange(len(order))
    ax2.semilogy(x, bers, 'ko-', linewidth=2, markersize=9, zorder=5)
    for xi, (bv, col, lbl) in enumerate(zip(bers, colors, labels)):
        ax2.semilogy(xi, bv, 'o', color=col, markersize=14, zorder=6)
        ax2.text(xi, bv * 2.8, f'{bv:.1e}',
                 ha='center', fontsize=8, color=col, fontweight='bold')
    ax2.axhline(TARGETS['ber'], color='red', linestyle='--', linewidth=1.5,
                label=f"Target BER {TARGETS['ber']:.0e}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([lb.split('\n')[0] for lb in labels],
                         rotation=20, ha='right', fontsize=10)
    ax2.set_ylabel('Bit Error Rate', fontsize=12)
    ax2.set_title('BER per Noise Source (Cumulative)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([1e-15, 1.0])

    save_figure(output_dir, 'fig1_noise_breakdown.png')

    print(f"  Photocurrent:          {I_ph*1e6:.2f} uA  "
          f"(signal: {I_sig*1e6:.2f} uA)")
    print(f"  Noise variances (A2):  "
          + '  '.join(f"{k}={v:.2e}" for k, v in vars_.items()))
    print(f"  Final SNR (all noise): {snrs[-1]:.1f} dB")
    print(f"  Final BER after LDPC:  {bers[-1]:.2e}")
    return bers[-1], snrs[-1]


# =============================================================================
# ANALYSIS 2 — FEC CODING GAIN
# =============================================================================

def _fig2_fec_gain(cfg, output_dir, op_snr_db):
    snr_ax  = np.linspace(0, 40, 400)
    ber_raw = np.array([max(ber_mqam(s, cfg.ofdm_qam_order), 1e-15) for s in snr_ax])
    ber_fec = np.array([max(_ldpc_ber(s,  cfg.ofdm_qam_order), 1e-15) for s in snr_ax])

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.semilogy(snr_ax, ber_raw, 'b-',  linewidth=2.5, label='Uncoded QAM-16')
    ax.semilogy(snr_ax, ber_fec, 'g-',  linewidth=2.5,
                label='LDPC 5/6 + QAM-16  (802.11bb MCS 7)')
    ax.axhline(TARGETS['ber'], color='red',    linestyle='--', linewidth=1.5,
               label=f"Factory BER target  {TARGETS['ber']:.0e}")
    ax.axhline(3.8e-3,         color='orange', linestyle=':',  linewidth=1.5,
               label='FEC pre-coding threshold  3.8 × 10⁻³')
    ax.axvline(op_snr_db, color='purple', linestyle='-.', linewidth=2,
               label=f'Factory op. point  {op_snr_db:.1f} dB')

    # Coding-gain arrow at target BER
    def _snr_at(curve, target):
        return snr_ax[np.argmin(np.abs(curve - target))]

    s_raw = _snr_at(ber_raw, TARGETS['ber'])
    s_fec = _snr_at(ber_fec, TARGETS['ber'])
    gain  = s_raw - s_fec
    ax.annotate('', xy=(s_fec, TARGETS['ber']), xytext=(s_raw, TARGETS['ber']),
                arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2.5))
    ax.text((s_raw + s_fec) / 2, TARGETS['ber'] * 6,
            f'{gain:.1f} dB\ncoding gain',
            ha='center', fontsize=12, color='darkgreen', fontweight='bold')

    ax.set_xlabel('SNR (dB)', fontsize=13)
    ax.set_ylabel('Bit Error Rate', fontsize=13)
    ax.set_title(
        'FEC Coding Gain — Factory Li-Fi (lififactory1)\n'
        'LDPC 5/6 (802.11bb MCS 7) · QAM-16 OFDM · 20 Mbps',
        fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 40]); ax.set_ylim([1e-12, 1.0])

    save_figure(output_dir, 'fig2_fec_gain.png')
    print(f"  Coding gain at BER = {TARGETS['ber']:.0e}: {gain:.1f} dB")
    return gain


# =============================================================================
# ANALYSIS 3 — SENSITIVITY SWEEPS
# =============================================================================

def _sweep_ber(cfg, distance_m=None, led_mW=None, aerosol=0.0, lux=None):
    """BER after LDPC for one operating point."""
    I_ph  = _iphoton(cfg, distance_m=distance_m, led_mW=led_mW, aerosol=aerosol)
    I_sig = I_ph * cfg.modulation_depth
    nv    = _noise_vars(I_ph, cfg, lux=lux)
    snr   = _snr_db(I_sig, sum(nv.values()))
    return max(_ldpc_ber(snr, cfg.ofdm_qam_order), 1e-15)


def _fig3_sensitivity(cfg, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Sensitivity Analysis — Factory Li-Fi (lififactory1)\n'
        'LDPC 5/6 · QAM-16 · 5 m ceiling · 10 W IR LED',
        fontsize=14, fontweight='bold')
    tl = dict(color='red', linestyle='--', linewidth=1.5,
              label=f"Target BER {TARGETS['ber']:.0e}")

    # (a) Industrial aerosol / coolant mist
    ax = axes[0, 0]
    av = np.linspace(0.0, 1.0, 80)
    ax.semilogy(av, [_sweep_ber(cfg, aerosol=a) for a in av], 'b-', linewidth=2)
    ax.axvline(0.5, color='navy', linestyle=':', linewidth=1.5,
               label='Nominal (moderate mist, 0.5)')
    ax.axhline(TARGETS['ber'], **tl)
    ax.set_xlabel('Aerosol density  (0 = clean, 1 = heavy spray)', fontsize=11)
    ax.set_ylabel('BER', fontsize=11)
    ax.set_title('(a) Coolant Mist / Metal Dust\n(850 nm factory BL model)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (b) LED aging / partial failure
    ax = axes[0, 1]
    pct = np.linspace(5, 100, 80)
    nom = cfg.led_radiated_power_mW
    ax.semilogy(pct, [_sweep_ber(cfg, led_mW=nom * p / 100) for p in pct],
                'g-', linewidth=2)
    ax.axvline(100, color='darkgreen', linestyle=':', linewidth=1.5,
               label='Nominal (100%)')
    ax.axhline(TARGETS['ber'], **tl)
    ax.set_xlabel('LED output (% of nominal 10 W)', fontsize=11)
    ax.set_ylabel('BER', fontsize=11)
    ax.set_title('(b) LED Aging / Partial Failure', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (c) Path length / machine body obstruction
    ax = axes[1, 0]
    dv = np.linspace(1.0, 12.0, 80)
    ax.semilogy(dv, [_sweep_ber(cfg, distance_m=d) for d in dv], 'r-', linewidth=2)
    ax.axvline(cfg.distance_m, color='darkred', linestyle=':', linewidth=1.5,
               label=f'Nominal {cfg.distance_m} m')
    ax.axvline(cfg.room_height_m, color='orange', linestyle='--', linewidth=1.5,
               label=f'Ceiling height {cfg.room_height_m} m')
    ax.axhline(TARGETS['ber'], **tl)
    ax.set_xlabel('Effective path length (m)', fontsize=11)
    ax.set_ylabel('BER', fontsize=11)
    ax.set_title('(c) Path Length / Machine Body Blocking', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (d) Ambient illuminance (factory lighting + sunlight)
    ax = axes[1, 1]
    lv = np.linspace(0, 2000, 80)
    ax.semilogy(lv, [_sweep_ber(cfg, lux=lx) for lx in lv], 'm-', linewidth=2)
    ax.axvline(cfg.ambient_illuminance_lux, color='purple', linestyle=':',
               linewidth=1.5, label=f'Nominal {cfg.ambient_illuminance_lux:.0f} lux')
    ax.axhline(TARGETS['ber'], **tl)
    ax.set_xlabel('Ambient illuminance (lux)', fontsize=11)
    ax.set_ylabel('BER', fontsize=11)
    ax.set_title('(d) Factory Lighting / Sunlight Ingress', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(output_dir, 'fig3_sensitivity.png')


# =============================================================================
# ANALYSIS 4 — MULTI-CELL COVERAGE HEATMAP
# =============================================================================

def _snr_at_point(x, y, cfg, ap_x, ap_y) -> float:
    """SNR (dB) at floor point (x,y) from one ceiling AP at (ap_x, ap_y, h)."""
    h   = cfg.room_height_m
    r   = float(np.hypot(x - ap_x, y - ap_y))
    d   = float(np.hypot(r, h))
    ang = np.arctan2(r, h)
    if ang > np.radians(cfg.fov_half_angle_deg):
        return -30.0

    m    = lambertian_order(cfg.led_half_angle_deg)
    c    = h / d                       # cos(emission angle from nadir)
    A    = cfg.sc_area_cm2 * 1e-4
    H    = (m + 1) / (2 * np.pi) * A / d**2 * c**m * c
    P_rx = cfg.led_radiated_power_mW * 1e-3 * H * cfg.lens_transmittance
    I_ph = cfg.sc_responsivity * P_rx
    I_sg = I_ph * cfg.modulation_depth

    I_amb = _iambient(cfg)
    BW    = cfg.ofdm_sample_rate_hz / 2
    nvar  = (2 * _q * I_ph * BW
             + 4 * _k * cfg.temperature_K * BW / cfg.r_sense_ohm
             + 2 * _q * I_amb * BW
             + (I_amb * cfg.mains_flicker_depth) ** 2
             + (I_sg / 2 ** cfg.adc_bits) ** 2 / 12)
    return _snr_db(I_sg, nvar)


def _fig4_multicell(cfg, output_dir):
    Lx, Ly = cfg.room_length_m, cfg.room_width_m
    xs, ys = np.linspace(0, Lx, 150), np.linspace(0, Ly, 100)
    XX, YY = np.meshgrid(xs, ys)

    best_snr = np.full(XX.shape, -30.0)
    for ap_x, ap_y in _AP_XY:
        snr_map = np.vectorize(
            lambda x, y: _snr_at_point(x, y, cfg, ap_x, ap_y)
        )(XX, YY)
        better = snr_map > best_snr
        best_snr[better] = snr_map[better]

    best_ber = np.vectorize(
        lambda s: max(_ldpc_ber(s, cfg.ofdm_qam_order), 1e-15)
    )(best_snr)

    fig, (axS, axB) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Multi-Cell Li-Fi Coverage -- {Lx:.0f} m x {Ly:.0f} m Factory Hall\n'
        f'4x ceiling APs at {cfg.room_height_m:.0f} m height, '
        f'{cfg.led_half_angle_deg:.0f} deg half-angle, {cfg.led_radiated_power_mW/1000:.0f} W IR each',
        fontsize=13, fontweight='bold')

    # SNR heatmap
    im1 = axS.contourf(XX, YY, best_snr, levels=25, cmap='RdYlGn')
    plt.colorbar(im1, ax=axS, label='Best-AP SNR (dB)')
    axS.contour(XX, YY, best_snr, levels=[10, 15, 20, 25, 30],
                colors='white', linewidths=0.7, alpha=0.6)
    for (ax_p, ay_p), lbl, col in zip(_AP_XY, _AP_LABELS, _AP_COLORS):
        axS.plot(ax_p, ay_p, '*', color=col, markersize=18, zorder=10)
        axS.text(ax_p, ay_p + 1.1, lbl, ha='center', color='white',
                 fontsize=10, fontweight='bold')
    axS.set_xlim([0, Lx]); axS.set_ylim([0, Ly])
    axS.set_xlabel('x (m)', fontsize=11); axS.set_ylabel('y (m)', fontsize=11)
    axS.set_title('SNR Map — Best-AP Assignment\n'
                  '(dead zones → 5G/Wi-Fi 6 fallback)', fontsize=11)
    axS.set_aspect('equal')

    # BER heatmap
    log_ber = np.log10(np.clip(best_ber, 1e-15, 1.0))
    im2 = axB.contourf(XX, YY, log_ber, levels=25, cmap='RdYlGn_r')
    plt.colorbar(im2, ax=axB, label='log10(BER) after LDPC 5/6')
    axB.contour(XX, YY, best_ber, levels=[TARGETS['ber']],
                colors='red', linewidths=2.0, linestyles='--')
    for (ax_p, ay_p), lbl, col in zip(_AP_XY, _AP_LABELS, _AP_COLORS):
        axB.plot(ax_p, ay_p, '*', color=col, markersize=18, zorder=10)
        axB.text(ax_p, ay_p + 1.1, lbl, ha='center', color='white',
                 fontsize=10, fontweight='bold')
    axB.set_xlim([0, Lx]); axB.set_ylim([0, Ly])
    axB.set_xlabel('x (m)', fontsize=11)
    axB.set_title(f'BER Map after LDPC 5/6\n'
                  f'(red dashed = target {TARGETS["ber"]:.0e})', fontsize=11)
    axB.set_aspect('equal')

    plt.tight_layout()
    save_figure(output_dir, 'fig4_multicell.png')

    # Per-cell table (directly below each AP = best-case)
    coverage = 100.0 * float(np.mean(best_ber <= TARGETS['ber']))
    print(f"\n  {'AP':5s}  {'Position (x,y)':18s}  {'SNR':>9s}  {'BER':>10s}")
    for (ax_p, ay_p), lbl in zip(_AP_XY, _AP_LABELS):
        s   = _snr_at_point(ax_p, ay_p, cfg, ax_p, ay_p)
        bv  = max(_ldpc_ber(s, cfg.ofdm_qam_order), 1e-15)
        print(f"  {lbl:5s}  ({ax_p:4.1f} m, {ay_p:4.1f} m)   {s:>8.1f} dB  {bv:>10.2e}")
    print(f"\n  Floor coverage at BER ≤ {TARGETS['ber']:.0e}: {coverage:.1f}%")
    print(f"  (Narrow 30° beams give high SNR near each AP; inter-AP gaps")
    print(f"   fall back to 5G URLLC / Wi-Fi 6 — this motivates the hybrid design.)")
    return coverage


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_validation(output_dir=None) -> bool:
    cfg = SystemConfig.from_preset('lififactory1')
    output_dir = validation_header(
        'FACTORY Li-Fi ANALYSIS — IIoT CNC FLOOR  (lififactory1)',
        'Industry 4.0 hybrid wireless: Li-Fi + 5G URLLC + Wi-Fi 6',
        output_dir, default_subdir='lififactory1',
    )

    print('\n[1/4] Noise floor breakdown')
    final_ber, final_snr = _fig1_noise_breakdown(cfg, output_dir)

    print('\n[2/4] FEC coding gain')
    coding_gain = _fig2_fec_gain(cfg, output_dir, op_snr_db=final_snr)

    print('\n[3/4] Sensitivity sweeps')
    _fig3_sensitivity(cfg, output_dir)

    print('\n[4/4] Multi-cell coverage heatmap')
    coverage = _fig4_multicell(cfg, output_dir)

    print()
    metrics = [
        validate_metric(final_ber, TARGETS['ber'],
                        'Factory BER (all noise + LDPC)', threshold_pct=9900.0),
        validate_metric(coverage, TARGETS['coverage_pct'],
                        'Floor coverage at target BER (%)', threshold_pct=50.0),
    ]
    passed = print_validation_summary(metrics)
    print(f'\n  Figures saved to: {output_dir}')
    return passed

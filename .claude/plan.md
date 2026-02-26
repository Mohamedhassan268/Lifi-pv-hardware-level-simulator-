# UX Improvements Plan: Tooltips, Wizard, Live Warnings

## Overview
Three features to make the simulator approachable for users without deep LiFi-PV expertise.

---

## Feature 1: Tooltips + Preset Descriptions

### 1a. Parameter Tooltips (in `gui/tab_system_setup.py`)
Add `.setToolTip()` to every parameter widget with:
- What it does in plain English
- Typical range / recommended values
- What happens if you increase/decrease it

**Widgets to add tooltips:**
| Widget | Tooltip |
|--------|---------|
| `_led_combo` | "LED used for optical transmission. Determines max power and spectral output." |
| `_driver_combo` | "Op-amp driving the LED. Affects bandwidth and modulation fidelity." |
| `_mod_depth` | "How deeply the LED is modulated (0-1). Higher = stronger signal but less DC for harvesting. Typical: 0.3-0.8" |
| `_led_power` | "Total radiated optical power from the LED. Higher = longer range but more power consumption." |
| `_half_angle` | "LED beam half-angle. Narrow (15°) = focused beam, Wide (60°) = broad coverage." |
| `_distance` | "Distance between LED transmitter and PV receiver. Most critical parameter for link budget." |
| `_tx_angle` | "Angle of LED off the vertical axis. 0° = pointing straight down." |
| `_rx_tilt` | "Tilt of the receiver from horizontal. 0° = facing straight up." |
| `_lens_t` | "Optical lens/filter transmittance (0-1). Accounts for concentrator or filter losses." |
| `_pv_combo` | "Photovoltaic cell used as receiver. Determines responsivity and harvesting capability." |
| `_ina_combo` | "Instrumentation amplifier for signal extraction from the PV cell." |
| `_comp_combo` | "Comparator for digital signal recovery (OOK demodulation)." |
| `_r_sense` | "Sense resistor converting photocurrent to voltage. Higher = more signal but more noise. Typical: 10-100 Ohm" |
| `_bpf_stages` | "Number of bandpass filter stages. More stages = cleaner signal but more delay. Typical: 1-2" |
| `_dcdc_fsw` | "DC-DC converter switching frequency. Should be well above the data rate to avoid interference." |
| `_data_rate` | "Bit rate for the communication link. Higher rates need stronger signals (more SNR)." |
| `_n_bits` | "Number of PRBS bits to simulate. More bits = better BER accuracy but longer simulation." |
| `_t_stop` | "Simulation end time. Must be >= n_bits / data_rate for complete transmission." |
| `_noise_enable` | Already has tooltip (keep as-is) |
| `_humidity` | "Relative humidity (0-1). Affects atmospheric absorption at long distances." |
| `_ofdm_nfft` | "FFT size for OFDM. Larger = more subcarriers, better spectral efficiency." |
| `_ofdm_qam` | "QAM constellation order (4/16/64/256). Higher = more bits/symbol but needs more SNR." |
| `_bfsk_f0`/`_f1` | "BFSK frequencies for mark/space. Should be within the BPF passband." |
| `_notch_freq` | "Notch filter center frequency to reject ambient light interference (e.g., 100/120 Hz mains flicker)." |
| `_amp_gain` | "Additional amplifier gain applied to received signal before demodulation." |

### 1b. Preset Description Card (in `gui/tab_system_setup.py`)
Add a `QLabel` below the preset dropdown that shows a human-readable description when a preset is selected.

**Add a `PRESET_DESCRIPTIONS` dict** (in `gui/tab_system_setup.py` or a small helper):
```python
PRESET_DESCRIPTIONS = {
    'kadirvelu2021': {
        'title': 'Indoor SLIPT System',
        'summary': 'Simultaneous light communication + energy harvesting at 30 cm, 5 kbps OOK.',
        'highlights': 'SPICE engine | Full RX chain (INA + BPF + comparator + DC-DC)',
    },
    'fakidis2020': {
        'title': 'Indoor VLC Link Budget',
        'summary': 'Detailed link budget analysis at 50 cm with OOK modulation.',
        'highlights': 'SPICE engine | Focus on receiver sensitivity',
    },
    'sarwar2017': {
        'title': 'OFDM LiFi Communication',
        'summary': 'High-speed OFDM with 16-QAM at 1.25 m, testing multi-carrier performance.',
        'highlights': 'Python engine | OFDM modulation | High data rate',
    },
    'xu2024': {
        'title': 'BFSK Energy-Harvesting Link',
        'summary': 'Low-power BFSK modulation optimized for indoor energy harvesting.',
        'highlights': 'Python engine | BFSK modulation',
    },
    'gonzalez2024': {
        'title': 'Manchester-Coded OOK',
        'summary': 'Manchester coding for improved clock recovery at moderate distances.',
        'highlights': 'SPICE engine | Manchester OOK',
    },
    'correa2025': {
        'title': 'Greenhouse SLIPT',
        'summary': 'PWM-ASK modulation for greenhouse monitoring with humidity effects.',
        'highlights': 'Python engine | PWM-ASK | Environmental factors',
    },
    'oliveira2024': {
        'title': 'High-Speed OFDM-QAM',
        'summary': 'Advanced OFDM with multi-cell PV receiver for high throughput.',
        'highlights': 'Python engine | 64-QAM OFDM | Multi-cell receiver',
    },
}
```

**UI change:** Add `self._preset_desc` QLabel below preset combo, styled with dim text. Update in `_on_preset_changed()`.

---

## Feature 2: Guided Quick Start Wizard

### New file: `gui/wizard.py`
A `QDialog` (not QWizard — simpler, more control over styling) with a stacked widget showing 3-4 pages.

**Page 1: "Choose a Scenario"**
- 3-4 clickable scenario cards:
  - "Indoor Desktop LiFi" → loads kadirvelu2021
  - "High-Speed OFDM Link" → loads sarwar2017
  - "Low-Power Sensor Node" → loads xu2024
  - "Custom Setup" → starts with defaults, lets user tweak
- Each card: icon/emoji-free title, 1-line description, key specs

**Page 2: "Adjust Key Parameters"**
- Only the 3-4 most important parameters:
  - Distance slider with visual diagram
  - LED Power slider
  - Data Rate selector (Low/Medium/High)
- Live SNR estimate shown as a gauge/bar

**Page 3: "Ready to Simulate"**
- Summary of what was configured
- "Open in System Setup" button → applies config and switches to Tab 1
- "Run Simulation Now" button → applies config and switches to Tab 4

**Integration:**
- Menu: Help > Quick Start Guide
- Also: a "Quick Start" button in Tab 1's left sidebar (above preset combo)
- Wizard returns the chosen SystemConfig; main_window applies it via `_tab_setup.set_config()`

**Styling:** Use COLORS from theme, keep consistent with Midnight Blue.

---

## Feature 3: Live Warnings Panel

### New widget: `WarningsPanel` (in `gui/widgets.py`)
A scrollable panel that displays warning/info/success messages.

**Structure:**
```python
class WarningsPanel(QWidget):
    """Displays live warnings and recommendations based on current config."""
    def update_from_config(self, config):
        # Run all checks, display results
```

**Checks to implement:**
| Check | Condition | Level | Message |
|-------|-----------|-------|---------|
| Weak signal | SNR < 6 dB | error | "Signal too weak — reduce distance or increase LED power" |
| Marginal signal | 6 < SNR < 12 dB | warning | "Marginal SNR — expect high BER. Consider reducing distance." |
| Good signal | SNR > 12 dB | success | "Good link budget (SNR: X dB)" |
| Geometry issue | tx_angle > half_angle | warning | "TX angle exceeds LED half-angle — receiver may be outside beam" |
| Short simulation | t_stop < n_bits/data_rate | warning | "Simulation too short to transmit all bits" |
| No SPICE engine | engine=spice & !spice_available | warning | "No SPICE found — will auto-fall back to Python engine" |
| High data rate for OOK | OOK & data_rate > 100kbps | info | "Consider OFDM for data rates above 100 kbps" |
| Very close range | distance < 0.02 m | info | "Very close range — near-field effects may not be modeled" |

**Integration in `tab_system_setup.py`:**
- Add below QuickInfoPanel in the left sidebar
- Called from `_recalculate()` alongside `_quick_info.update_from_config()`

**Styling:** Each message has an icon prefix (colored dot or text), uses COLORS[warning/error/success/info].

---

## Files to Create
1. `gui/wizard.py` — Quick Start Wizard dialog (~200 lines)

## Files to Modify
1. `gui/widgets.py` — Add `WarningsPanel` class (~70 lines)
2. `gui/tab_system_setup.py` — Add tooltips, preset description label, warnings panel, wizard button
3. `gui/main_window.py` — Add "Quick Start" menu item under Help
4. `gui/theme.py` — Add QSS for wizard cards and warnings panel

## Implementation Order
1. Tooltips + Preset Cards (quick win, tab_system_setup.py only)
2. Live Warnings Panel (widgets.py + tab_system_setup.py)
3. Guided Wizard (new file + main_window.py integration)

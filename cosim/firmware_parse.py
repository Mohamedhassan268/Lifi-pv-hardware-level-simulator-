"""Parse PHY-relevant constants out of ESP32 Arduino (.ino) firmware.

The two ESP32s in the LiFi link own only the *parameters* of the PHY (the physics
is in SPICE + the optical channel): the TX firmware sets the carrier / PWM / data
rate / drive, the RX firmware sets the ADC pin / resolution / sample rate. This
module statically extracts those constants and maps them onto SystemConfig fields
— no execution, no toolchain.

It returns two kinds of result:
  - ``findings`` — numeric SystemConfig overrides applied to the run, each with
    the source token it matched and a confidence.
  - ``info`` — descriptive values shown to the user but not fed as numeric config
    (pin assignments, the MCU board, the detected modulation scheme, threshold).

It is intentionally heuristic and tolerant: anything it can't find is left unset
(the preset/config default stands), and everything carries provenance so the UI
can show exactly what was read and let the user correct it.

Entry point: ``parse_firmware(source, role) -> ParseResult`` with role 'tx'|'rx'.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


@dataclass
class Finding:
    """A numeric parameter applied to the run, with provenance."""
    config_field: str
    value: float
    source: str
    label: str
    confidence: str = "high"  # high | medium | low


@dataclass
class InfoItem:
    """A descriptive value shown to the user (not a numeric config override)."""
    label: str
    value: str
    source: str


@dataclass
class ParseResult:
    role: str
    params: dict[str, float] = field(default_factory=dict)   # config_field -> value
    findings: list[Finding] = field(default_factory=list)
    info: list[InfoItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, f: Optional[Finding]) -> None:
        if f is None or f.config_field in self.params:
            return  # first finding for a field wins (callers try highest first)
        self.params[f.config_field] = f.value
        self.findings.append(f)

    def add_info(self, item: Optional[InfoItem]) -> None:
        if item is None:
            return
        if any(i.label == item.label for i in self.info):
            return
        self.info.append(item)


# ---------------------------------------------------------------------------
# Low-level extraction helpers
# ---------------------------------------------------------------------------

def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    return src


def _symbol_table(src: str) -> dict[str, float]:
    """Numeric #defines and simple typed assignments -> {name: value}."""
    syms: dict[str, float] = {}
    for m in re.finditer(r"#define\s+(\w+)\s+\(?\s*(" + _NUM + r")\s*\)?", src):
        syms.setdefault(m.group(1), float(m.group(2)))
    typ = r"(?:const\s+)?(?:static\s+)?(?:unsigned\s+)?(?:int|long|float|double|uint\d+_t|int\d+_t|byte)"
    for m in re.finditer(typ + r"\s+(\w+)\s*=\s*(" + _NUM + r")\s*[;,]", src):
        syms.setdefault(m.group(1), float(m.group(2)))
    return syms


def _string_defines(src: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r'#define\s+(\w+)\s+"([^"]*)"', src):
        out.setdefault(m.group(1), m.group(2))
    return out


def _resolve(token: str, syms: dict[str, float]) -> Optional[float]:
    token = token.strip()
    if re.fullmatch(_NUM, token):
        return float(token)
    return syms.get(token)


def _call_args(src: str, fn: str) -> list[list[str]]:
    out: list[list[str]] = []
    for m in re.finditer(re.escape(fn) + r"\s*\(([^)]*)\)", src):
        inner = m.group(1).strip()
        out.append([a.strip() for a in inner.split(",")] if inner else [])
    return out


def _find_symbol(syms: dict[str, float], *patterns: str) -> Optional[tuple[str, float]]:
    """First symbol whose name matches ALL given regex fragments (case-insens.)."""
    for name, val in syms.items():
        if all(re.search(p, name, re.I) for p in patterns):
            return name, val
    return None


def _first_call_arg(src: str, fn: str, idx: int, syms: dict[str, float]) -> Optional[float]:
    for args in _call_args(src, fn):
        if len(args) > idx and (v := _resolve(args[idx], syms)) is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Shared (either file may carry these)
# ---------------------------------------------------------------------------

_SCHEMES = [
    (r"pwm.?ask|ask.?pwm", "PWM_ASK"),
    (r"manchester", "OOK_Manchester"),
    (r"\bofdm\b", "OFDM"),
    (r"\bb?fsk\b", "BFSK"),
    (r"\book\b", "OOK"),
]


def _parse_common(src: str, syms: dict[str, float], strs: dict[str, str],
                  res: ParseResult) -> None:
    # Data rate (named bit/data rate, or derived from a symbol period).
    hit = _find_symbol(syms, "data", "rate") or _find_symbol(syms, "bit", "rate") \
        or _find_symbol(syms, r"baud(?!.*serial)")
    if hit:
        res.add(Finding("data_rate_bps", hit[1], hit[0], "Data rate"))
    else:
        per = _find_symbol(syms, "symbol", "(ms|period|us|len)")
        if per:
            name, v = per
            secs = v / 1000.0 if re.search("ms", name, re.I) else \
                v / 1e6 if re.search("us", name, re.I) else v
            if secs > 0:
                res.add(Finding("data_rate_bps", round(1.0 / secs, 3), name,
                                "Data rate (from symbol period)", "medium"))

    # Modulation scheme: explicit string define, else keyword scan (info only).
    scheme = None
    for v in strs.values():
        for pat, name in _SCHEMES:
            if re.search(pat, v, re.I):
                scheme = name
                break
    if scheme is None:
        for pat, name in _SCHEMES:
            if re.search(pat, src, re.I):
                scheme = name
                break
    if scheme:
        res.add_info(InfoItem("Modulation scheme", scheme, "source"))

    # MCU board.
    for k, v in strs.items():
        if re.search(r"board|mcu|chip", k, re.I):
            res.add_info(InfoItem("MCU board", v, k))
            break
    else:
        m = re.search(r"ESP32(?:-?[A-Z0-9]+)?|Arduino\s*\w*|RP2040|Pico", src)
        if m:
            res.add_info(InfoItem("MCU board", m.group(0), "source"))

    # CPU clock: setCpuFrequencyMhz(N), or F_CPU (Hz), or a *_MHZ symbol.
    mhz = _first_call_arg(src, "setCpuFrequencyMhz", 0, syms)
    if mhz is None:
        h = _find_symbol(syms, "cpu", "mhz") or _find_symbol(syms, r"\bf_cpu\b")
        if h:
            mhz = h[1] / 1e6 if h[1] > 1e5 else h[1]   # F_CPU is in Hz
    if mhz:
        res.add(Finding("mcu_clock_MHz", float(mhz), "setCpuFrequencyMhz/F_CPU",
                        "CPU clock", "medium"))


# ---------------------------------------------------------------------------
# Role parsers
# ---------------------------------------------------------------------------

def _parse_tx(src: str, syms: dict[str, float], res: ParseResult) -> None:
    # Carrier frequency.
    hit = _find_symbol(syms, "carrier", "freq") or _find_symbol(syms, r"\bf?c\b", "freq") \
        or _find_symbol(syms, "carrier")
    if hit:
        res.add(Finding("carrier_freq_hz", hit[1], hit[0], "Carrier frequency"))
    if "carrier_freq_hz" not in res.params:
        v = _first_call_arg(src, "ledcWriteTone", 1, syms)
        if v:
            res.add(Finding("carrier_freq_hz", v, "ledcWriteTone(...)", "Carrier frequency", "medium"))

    # PWM (dimming) frequency, and PWM resolution bits (info).
    hit = _find_symbol(syms, "pwm", "freq")
    if hit:
        res.add(Finding("pwm_freq_hz", hit[1], hit[0], "PWM frequency"))
    for args in _call_args(src, "ledcSetup"):
        if "pwm_freq_hz" not in res.params and len(args) >= 2 and (v := _resolve(args[1], syms)):
            res.add(Finding("pwm_freq_hz", v, "ledcSetup(...)", "PWM frequency", "low"))
        if len(args) >= 3 and (rb := _resolve(args[2], syms)):
            res.add_info(InfoItem("PWM resolution", f"{int(rb)}-bit", "ledcSetup(...)"))

    # Modulation depth: an explicit MOD_DEPTH wins; otherwise derive it from the
    # span of the PWM-ASK duty levels.
    hit = _find_symbol(syms, "mod", "depth") or _find_symbol(syms, "modulation", "index")
    if hit:
        d = hit[1] / 100.0 if hit[1] > 1.0 else hit[1]
        res.add(Finding("modulation_depth", round(d, 4), hit[0], "Modulation depth"))

    # PWM-ASK symbol duty levels (e.g. DUTY_A/B/C = 25/50/75). Report them, and
    # if no explicit depth was found, derive it from the min/max:
    #   depth = (d_max - d_min) / (d_max + d_min)   (optical-power contrast).
    duties = sorted(v for name, v in syms.items()
                    if re.search(r"duty", name, re.I) and 0 < v <= 100)
    if duties:
        res.add_info(InfoItem("PWM duty levels",
                              "/".join(str(int(v)) for v in duties) + "%", "DUTY_*"))
        if "modulation_depth" not in res.params and len(duties) >= 2:
            d_min, d_max = duties[0], duties[-1]
            if d_max + d_min > 0:
                depth = (d_max - d_min) / (d_max + d_min)
                res.add(Finding("modulation_depth", round(depth, 4),
                                "duty span", "Modulation depth (from duty levels)", "medium"))

    # LED bias current (mA or A) and radiated optical power (mW).
    hit = _find_symbol(syms, "(led|bias|drive)", "current", "ma") or _find_symbol(syms, "current", "ma")
    if hit:
        res.add(Finding("bias_current_A", round(hit[1] / 1000.0, 6), hit[0], "LED bias current"))
    else:
        hit = _find_symbol(syms, "(led|bias)", "current")
        if hit:
            res.add(Finding("bias_current_A", hit[1], hit[0], "LED bias current", "medium"))
    hit = _find_symbol(syms, "(led|opt)", "power", "mw") or _find_symbol(syms, r"\bpe\b", "mw")
    if hit:
        res.add(Finding("led_radiated_power_mW", hit[1], hit[0], "LED optical power"))

    # PRBS order and packet length.
    hit = _find_symbol(syms, "prbs", "order") or _find_symbol(syms, r"\bprbs\b")
    if hit and 3 <= hit[1] <= 31:
        res.add(Finding("prbs_order", int(hit[1]), hit[0], "PRBS order"))
    hit = _find_symbol(syms, "(n_?bits|num_?bits|packet_?bits|frame_?bits)")
    if hit:
        res.add(Finding("n_bits", int(hit[1]), hit[0], "Packet length"))

    # LED / TX GPIO pin (info).
    pin = _pin(src, syms, "(led|tx)_?pin", ("ledcAttachPin", 0), ("digitalWrite", 0), ("pinMode", 0))
    if pin is not None:
        res.add_info(InfoItem("LED GPIO", f"GPIO{int(pin)}", "TX pin"))

    if not res.params and not res.info:
        res.warnings.append(
            "No TX PHY constants recognised. Name them clearly "
            "(e.g. CARRIER_FREQ, DATA_RATE, PWM_FREQ, MOD_DEPTH).")


def _parse_rx(src: str, syms: dict[str, float], res: ParseResult) -> None:
    # ADC resolution.
    v = _first_call_arg(src, "analogReadResolution", 0, syms)
    if v:
        res.add(Finding("adc_bits", v, "analogReadResolution(...)", "ADC resolution"))
    if "adc_bits" not in res.params:
        hit = _find_symbol(syms, "adc", "(res|bits)")
        if hit:
            res.add(Finding("adc_bits", hit[1], hit[0], "ADC resolution", "medium"))

    # ADC reference voltage.
    hit = _find_symbol(syms, "(adc_?)?vref") or _find_symbol(syms, "(adc|ref)", "volt")
    if hit:
        ref = hit[1] / 1000.0 if hit[1] > 100 else hit[1]   # mV -> V if large
        res.add(Finding("adc_vref", round(ref, 4), hit[0], "ADC reference"))

    # Sample rate: named rate / period / timer / delayMicroseconds.
    hit = _find_symbol(syms, "sample", "(rate|freq|hz)")
    if hit:
        res.add(Finding("mcu_sample_rate_hz", hit[1], hit[0], "ADC sample rate"))
    else:
        per = _find_symbol(syms, "sample", "(us|period|interval|micros)")
        cand = None
        if per and per[1] > 0:
            cand = (per[1], per[0], "medium")
        else:
            v = _first_call_arg(src, "timerAlarmWrite", 1, syms) \
                or _first_call_arg(src, "delayMicroseconds", 0, syms)
            if v and v > 0:
                cand = (v, "timer/delayMicroseconds(...)", "low")
        if cand:
            res.add(Finding("mcu_sample_rate_hz", round(1e6 / cand[0], 1),
                            cand[1], "ADC sample rate (from period)", cand[2]))

    # ADC GPIO pin + threshold (info).
    pin = _pin(src, syms, "(adc|rx)_?pin", ("analogRead", 0))
    if pin is not None:
        res.add_info(InfoItem("ADC GPIO", f"GPIO{int(pin)}", "RX pin"))
    hit = _find_symbol(syms, "(threshold|thresh|slice)")
    if hit:
        res.add_info(InfoItem("Threshold", str(int(hit[1])), hit[0]))

    if "mcu_sample_rate_hz" not in res.params:
        res.warnings.append(
            "No ADC sample rate found — set it manually or the sim uses the ideal-ADC default.")


def _pin(src, syms, name_pat, *calls) -> Optional[float]:
    """A GPIO pin number: a *_PIN symbol, else the first arg of one of the
    listed Arduino calls (e.g. analogRead(pin), ledcAttachPin(pin, ch))."""
    hit = _find_symbol(syms, name_pat)
    if hit and 0 <= hit[1] <= 48:
        return hit[1]
    for fn, idx in calls:
        v = _first_call_arg(src, fn, idx, syms)
        if v is not None and 0 <= v <= 48:
            return v
    return None


def parse_firmware(source: str, role: str) -> ParseResult:
    """Extract PHY parameters from one ESP32 .ino. role is 'tx' or 'rx'."""
    role = (role or "").lower()
    res = ParseResult(role=role)
    if role not in ("tx", "rx"):
        res.warnings.append(f"Unknown role {role!r}; expected 'tx' or 'rx'.")
        return res

    src = _strip_comments(source or "")
    syms = _symbol_table(src)
    strs = _string_defines(source or "")  # strings: keep original (comments stripped anyway)
    _parse_common(src, syms, strs, res)
    if role == "tx":
        _parse_tx(src, syms, res)
    else:
        _parse_rx(src, syms, res)
    return res

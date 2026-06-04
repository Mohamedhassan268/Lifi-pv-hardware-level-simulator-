"""Parse PHY-relevant constants out of ESP32 Arduino (.ino) firmware.

The two ESP32s in the LiFi link own only the *parameters* of the PHY (the physics
is in SPICE + the optical channel): the TX firmware sets the carrier / PWM / data
rate, the RX firmware sets the ADC pin / resolution / sample rate. This module
statically extracts those constants and maps them onto SystemConfig fields — no
execution, no toolchain.

It is intentionally heuristic and *transparent*: every value it pulls comes back
with the source token it matched, so the UI can show the user exactly what was
read and let them correct it. Anything it can't find is simply left unset (the
preset/config default stands).

Entry point: ``parse_firmware(source, role) -> ParseResult`` with role 'tx'|'rx'.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


@dataclass
class Finding:
    """One extracted parameter: the config field, its value, and where it came
    from (so the UI can show provenance)."""
    config_field: str
    value: float
    source: str          # the token/line this was read from
    label: str           # human label
    confidence: str = "high"  # high | medium | low


@dataclass
class ParseResult:
    role: str
    params: dict[str, float] = field(default_factory=dict)   # config_field -> value
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, f: Optional[Finding]) -> None:
        if f is None:
            return
        # First finding for a field wins (callers try highest-confidence first).
        if f.config_field in self.params:
            return
        self.params[f.config_field] = f.value
        self.findings.append(f)


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


def _resolve(token: str, syms: dict[str, float]) -> Optional[float]:
    token = token.strip()
    if re.fullmatch(_NUM, token):
        return float(token)
    return syms.get(token)


def _call_args(src: str, fn: str) -> list[list[str]]:
    """Every call ``fn(a, b, ...)`` -> list of trimmed arg-token lists."""
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


# ---------------------------------------------------------------------------
# Role parsers
# ---------------------------------------------------------------------------

def _parse_tx(src: str, syms: dict[str, float], res: ParseResult) -> None:
    # Carrier frequency: a named symbol, else ledcWriteTone(ch, freq), else a
    # ledcSetup whose channel name reads "carrier".
    hit = _find_symbol(syms, "carrier", "freq") or _find_symbol(syms, r"\bf?c\b", "freq") \
        or _find_symbol(syms, "carrier")
    if hit:
        res.add(Finding("carrier_freq_hz", hit[1], hit[0], "Carrier frequency"))
    for args in _call_args(src, "ledcWriteTone"):
        if len(args) >= 2 and (v := _resolve(args[1], syms)):
            res.add(Finding("carrier_freq_hz", v, "ledcWriteTone(...)", "Carrier frequency", "medium"))

    # PWM frequency (the slow dimming/symbol envelope).
    hit = _find_symbol(syms, "pwm", "freq")
    if hit:
        res.add(Finding("pwm_freq_hz", hit[1], hit[0], "PWM frequency"))

    # Data rate: named bit/data rate, or derived from a symbol period.
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

    # Modulation depth, only when explicitly named (duty cycles alone are ambiguous).
    hit = _find_symbol(syms, "mod", "depth") or _find_symbol(syms, "modulation", "index")
    if hit:
        d = hit[1] / 100.0 if hit[1] > 1.0 else hit[1]
        res.add(Finding("modulation_depth", d, hit[0], "Modulation depth"))


def _parse_rx(src: str, syms: dict[str, float], res: ParseResult) -> None:
    # ADC resolution -> bits.
    for args in _call_args(src, "analogReadResolution"):
        if args and (v := _resolve(args[0], syms)):
            res.add(Finding("adc_bits", v, "analogReadResolution(...)", "ADC resolution"))
    hit = _find_symbol(syms, "adc", "(res|bits)")
    if hit:
        res.add(Finding("adc_bits", hit[1], hit[0], "ADC resolution", "medium"))

    # Sample rate: named rate, named period, hardware-timer period, or a
    # delayMicroseconds in the sampling loop.
    hit = _find_symbol(syms, "sample", "(rate|freq|hz)")
    if hit:
        res.add(Finding("mcu_sample_rate_hz", hit[1], hit[0], "ADC sample rate"))
    else:
        per = _find_symbol(syms, "sample", "(us|period|interval|micros)")
        if per:
            name, v = per
            if v > 0:
                res.add(Finding("mcu_sample_rate_hz", round(1e6 / v, 1), name,
                                "ADC sample rate (from period)", "medium"))
        else:
            # timerAlarmWrite(timer, period_us, ...) or delayMicroseconds(us)
            cand = None
            for args in _call_args(src, "timerAlarmWrite"):
                if len(args) >= 2 and (v := _resolve(args[1], syms)) and v > 0:
                    cand = (v, "timerAlarmWrite(...)")
                    break
            if cand is None:
                for args in _call_args(src, "delayMicroseconds"):
                    if args and (v := _resolve(args[0], syms)) and v > 0:
                        cand = (v, "delayMicroseconds(...)")
                        break
            if cand:
                res.add(Finding("mcu_sample_rate_hz", round(1e6 / cand[0], 1),
                                cand[1], "ADC sample rate (from period)", "low"))

    if "mcu_sample_rate_hz" not in res.params:
        res.warnings.append(
            "No ADC sample rate found — set it manually or the sim uses the ideal-ADC default.")


def parse_firmware(source: str, role: str) -> ParseResult:
    """Extract PHY parameters from one ESP32 .ino. role is 'tx' or 'rx'."""
    role = (role or "").lower()
    res = ParseResult(role=role)
    if role not in ("tx", "rx"):
        res.warnings.append(f"Unknown role {role!r}; expected 'tx' or 'rx'.")
        return res

    src = _strip_comments(source or "")
    syms = _symbol_table(src)
    if role == "tx":
        _parse_tx(src, syms, res)
        if not res.params:
            res.warnings.append(
                "No TX PHY constants recognised. Name them clearly "
                "(e.g. CARRIER_FREQ, DATA_RATE, PWM_FREQ).")
    else:
        _parse_rx(src, syms, res)
    return res

"""Generate flashable ESP32 starter firmware from the configured PHY.

The inverse of cosim.firmware_parse: take a SystemConfig and emit TX + RX
Arduino sketches for the scheme the real board runs. Two schemes are supported,
both of which an ESP32 can drive *faithfully* (per the agreed scope):

  - PWM_ASK  carrier on the LED pin, bit shifts the PWM duty (bright/dim by the
             modulation depth); RX envelope-thresholds the ADC.
  - BFSK     two LEDC tones (f0/f1) per bit; RX discriminates them with a pair
             of Goertzel filters.

The RX sketch prints its capture in the exact CSV shape the Measured-vs-Sim
panel ingests (`# sample_rate_hz=`, `# ber=`, then one ADC sample per line), so
generate -> flash -> capture -> compare is a closed loop.

OFDM is intentionally not generated — a faithful DCO-OFDM transmitter needs an
IFFT + DAC + tight timing that a bare ESP32 can't honour, so emitting it would
be unfaithful code.
"""

from __future__ import annotations

from string import Template

# Maximal-length LFSR feedback tap (the second tap; the first is the order N)
# for x^N + x^TAP + 1. Covers the PRBS orders the presets use.
_PRBS_TAPS = {3: 2, 4: 3, 5: 3, 6: 5, 7: 6, 9: 5, 10: 7,
              11: 9, 15: 14, 17: 14, 18: 11, 20: 17, 23: 18, 31: 28}

# Default board pinout (matches the canvas DRIVE/MCU nodes; user can edit).
_LED_PIN = 25
_ADC_PIN = 34


def _prbs_block(order: int) -> str:
    """A self-contained PRBS generator function for the sketch."""
    tap = _PRBS_TAPS.get(order, 6)
    return Template(
        """// Maximal-length PRBS-$N LFSR (x^$N + x^$TAP + 1), shared by TX & RX.
static uint32_t lfsr = 1;  // any non-zero seed; TX and RX must match
bool prbs_next() {
  bool b = ((lfsr >> ($N - 1)) ^ (lfsr >> ($TAP - 1))) & 1u;
  lfsr = ((lfsr << 1) | b) & ((1u << $N) - 1u);
  return b;
}"""
    ).substitute(N=order, TAP=tap)


def _default_sample_rate(scheme: str, data_rate: float, hi_freq: float) -> int:
    """Pick a realistic ADC rate if the config doesn't pin one (analogRead caps
    out around ~20 kHz in a simple loop)."""
    if scheme == "BFSK":
        want = 8.0 * hi_freq          # resolve the tones for Goertzel
    else:
        want = 20.0 * data_rate       # ~20 envelope samples per bit
    return int(min(max(want, 8000.0), 20000.0))


def _params(cfg) -> dict:
    d = cfg.to_dict()
    data_rate = float(d.get("data_rate_bps") or 1000.0)
    scheme = (d.get("modulation") or "").upper().replace("-", "_")
    f0 = float(d.get("bfsk_f0_hz") or 1600.0)
    f1 = float(d.get("bfsk_f1_hz") or 2000.0)
    hi_freq = max(f0, f1, float(d.get("carrier_freq_hz") or 0.0))
    sr = float(d.get("mcu_sample_rate_hz") or 0.0)
    sample_rate = int(sr) if sr > 0 else _default_sample_rate(scheme, data_rate, hi_freq)
    n_bits = int(min(max(int(d.get("n_bits") or 128), 32), 256))  # bound the RX buffer
    return {
        "scheme": scheme,
        "data_rate": int(round(data_rate)),
        "carrier": int(round(float(d.get("carrier_freq_hz") or 10000.0))),
        "depth": round(float(d.get("modulation_depth") or 0.9), 3),
        "f0": int(round(f0)),
        "f1": int(round(f1)),
        "prbs_order": int(d.get("prbs_order") or 7),
        "adc_bits": int(d.get("adc_bits") or 12),
        "sample_rate": sample_rate,
        "n_bits": n_bits,
    }


# --------------------------------------------------------------------------- #
# PWM-ASK
# --------------------------------------------------------------------------- #

def _pwm_ask(p: dict) -> tuple[str, str]:
    tx = Template(
        '''// OptiSim — PWM-ASK transmitter (ESP32). Carrier on the LED pin; each PRBS
// bit shifts the PWM duty between a bright/dim level set by the modulation depth.
#define LED_PIN      $LED
#define CARRIER_HZ   $CARRIER
#define BIT_RATE_HZ  $BITRATE
#define PWM_RES_BITS 10
#define DUTY_MAX     ((1 << PWM_RES_BITS) - 1)
const float DEPTH = $DEPTH;            // 0..1 optical modulation depth
const uint32_t BIT_US = 1000000UL / BIT_RATE_HZ;

$PRBS

void setup() {
  ledcSetup(0, CARRIER_HZ, PWM_RES_BITS);
  ledcAttachPin(LED_PIN, 0);
}

void loop() {
  bool bit = prbs_next();
  int mid = DUTY_MAX / 2;
  int hi  = mid + (int)(mid * DEPTH);  // bright = '1'
  int lo  = mid - (int)(mid * DEPTH);  // dim    = '0'
  ledcWrite(0, bit ? hi : lo);
  delayMicroseconds(BIT_US);
}
'''
    ).substitute(LED=_LED_PIN, CARRIER=p["carrier"], BITRATE=p["data_rate"],
                 DEPTH=p["depth"], PRBS=_prbs_block(p["prbs_order"]))

    rx = Template(
        '''// OptiSim — PWM-ASK receiver (ESP32). Samples the PV node, integrates each
// bit window, thresholds it, scores BER vs the PRBS, then prints the capture in
// OptiSim's Measured-vs-Sim CSV format.
#define ADC_PIN      $ADC
#define ADC_BITS     $ADCBITS
#define SAMPLE_HZ    $SAMPLE
#define BIT_RATE_HZ  $BITRATE
#define N_BITS       $NBITS
const int   SPB   = SAMPLE_HZ / BIT_RATE_HZ;   // ADC samples per bit
const int   NSAMP = N_BITS * SPB;
uint16_t buf[NSAMP];

$PRBS

void setup() {
  Serial.begin(115200);
  analogReadResolution(ADC_BITS);
}

void loop() {
  static bool done = false;
  if (done) { delay(2000); return; }
  done = true;

  // 1. Capture at a fixed rate.
  const uint32_t period_us = 1000000UL / SAMPLE_HZ;
  uint32_t t = micros();
  for (int i = 0; i < NSAMP; i++) {
    buf[i] = analogRead(ADC_PIN);
    t += period_us;
    while ((int32_t)(micros() - t) < 0) { }
  }

  // 2. Envelope threshold = mean of the whole capture; integrate each bit.
  long sum = 0;
  for (int i = 0; i < NSAMP; i++) sum += buf[i];
  uint16_t thresh = sum / NSAMP;
  long errors = 0;
  for (int b = 0; b < N_BITS; b++) {
    long e = 0;
    for (int k = 0; k < SPB; k++) e += buf[b * SPB + k];
    bool rx = (e / SPB) > thresh;
    if (rx != prbs_next()) errors++;
  }
  float ber = (float)errors / N_BITS;

  // 3. Emit the capture for OptiSim's Measured panel.
  Serial.printf("# sample_rate_hz=%d\\n", SAMPLE_HZ);
  Serial.printf("# ber=%.4f\\n", ber);
  for (int i = 0; i < NSAMP; i++) Serial.println(buf[i]);
}
'''
    ).substitute(ADC=_ADC_PIN, ADCBITS=p["adc_bits"], SAMPLE=p["sample_rate"],
                 BITRATE=p["data_rate"], NBITS=p["n_bits"], PRBS=_prbs_block(p["prbs_order"]))
    return tx, rx


# --------------------------------------------------------------------------- #
# BFSK
# --------------------------------------------------------------------------- #

def _bfsk(p: dict) -> tuple[str, str]:
    tx = Template(
        '''// OptiSim — BFSK transmitter (ESP32). Each PRBS bit drives the LED with a
// 50%-duty tone: f0 for '0', f1 for '1'.
#define LED_PIN      $LED
#define F0_HZ        $F0
#define F1_HZ        $F1
#define BIT_RATE_HZ  $BITRATE
const uint32_t BIT_US = 1000000UL / BIT_RATE_HZ;

$PRBS

void setup() {
  ledcSetup(0, F0_HZ, 8);
  ledcAttachPin(LED_PIN, 0);
}

void loop() {
  bool bit = prbs_next();
  ledcChangeFrequency(0, bit ? F1_HZ : F0_HZ, 8);
  ledcWrite(0, 128);                   // 50% duty square tone
  delayMicroseconds(BIT_US);
}
'''
    ).substitute(LED=_LED_PIN, F0=p["f0"], F1=p["f1"], BITRATE=p["data_rate"],
                 PRBS=_prbs_block(p["prbs_order"]))

    rx = Template(
        '''// OptiSim — BFSK receiver (ESP32). Samples the PV node, then for each bit
// compares Goertzel energy at f0 vs f1 to decide the bit. Scores BER vs the
// PRBS and prints the capture in OptiSim's Measured-vs-Sim CSV format.
#define ADC_PIN      $ADC
#define ADC_BITS     $ADCBITS
#define SAMPLE_HZ    $SAMPLE
#define F0_HZ        $F0
#define F1_HZ        $F1
#define BIT_RATE_HZ  $BITRATE
#define N_BITS       $NBITS
const int   SPB   = SAMPLE_HZ / BIT_RATE_HZ;
const int   NSAMP = N_BITS * SPB;
uint16_t buf[NSAMP];

$PRBS

// Goertzel power at frequency f over n samples.
float goertzel(const uint16_t* x, int n, float f) {
  float w = 2.0f * PI * f / SAMPLE_HZ;
  float c = 2.0f * cosf(w);
  float s1 = 0, s2 = 0;
  for (int i = 0; i < n; i++) {
    float s0 = (float)x[i] + c * s1 - s2;
    s2 = s1; s1 = s0;
  }
  return s1 * s1 + s2 * s2 - c * s1 * s2;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(ADC_BITS);
}

void loop() {
  static bool done = false;
  if (done) { delay(2000); return; }
  done = true;

  const uint32_t period_us = 1000000UL / SAMPLE_HZ;
  uint32_t t = micros();
  for (int i = 0; i < NSAMP; i++) {
    buf[i] = analogRead(ADC_PIN);
    t += period_us;
    while ((int32_t)(micros() - t) < 0) { }
  }

  long errors = 0;
  for (int b = 0; b < N_BITS; b++) {
    const uint16_t* w = &buf[b * SPB];
    bool rx = goertzel(w, SPB, F1_HZ) > goertzel(w, SPB, F0_HZ);
    if (rx != prbs_next()) errors++;
  }
  float ber = (float)errors / N_BITS;

  Serial.printf("# sample_rate_hz=%d\\n", SAMPLE_HZ);
  Serial.printf("# ber=%.4f\\n", ber);
  for (int i = 0; i < NSAMP; i++) Serial.println(buf[i]);
}
'''
    ).substitute(ADC=_ADC_PIN, ADCBITS=p["adc_bits"], SAMPLE=p["sample_rate"],
                 F0=p["f0"], F1=p["f1"], BITRATE=p["data_rate"], NBITS=p["n_bits"],
                 PRBS=_prbs_block(p["prbs_order"]))
    return tx, rx


def generate_firmware(cfg) -> dict:
    """Return ``{scheme, tx, rx, notes}`` starter sketches for the config's
    modulation. Raises ValueError for unsupported schemes."""
    p = _params(cfg)
    scheme = p["scheme"]
    if scheme == "PWM_ASK":
        tx, rx = _pwm_ask(p)
    elif scheme == "BFSK":
        tx, rx = _bfsk(p)
    else:
        raise ValueError(
            f"Firmware generation supports PWM_ASK and BFSK; '{scheme or 'unset'}' "
            "is simulator-only (no faithful ESP32 implementation)."
        )
    notes = [
        f"LED on GPIO{_LED_PIN}, PV/ADC on GPIO{_ADC_PIN} — edit to match your wiring.",
        f"{p['data_rate']} bps, {p['sample_rate']} Hz ADC, PRBS-{p['prbs_order']}; "
        "TX and RX share the same PRBS seed.",
        "RX prints CSV for the Measured-vs-Sim panel (generate -> flash -> capture -> compare).",
    ]
    return {"scheme": scheme, "tx": tx, "rx": rx, "notes": notes}

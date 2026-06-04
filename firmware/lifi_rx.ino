// LiFi PoC receiver (ESP32) — PWM-ASK envelope demod
//
// Samples the TL072 gain-chain output on ADC_PIN (GPIO34), envelope-detects per
// bit, thresholds, and decodes. The simulator parses the constants below to set
// the RX PHY (ADC pin / resolution / sample rate); it runs its own demod.

#define ADC_PIN       34        // GPIO34 — TL072 output -> R9 -> ADC
#define SAMPLE_RATE   200000    // Hz — ADC sampling rate
#define DATA_RATE     1000      // bps
#define ADC_BITS      12        // ADC resolution

int threshold = 2048;           // mid-scale slice (re-derived adaptively)

void setup() {
  analogReadResolution(ADC_BITS);
  Serial.begin(115200);
}

void loop() {
  int s = analogRead(ADC_PIN);
  // ... accumulate per-bit envelope, threshold, decode symbol ...
  delayMicroseconds(5);         // ~200 kHz sample loop
}

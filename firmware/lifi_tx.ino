// LiFi PoC transmitter (ESP32) — PWM-ASK
//
// Drives the 2N2222 LED switch on LED_PIN: an ASK carrier gated by the data
// bits, dimmed by a slow PWM envelope. The simulator parses the constants below
// to set the TX PHY (it does not run this code).

#define LED_PIN       25       // GPIO -> R1 -> 2N2222 base
#define CARRIER_FREQ  10000    // Hz  — ASK carrier within each ON symbol
#define PWM_FREQ      10       // Hz  — slow dimming envelope
#define DATA_RATE     1000     // bps — symbol/bit rate
#define MOD_DEPTH     90       // %   — optical modulation depth
#define SYMBOL_MS     1        // ms  — one symbol

// PWM-ASK symbol duty levels (A/B/C)
const int DUTY_A = 25;
const int DUTY_B = 50;
const int DUTY_C = 75;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  ledcSetup(0, CARRIER_FREQ, 8);     // channel 0 at the carrier frequency
  ledcAttachPin(LED_PIN, 0);
  Serial.begin(115200);              // debug only — not the data rate
}

void loop() {
  // ... generate PRBS bits, gate the carrier per symbol, write duty ...
}

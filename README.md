# ECG → ESP32: receive on core 0, predict on core 1, display the exact signal

```
 Browser page (ecg-simulator-esp32.html)          ESP32 (dual core)
 or python/send_ecg.py                  USB
 ┌──────────────────────┐   serial   ┌────────────────────────────────────────────┐
 │ signal(t) @ 250 Hz   │ ─────────► │ CORE 0  rxTask   parse frames, CRC check    │
 │ + age/height/weight  │  115200    │         │  ├─► stream buffer ─► dispTask ─► ILI9341 TFT
 │ + diagnosis (label)  │            │         │  └─► 125 Hz ring buffer (mutex)   (exact samples)
 │                      │ ◄───────── │ CORE 1  mlTask   every 1 s: last 6 s → features → neural net
 │ shows ESP32 result   │  "P,3,97,9,108"      └─► class + confidence → display + back to PC
 └──────────────────────┘            └────────────────────────────────────────────┘
```

| Part | Where |
|---|---|
| Sender web app (Web Serial) | `web/ecg-simulator-esp32.html` |
| Sender in Python (no browser) | `python/send_ecg.py` |
| Signal generator (Python port) + feature extractor | `python/ecg_sim.py` |
| Train the model, export weights | `python/train_model.py` → `firmware/.../model_weights.h` |
| ESP32 sketch | `firmware/esp32_ecg_monitor/esp32_ecg_monitor.ino` |
| On-device features + neural network | `firmware/esp32_ecg_monitor/ecg_ml.h` |

A trained `model_weights.h` is already included, so you can flash straight away.

## 1. Hardware

ESP32 DevKit (any dual-core ESP32) and a 2.4"/2.8" **ILI9341** SPI TFT (320×240).

| TFT pin | ESP32 pin |
|---|---|
| VCC, LED | 3V3 |
| GND | GND |
| CS | GPIO 5 |
| RESET | GPIO 22 |
| DC / RS | GPIO 21 |
| SCK | GPIO 18 |
| MOSI (SDI) | GPIO 23 |
| MISO (SDO) | GPIO 19 (optional) |

Different pins for CS, DC or RESET: edit the three `#define`s at the top of the `.ino`. SCK/MOSI/MISO are the ESP32's fixed VSPI pins.

## 2. Flash the ESP32

1. Arduino IDE → Boards Manager → install **esp32 by Espressif**.
2. Library Manager → install **Adafruit GFX Library** and **Adafruit ILI9341**.
3. Open `firmware/esp32_ecg_monitor/esp32_ecg_monitor.ino` (keep the three files in that folder together).
4. Board: **ESP32 Dev Module** → upload. Then **close the Serial Monitor** (it would hold the port).

The screen shows "AI  waiting for signal" until data arrives.

## 3. Send a signal

**Browser (recommended)**: open `web/ecg-simulator-esp32.html` in Chrome or Edge (double-click the file). Pick a diagnosis and patient, press **Connect ESP32**, choose the port. The page streams the very samples it draws on screen (`signal(t)` at 250 Hz, 1 count = 0.001 mV).

**Python**:
```
pip install numpy scikit-learn pyserial
cd python
python send_ecg.py --port COM5 --dx afib --age 6
python send_ecg.py --port /dev/ttyUSB0 --dx stemi --age 58 --height 175 --weight 92
```

After a diagnosis or patient change, the board needs **6 seconds** to fill its analysis window, then updates once per second.

### What the TFT shows
* Header: the model's prediction and confidence (core 1).
* Middle: the received signal, drawn as it arrives — 25 mm/s, 10 mm/mV, a sweeping pen with an erase bar, ECG grid (core 0).
* `Sent …`: the diagnosis label the sender says it is transmitting, with `MATCH`/`DIFF` against the prediction.
* Bottom: patient, heart rate measured on the board, inference time, `LIVE` / `NO SIGNAL`.

## 4. Why it is built this way
* `rxTask` (priority 3) and `dispTask` (priority 2) share core 0. The UART has a 4 KB receive buffer, so a slow TFT redraw never loses bytes.
* `mlTask` alone owns core 1, so a ~1 s inference cadence never disturbs the drawing.
* Data crosses cores through a FreeRTOS **mutex-protected ring buffer** (ML) and a **stream buffer** (display).

## 5. The model
Input: last 750 samples (6 s @ 125 Hz; the 250 Hz stream is averaged in pairs) plus age, height, weight.
Features (165): R-peak rhythm statistics (rate, irregularity), a beat-aligned average waveform (P/QRS/ST/T shape), 26 spectral bins, amplitude statistics, patient values.
Network: 165 → 64 → 32 → 10 (ReLU, softmax), ≈ 13 k weights (~55 KB flash as float32).
Heart rate is judged **relative to the patient's age** (a 130 bpm rhythm is tachycardia at 35 y but normal at 6 months).

`ecg_ml.h` and `python/ecg_sim.py::extract_features` implement the *same* algorithm. Verified on a PC: features agree to < 1e-4, and 110 streams from the web page's own JavaScript (10 diagnoses × 11 patients) pass through the firmware's parser and classifier with 110/110 correct.

Retrain (also after changing the generator):
```
cd python
python train_model.py --per-class 3000
```
It rewrites `model_weights.h`; re-flash.

## 6. Serial protocol (little-endian)
```
0xA5 0x5A  TYPE  LEN  PAYLOAD[LEN]  CRC8        CRC8: poly 0x07, init 0, over TYPE,LEN,PAYLOAD
TYPE 0x01  samples  int16 × N (0.001 mV per count), 250 Hz, N ≤ 100
TYPE 0x02  config   float32 age_years, float32 height_cm, float32 weight_kg, uint8 dx (0–9, 255 unknown)
ESP32 → PC text:    P,<class 0-9>,<confidence %>,<inference ms>,<heart rate>\r\n
```
Class order: normal sinus, sinus bradycardia, sinus tachycardia, atrial fibrillation, atrial flutter, first-degree AV block, LBBB, STEMI, ventricular tachycardia, ventricular fibrillation.

To feed your own ECG (e.g. an AD8232 on an ADC pin) into the same pipeline, send 250 Hz samples in this format, or push them straight into `handleFrame()`.

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| "Could not open the port" | Close Arduino Serial Monitor / other programs. |
| Browser has no Connect option | Use desktop Chrome or Edge; open the HTML from a file or https. |
| White or blank screen | Recheck wiring; some panels need `LED` on 3V3; try `tft.begin(24000000)` for long wires. |
| Colours or mirroring wrong | Change `tft.setRotation(1)` to 3, or swap the panel driver library. |
| Board resets when the page connects | Normal (DTR toggles reset). The page waits ~2 s before streaming. |
| "NO SIGNAL" while paused | Expected; the sender stops when paused. |

## 8. Limits
* Tested here on a PC: firmware parsing, decimation, feature extraction, network and a syntax/type check of the sketch against stub headers. It has **not been run on real ESP32 + TFT hardware**, so pins, SPI speed and timings may need small adjustments. The inference time printed on screen is the real measurement.
* The network was trained only on this simulator's signals. The ~100 % accuracy shown is on synthetic data and says nothing about real patients; real ECGs have far more variability. Educational / prototyping use only, not a medical device.
* Single lead (lead II). Real diagnosis such as STEMI or bundle branch block needs 12 leads.

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

## 9. Version 2: no display needed, WiFi input, live diagnostics

### Three ways to run it
| Option | How the signal reaches the board | Where you read the results |
|---|---|---|
| **1. USB + web page** | USB cable, page opened in Chrome/Edge | prediction and diagnostics panel on the page |
| **2. WiFi (MQTT) from anywhere** | internet broker, any browser, no cable | same panel, but the board only needs power + WiFi |
| **3. Standalone check** | either of the above | Arduino Serial Monitor (115200) text lines and the blinking LED on GPIO 2 |

### No display yet
Leave `#define USE_DISPLAY 0` in the sketch. Everything else works, and the onboard LED blinks once per second while a signal arrives. Set it to `1` after you wire the TFT.

### WiFi (MQTT) setup
1. Install the **PubSubClient** library (Nick O'Leary).
2. In the sketch set `USE_WIFI 1`, `WIFI_SSID`, `WIFI_PASS` (2.4 GHz WiFi only) and a long secret `ECG_CODE` of your own.
3. Upload. In the web page choose **WiFi**, type the same code, press **Connect WiFi**.
The page and board meet on a public test broker (`broker.hivemq.com`) under the topic `ecgmon/<code>/in` (signal to the board) and `ecgmon/<code>/out` (results back). Anyone who knows or guesses your code can read or send data to it, so use a long random code and only synthetic signals. **Never commit your WiFi password or code to a public GitHub repository**: keep the placeholder values in the copy you publish.

### Diagnostics the board reports every second
`S,key=value,...` lines, shown as cards on the page and readable in the Serial Monitor:
core 0 / core 1 CPU load, samples per second received (expect 250), frames per second, checksum errors, ML time (features and network separately), worst ML time and missed deadlines, analysis-window fill, free and lowest free memory, core-0 wait for shared data, display columns per second and dropped bytes, WiFi signal and reconnects, uptime, and the free stack of every task.
The **Send rate** control (1x, 2x, 5x) is a stress test: it sends samples faster than real time, so predictions become wrong, but you can watch CPU load, errors and timing climb.
If the board stops reporting for 5 seconds the page warns you.

### Verified on a PC, not on hardware
The firmware receive path, CRC handling, MQTT path (through a local broker), ML cycle and report line were run on a PC with stub Arduino headers: 110 of 110 streams classified correctly, corrupted data rejected by the checksum, the page sent about 250 samples per second at 1x and about 1250 at 5x. The CPU-load meter (`esp_register_freertos_idle_hook_for_cpu`) and the real WiFi/MQTT library calls have **not** been run on an ESP32. If the sketch fails to compile because of the CPU meter, set `USE_CPU_METER 0`.

---

## v2: three ways to run it, plus live diagnostics

Edit the switches at the top of `esp32_ecg_monitor.ino`, upload, done.

| Setup | `USE_DISPLAY` | `USE_WIFI` | Where you see results |
|---|---|---|---|
| **1. USB, no display (default)** | 0 | 0 | Web page (prediction + diagnostics), onboard LED on GPIO 2 (blinks 1x/s while a signal arrives), Serial Monitor |
| **2. USB + TFT display** | 1 | 0 | Display + web page. Needs the wiring table above and the *Adafruit GFX* and *Adafruit ILI9341* libraries |
| **3. WiFi (no cable to the PC)** | 0 or 1 | 1 | Web page in the **WiFi** tab (any browser, any computer). Needs the *PubSubClient* library. Fill in `WIFI_SSID`, `WIFI_PASS`, `ECG_CODE` |

WiFi mode: the page and the board both talk to an MQTT broker (default `broker.hivemq.com`, a public test broker). Topics are `ecgmon/<code>/in` (page to board) and `ecgmon/<code>/out` (board to page). Use a long random `ECG_CODE` and type the same code into the page. Anyone who knows the code could read or inject data, and the broker is a third-party service that can change or go down, so for anything private run your own broker. The board needs 2.4 GHz WiFi.

### Diagnostics ("stress") report
Every second the board sends `S,key=value,...`. The page turns it into cards that go green, amber or red: CPU load of core 0 and core 1, samples per second, frame and checksum errors, ML time (features and network separately, worst case, missed deadlines), free and lowest-ever memory, time core 0 waits for core 1, display drops, WiFi signal and reconnects, uptime, and the free stack of every task.
The **Send rate** box (1x, 2x, 5x) pushes the signal faster than real time to load the board. Predictions are meaningless then; watch the cards instead.
The raw lines are in "Raw messages from the board", and also in the Arduino Serial Monitor if the page is not connected.

### Notes on the CPU-load meter
It counts how often each core's idle task runs and compares with a 0.4 s idle-only calibration at boot. It needs the `esp_register_freertos_idle_hook_for_cpu` API, present in Arduino-ESP32 2.x and 3.x; if your core lacks it the cards show "n/a" and everything else still works.

---

## v3: the ESP32's own WiFi network (no internet, no router)

Set in `esp32_ecg_monitor.ino`:
```cpp
#define USE_AP       1          // the board creates its own WiFi network and serves the web page itself
#define USE_WIFI     0          // must be 0 (you cannot use both)
#define AP_SSID      "ECG-Monitor"
#define AP_PASS      "ecg12345"  // at least 8 characters
```
The sketch folder must contain: `esp32_ecg_monitor.ino`, `ecg_ml.h`, `model_weights.h` and **`page_html.h`** (the web page stored in flash).

1. Upload. The Serial Monitor (115200) prints `# AP: started - WiFi name "ECG-Monitor", password "ecg12345" -> open http://192.168.4.1/ in a browser`.
2. On a laptop or phone, join the WiFi network **ECG-Monitor**. (Phones: turn off mobile data. If the device says "no internet", choose to stay connected.)
3. In the browser type **http://192.168.4.1** (include `http://`). The page is served by the board and selects the *ESP32's own WiFi* mode by itself.
4. Press **Connect to this ESP32**, choose a diagnosis and patient.

How it works: the page POSTs the samples as hex text to `/rx` about ten times per second, and polls `/out?since=N` once per second for the newest `P` and `S` lines. Up to 3 devices can join at once. The GitHub copy of the page cannot use this mode (a page from the internet cannot reach the board), it explains that when you click it.

If you change `web/ecg-simulator-esp32.html`, run `python python/make_page_header.py` to rebuild `page_html.h`. It strips the internet-only parts (Google Fonts and the MQTT library) because a device on the board's own WiFi has no internet, and the browser would wait for them before showing the page.

If Arduino reports the sketch is too big, choose Tools, Partition Scheme, **Huge APP (3MB No OTA)**.

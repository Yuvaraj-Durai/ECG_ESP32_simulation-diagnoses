/*
 * ESP32 ECG monitor
 * -----------------------------------------------------------------------------
 *  PC / browser  --USB serial-->  ESP32
 *
 *  CORE 0  rxTask    : receives framed ECG samples + patient info over UART0
 *          dispTask  : draws the exact received signal on an ILI9341 TFT
 *  CORE 1  mlTask    : every second, runs the neural network on the latest 6 s
 *                      and publishes the predicted diagnosis
 *
 *  Libraries (Library Manager):  "Adafruit GFX Library", "Adafruit ILI9341"
 *  Board: "ESP32 Dev Module"  (Arduino-ESP32 core 2.x or 3.x)
 *
 *  Wiring (ILI9341 SPI TFT, 3.3 V):
 *      TFT VCC  -> 3V3        TFT GND  -> GND        TFT LED -> 3V3
 *      TFT CS   -> GPIO 5     TFT RESET-> GPIO 22    TFT DC  -> GPIO 21
 *      TFT SCK  -> GPIO 18    TFT MOSI -> GPIO 23    TFT MISO-> GPIO 19 (optional)
 *
 *  Serial protocol (115200 8N1), all little-endian:
 *      0xA5 0x5A  TYPE  LEN  PAYLOAD[LEN]  CRC8
 *      CRC8 = poly 0x07, init 0, over TYPE, LEN and PAYLOAD
 *      TYPE 0x01  samples : LEN/2 x int16, microvolt-scaled (1 LSB = 0.001 mV), 250 Hz
 *      TYPE 0x02  config  : float age_years, float height_cm, float weight_kg, uint8 dx (0-9, 255 = unknown)
 *  ESP32 -> PC (text, one line per second):  P,<class>,<confidence %>,<inference ms>,<hr bpm>
 */
#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/stream_buffer.h"
#include "ecg_ml.h"

// ------------------------------------------------------------------ settings
#define TFT_CS   5
#define TFT_DC   21
#define TFT_RST  22
#define SERIAL_BAUD 115200

constexpr int    SRC_FS      = 250;      // sample rate of the incoming stream
constexpr int    ML_PERIOD   = 1000;     // ms between predictions
constexpr int    DEC_N       = 1024;     // decimated (125 Hz) ring buffer, >= ecgml::WIN
constexpr int    SIGNAL_TIMEOUT_MS = 1500;

// screen layout (landscape 320 x 240)
constexpr int SCR_W = 320, SCR_H = 240;
constexpr int ECG_Y0 = 32, ECG_H = 160;          // trace area
constexpr int BASE_Y = ECG_Y0 + ECG_H / 2;       // 0 mV line
constexpr int PX_PER_MM = 2;                     // 25 mm/s -> 50 px/s
constexpr int MM_PER_MV = 10;                    // gain 10 mm/mV
constexpr int PX_PER_MV = PX_PER_MM * MM_PER_MV;
constexpr int GROUP  = SRC_FS / (25 * PX_PER_MM);// samples per pixel column = 5
constexpr int GAP    = 10;                       // erase-bar width in pixels

// colours (RGB565)
constexpr uint16_t C_BG = 0x0000, C_GRID = 0x10A2, C_GRID_MAJOR = 0x2945, C_TRACE = 0x07E0;
constexpr uint16_t C_TXT = 0xFFFF, C_DIM = 0x8410, C_OK = 0x07E0, C_WARN = 0xFDA0,
                   C_BAD = 0xF800, C_YEL = 0xFFE0;

// ------------------------------------------------------------------ shared data
struct Patient { float age, h, w; uint8_t dx; };
struct Result  {
  bool     valid;       // false while the window is still filling
  int      cls;
  float    conf;
  float    hr;
  int      beats;
  uint32_t inferMs;
  int      fillPct;     // how full the 6 s window is
};

static SemaphoreHandle_t   g_mtx;
static StreamBufferHandle_t g_dispBuf;          // raw int16 bytes rx -> display
static float               g_dec[DEC_N];        // 125 Hz samples (mV), written by core 0, read by core 1
static uint32_t            g_decHead  = 0;
static uint32_t            g_decTotal = 0;
static volatile uint32_t   g_lastRxMs = 0;
static volatile uint32_t   g_cfgVersion = 0;
static Patient             g_patient = {35.0f, 172.0f, 70.0f, 255};
static Result              g_result  = {false, 0, 0, 0, 0, 0, 0};

// ------------------------------------------------------------------ protocol helpers
static uint8_t crc8(uint8_t crc, uint8_t b) {
  crc ^= b;
  for (int i = 0; i < 8; i++) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  return crc;
}

static void handleFrame(uint8_t type, const uint8_t* p, uint8_t len) {
  static float decAcc = 0;
  static uint8_t decPhase = 0;

  if (type == 0x02 && len == 13) {                       // patient config
    Patient n;
    memcpy(&n.age, p, 4); memcpy(&n.h, p + 4, 4); memcpy(&n.w, p + 8, 4); n.dx = p[12];
    if (!(n.age >= 0 && n.age <= 120)) n.age = 35;
    if (!(n.h >= 30 && n.h <= 250))    n.h = 172;
    if (!(n.w >= 1 && n.w <= 300))     n.w = 70;
    xSemaphoreTake(g_mtx, portMAX_DELAY);
    bool changed = fabsf(n.age - g_patient.age) > 0.001f || fabsf(n.h - g_patient.h) > 0.01f ||
                   fabsf(n.w - g_patient.w) > 0.01f || n.dx != g_patient.dx;
    if (changed) {
      g_patient = n;
      g_decHead = 0; g_decTotal = 0; decAcc = 0; decPhase = 0;   // new patient/diagnosis -> new window
      g_cfgVersion = g_cfgVersion + 1;
    }
    xSemaphoreGive(g_mtx);
    if (changed) xStreamBufferReset(g_dispBuf);
  }
  else if (type == 0x01 && len >= 2 && (len & 1) == 0) { // ECG samples
    xStreamBufferSend(g_dispBuf, p, len, 0);             // for the display (drop if full)
    xSemaphoreTake(g_mtx, portMAX_DELAY);
    for (int i = 0; i < len; i += 2) {                   // 250 Hz -> 125 Hz (average pairs) for the ML window
      int16_t s = (int16_t)(p[i] | (p[i + 1] << 8));
      decAcc += s * 0.001f;
      if (++decPhase == 2) {
        g_dec[g_decHead] = decAcc * 0.5f;
        g_decHead = (g_decHead + 1) % DEC_N;
        if (g_decTotal < 0x0FFFFFFF) g_decTotal++;
        decAcc = 0; decPhase = 0;
      }
    }
    xSemaphoreGive(g_mtx);
    g_lastRxMs = millis();
  }
}

// ------------------------------------------------------------------ CORE 0: serial receiver
static void rxTask(void*) {
  enum { S1, S2, TYPE, LEN, PAY, CRC } st = S1;
  uint8_t type = 0, len = 0, idx = 0, crc = 0;
  static uint8_t pay[256];
  for (;;) {
    int avail = Serial.available();
    if (avail <= 0) { vTaskDelay(1); continue; }
    while (avail-- > 0) {
      uint8_t b = (uint8_t)Serial.read();
      switch (st) {
        case S1:   if (b == 0xA5) st = S2; break;
        case S2:   st = (b == 0x5A) ? TYPE : (b == 0xA5 ? S2 : S1); break;
        case TYPE: type = b; crc = crc8(0, b); st = LEN; break;
        case LEN:  len = b; crc = crc8(crc, b); idx = 0; st = len ? PAY : CRC; break;
        case PAY:  pay[idx++] = b; crc = crc8(crc, b); if (idx >= len) st = CRC; break;
        case CRC:  if (b == crc) handleFrame(type, pay, len); st = S1; break;
      }
    }
  }
}

// ------------------------------------------------------------------ CORE 1: machine learning
static float s_win[ecgml::WIN];
static float s_feat[ecgml::NFEAT];

static void mlTask(void*) {
  float hist[3][ecgml::NCLASS];
  int histN = 0, histPos = 0;
  uint32_t seenVersion = 0;

  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(ML_PERIOD));

    bool ready = false;
    Patient p;
    int fill;
    xSemaphoreTake(g_mtx, portMAX_DELAY);
    p = g_patient;
    if (g_cfgVersion != seenVersion) { seenVersion = g_cfgVersion; histN = 0; histPos = 0; }
    fill = (int)((g_decTotal > (uint32_t)ecgml::WIN ? (uint32_t)ecgml::WIN : g_decTotal) * 100 / ecgml::WIN);
    if (g_decTotal >= (uint32_t)ecgml::WIN && (millis() - g_lastRxMs) < (uint32_t)SIGNAL_TIMEOUT_MS) {
      int start = (int)((g_decHead + DEC_N - ecgml::WIN) % DEC_N);
      for (int i = 0; i < ecgml::WIN; i++) s_win[i] = g_dec[(start + i) % DEC_N];
      ready = true;
    }
    xSemaphoreGive(g_mtx);

    if (!ready) {
      xSemaphoreTake(g_mtx, portMAX_DELAY);
      g_result.valid = false; g_result.fillPct = fill;
      xSemaphoreGive(g_mtx);
      histN = 0; histPos = 0;
      continue;
    }

    uint32_t t0 = micros();
    ecgml::Info info;
    float probs[ecgml::NCLASS];
    ecgml::extractFeatures(s_win, p.age, p.h, p.w, s_feat, &info);
    ecgml::predict(s_feat, probs);
    uint32_t us = micros() - t0;

    // average the last 3 predictions so the label does not flicker
    memcpy(hist[histPos], probs, sizeof(probs));
    histPos = (histPos + 1) % 3;
    if (histN < 3) histN++;
    float avg[ecgml::NCLASS] = {0};
    for (int k = 0; k < histN; k++) for (int c = 0; c < ecgml::NCLASS; c++) avg[c] += hist[k][c] / histN;
    int best = 0;
    for (int c = 1; c < ecgml::NCLASS; c++) if (avg[c] > avg[best]) best = c;

    xSemaphoreTake(g_mtx, portMAX_DELAY);
    g_result.valid = true;   g_result.cls = best;   g_result.conf = avg[best];
    g_result.hr = info.hr;   g_result.beats = info.beats;
    g_result.inferMs = (us + 500) / 1000;  g_result.fillPct = 100;
    xSemaphoreGive(g_mtx);

    Serial.printf("P,%d,%d,%lu,%d\r\n", best, (int)(avg[best] * 100 + 0.5f), (unsigned long)((us + 500) / 1000), (int)(info.hr + 0.5f));
  }
}

// ------------------------------------------------------------------ CORE 0: display
static Adafruit_ILI9341 tft(TFT_CS, TFT_DC, TFT_RST);
static uint16_t colGrid[ECG_H], colPlain[ECG_H];

struct Line { char txt[48]; uint16_t fg; };
static Line lineCache[4];

static void paintColumn(int x) {
  tft.startWrite();
  tft.setAddrWindow(x, ECG_Y0, 1, ECG_H);
  tft.writePixels((x % 10 == 0) ? colGrid : colPlain, ECG_H);
  tft.endWrite();
}

static void clearTrace() {
  for (int x = 0; x < SCR_W; x++) paintColumn(x);
}

static void drawLine(int slot, int y, uint8_t size, uint16_t fg, const char* s, int width) {
  char buf[48];
  snprintf(buf, sizeof(buf), "%-*.*s", width, width, s);
  if (strcmp(buf, lineCache[slot].txt) == 0 && fg == lineCache[slot].fg) return;
  strcpy(lineCache[slot].txt, buf); lineCache[slot].fg = fg;
  tft.setTextSize(size);
  tft.setTextColor(fg, C_BG);
  tft.setCursor(4, y);
  tft.print(buf);
}

static void fmtAge(char* out, size_t n, float a) {
  if (a < 1.0f / 12.0f)     snprintf(out, n, "%dd", (int)(a * 365 + 0.5f));
  else if (a < 2.0f)        snprintf(out, n, "%dmo", (int)(a * 12 + 0.5f));
  else                      snprintf(out, n, "%dy", (int)(a + 0.5f));
}

static uint16_t classColor(int c) {
  if (c == 0) return C_OK;
  if (c == 1 || c == 2) return C_YEL;
  if (c >= 7) return C_BAD;
  return C_WARN;
}

static void updateTexts() {
  Patient p; Result r;
  xSemaphoreTake(g_mtx, portMAX_DELAY);
  p = g_patient; r = g_result;
  xSemaphoreGive(g_mtx);
  bool live = (millis() - g_lastRxMs) < (uint32_t)SIGNAL_TIMEOUT_MS && g_lastRxMs != 0;
  char b[64];

  // header: prediction from core 1
  if (!live)          { drawLine(0, 8, 2, C_DIM, "AI  waiting for signal", 25); }
  else if (!r.valid)  { snprintf(b, sizeof(b), "AI  collecting %d%%", r.fillPct); drawLine(0, 8, 2, C_YEL, b, 25); }
  else {
    snprintf(b, sizeof(b), "AI %-14s %3d%%", ecgml::CLASS_SHORT[r.cls], (int)(r.conf * 100 + 0.5f));
    drawLine(0, 8, 2, classColor(r.cls), b, 25);
  }

  // what the sender says it is transmitting, and whether the AI agrees
  if (p.dx < ecgml::NCLASS) {
    bool match = r.valid && live && r.cls == p.dx;
    snprintf(b, sizeof(b), "Sent %-14s %s", ecgml::CLASS_SHORT[p.dx], (r.valid && live) ? (match ? "MATCH" : "DIFF") : "");
    drawLine(1, 198, 2, (r.valid && live) ? (match ? C_OK : C_BAD) : C_TXT, b, 25);
  } else {
    drawLine(1, 198, 2, C_DIM, "Sent (unlabelled signal)", 25);
  }

  char ag[12]; fmtAge(ag, sizeof(ag), p.age);
  snprintf(b, sizeof(b), "Age %s  %dcm  %.1fkg   HR %d bpm  beats %d", ag, (int)(p.h + 0.5f), p.w,
           (int)(r.hr + 0.5f), r.beats);
  drawLine(2, 220, 1, C_TXT, b, 52);

  snprintf(b, sizeof(b), "%s   ML %lu ms   25 mm/s 10 mm/mV", live ? "LIVE" : "NO SIGNAL", (unsigned long)r.inferMs);
  drawLine(3, 230, 1, live ? C_OK : C_BAD, b, 52);
}

static void resetScreenLayout() {
  tft.fillScreen(C_BG);
  tft.drawFastHLine(0, ECG_Y0 - 2, SCR_W, C_GRID_MAJOR);
  tft.drawFastHLine(0, ECG_Y0 + ECG_H + 1, SCR_W, C_GRID_MAJOR);
  clearTrace();
  for (int i = 0; i < 4; i++) { lineCache[i].txt[0] = 1; lineCache[i].txt[1] = 0; }   // force redraw
}

static void dispTask(void*) {
  tft.begin(40000000);
  tft.setRotation(1);                    // 320 x 240 landscape
  for (int y = 0; y < ECG_H; y++) {
    colGrid[y]  = C_GRID_MAJOR;
    colPlain[y] = (((y - ECG_H / 2) % 10) == 0) ? C_GRID : C_BG;
  }
  resetScreenLayout();

  uint32_t seenVersion = g_cfgVersion;
  uint32_t lastText = 0;
  int x = 0, prevY = BASE_Y;
  bool havePrev = false;
  int16_t grp[GROUP]; int gn = 0;
  uint8_t tmp[64];

  for (;;) {
    if (g_cfgVersion != seenVersion) {          // patient or diagnosis changed: start a clean trace
      seenVersion = g_cfgVersion;
      clearTrace(); x = 0; havePrev = false; gn = 0;
    }

    size_t n;
    while ((n = xStreamBufferReceive(g_dispBuf, tmp, sizeof(tmp), 0)) >= 2) {
      for (size_t i = 0; i + 1 < n; i += 2) {
        grp[gn++] = (int16_t)(tmp[i] | (tmp[i + 1] << 8));
        if (gn == GROUP) {                      // 5 samples = 1 pixel column at 25 mm/s
          gn = 0;
          int ymin = 9999, ymax = -9999, yLast = BASE_Y, yFirst = BASE_Y;
          for (int k = 0; k < GROUP; k++) {
            int y = BASE_Y - (int)lroundf(grp[k] * 0.001f * PX_PER_MV);
            if (y < ECG_Y0 + 1) y = ECG_Y0 + 1;
            if (y > ECG_Y0 + ECG_H - 2) y = ECG_Y0 + ECG_H - 2;
            if (k == 0) yFirst = y;
            if (y < ymin) ymin = y;
            if (y > ymax) ymax = y;
            yLast = y;
          }
          paintColumn((x + GAP) % SCR_W);       // erase-bar just ahead of the pen
          if (havePrev) tft.drawLine(x - 1, prevY, x, yFirst, C_TRACE);
          tft.drawFastVLine(x, ymin, ymax - ymin + 1, C_TRACE);
          prevY = yLast; havePrev = true;
          x++;
          if (x >= SCR_W) { x = 0; havePrev = false; }
        }
      }
    }

    if (millis() - lastText > 250) { lastText = millis(); updateTexts(); }
    vTaskDelay(pdMS_TO_TICKS(8));
  }
}

// ------------------------------------------------------------------ Arduino entry points
void setup() {
  Serial.setRxBufferSize(4096);
  Serial.begin(SERIAL_BAUD);

  g_mtx     = xSemaphoreCreateMutex();
  g_dispBuf = xStreamBufferCreate(4096, 1);

  //                                  name    stack  arg  prio  handle  core
  xTaskCreatePinnedToCore(rxTask,   "rx",    4096, NULL, 3, NULL, 0);   // core 0: receive
  xTaskCreatePinnedToCore(dispTask, "disp",  8192, NULL, 2, NULL, 0);   // core 0: display
  xTaskCreatePinnedToCore(mlTask,   "ml",   12288, NULL, 1, NULL, 1);   // core 1: ML model
}

void loop() {
  vTaskDelay(portMAX_DELAY);   // everything happens in the tasks above
}

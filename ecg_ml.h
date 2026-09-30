// ecg_ml.h - on-device ECG classifier (feature extraction + tiny neural network)
//
// Pure C++, no Arduino dependencies, so it can also be compiled and tested on a PC.
// The feature algorithm is a line-by-line twin of python/ecg_sim.py::extract_features().
// If you change one, change the other and retrain (python/train_model.py).
//
// Input : 750 samples of lead II in mV at 125 Hz (6 s window) + patient age/height/weight
// Output: 10 class probabilities
#pragma once
#include <math.h>
#include <stdint.h>
#include "model_weights.h"

namespace ecgml {

constexpr int FS = 125;
constexpr int WIN = 750;
constexpr int PRE = 50;
constexpr int POST = 75;
constexpr int TL = PRE + POST;      // template length
constexpr int REFRACT = 25;         // 0.2 s
constexpr int NSPEC = 26;
constexpr int NFEAT = 8 + TL + NSPEC + 3 + 3;   // 165
constexpr int NCLASS = 10;

static_assert(MODEL_N_IN == NFEAT, "model_weights.h does not match the feature vector - retrain");
static_assert(MODEL_N_OUT == NCLASS, "model_weights.h must have 10 outputs");

static const char* const CLASS_NAMES[NCLASS] = {
  "Normal sinus rhythm", "Sinus bradycardia", "Sinus tachycardia", "Atrial fibrillation",
  "Atrial flutter", "First-degree AV block", "Left bundle branch block", "ST-elevation MI",
  "Ventricular tachycardia", "Ventricular fibrillation"};
static const char* const CLASS_SHORT[NCLASS] = {
  "Normal sinus", "Sinus brady", "Sinus tachy", "Atrial fib", "Atrial flutter",
  "1st-deg AV blk", "LBBB", "STEMI", "V-tach", "V-fib"};

static const float SPEC_F[NSPEC] = {
  0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
  8.0f, 8.5f, 9.0f, 9.5f, 10.0f, 12.0f, 15.0f, 20.0f, 25.0f, 30.0f, 40.0f};

// expected resting heart rate by age (years) - same table as the simulator
static const float HR_TAB[][2] = {
  {0.03f, 145}, {0.5f, 130}, {1, 120}, {3, 105}, {5, 98}, {8, 88}, {12, 80},
  {16, 74}, {30, 72}, {65, 70}, {100, 66}};

inline float lerpTable(const float tab[][2], int n, float x) {
  if (x <= tab[0][0]) return tab[0][1];
  for (int i = 1; i < n; i++) {
    if (x <= tab[i][0]) {
      float x0 = tab[i - 1][0], y0 = tab[i - 1][1], x1 = tab[i][0], y1 = tab[i][1];
      return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
  }
  return tab[n - 1][1];
}

struct Info {          // by-products of feature extraction that the UI can show
  float hr;            // bpm measured from R peaks (0 if not measurable)
  int   beats;         // R peaks found
  float cv;            // RR variability
};

// Builds the 165-value feature vector. `x` = WIN samples (mV, 125 Hz).
inline void extractFeatures(const float* x, float age, float h, float w, float* f, Info* info = nullptr) {
  // scratch memory is static so it never lands on a task stack
  static float  d[WIN];
  static double cs[WIN + 1];
  static double env[WIN];
  static int    pk[WIN / 4];
  static int    rp[WIN / 4];
  static double tm[TL];
  const int n = WIN;

  for (int i = 0; i < NFEAT; i++) f[i] = 0.0f;

  double sum = 0;
  float mn = x[0], mxv = x[0];
  for (int i = 0; i < n; i++) { sum += x[i]; if (x[i] < mn) mn = x[i]; if (x[i] > mxv) mxv = x[i]; }
  const double mu = sum / n;
  double ss = 0;
  for (int i = 0; i < n; i++) { double q = (double)x[i] - mu; ss += q * q; }
  const double sd = sqrt(ss / n);
  const double p2p = (double)(mxv - mn);

  // slope energy -> smoothed envelope
  d[0] = 0.0f;
  float maxAbsD = 0.0f;
  for (int i = 1; i < n; i++) {
    d[i] = x[i] - x[i - 1];
    float a = fabsf(d[i]);
    if (a > maxAbsD) maxAbsD = a;
  }
  cs[0] = 0.0;
  for (int i = 0; i < n; i++) cs[i + 1] = cs[i] + (double)d[i] * (double)d[i];
  double mx = 0.0;
  for (int i = 0; i < n; i++) {
    int hi = i + 5 < n ? i + 5 : n;
    int lo = i - 5 > 0 ? i - 5 : 0;
    env[i] = cs[hi] - cs[lo];
    if (env[i] > mx) mx = env[i];
  }
  const double thr = 0.30 * mx;

  // peak picking with 0.2 s refractory period
  int npkRaw = 0;
  if (mx > 1e-6) {
    int last = -1;
    for (int i = 1; i < n - 1; i++) {
      double e = env[i];
      if (e > thr && e >= env[i - 1] && e > env[i + 1]) {
        if (last < 0 || i - last >= REFRACT) {
          if (npkRaw < WIN / 4) { pk[npkRaw++] = i; last = i; }
        } else if (e > env[last]) {
          pk[npkRaw - 1] = i; last = i;
        }
      }
    }
  }
  // refine each peak to the largest deviation from the mean
  int npk = npkRaw;
  for (int k = 0; k < npk; k++) {
    int p = pk[k];
    int lo = p - 8 > 0 ? p - 8 : 0;
    int hi = p + 8 < n - 1 ? p + 8 : n - 1;
    int best = lo;
    double bv = fabs((double)x[lo] - mu);
    for (int i = lo + 1; i <= hi; i++) {
      double v = fabs((double)x[i] - mu);
      if (v > bv) { bv = v; best = i; }
    }
    rp[k] = best;
  }

  double meanRR = 0, cv = 0, rmssd = 0, maxmin = 0, hr = 0;
  if (npk >= 2) {
    static double rr[WIN / 4];
    int m = npk - 1;
    for (int i = 0; i < m; i++) { rr[i] = (double)(rp[i + 1] - rp[i]) / (double)FS; meanRR += rr[i]; }
    meanRR /= m;
    double v = 0, rmin = rr[0], rmax = rr[0];
    for (int i = 0; i < m; i++) {
      double q = rr[i] - meanRR; v += q * q;
      if (rr[i] < rmin) rmin = rr[i];
      if (rr[i] > rmax) rmax = rr[i];
    }
    cv = sqrt(v / m) / meanRR;
    if (m >= 2) {
      double s2 = 0;
      for (int i = 1; i < m; i++) { double q = rr[i] - rr[i - 1]; s2 += q * q; }
      rmssd = sqrt(s2 / (m - 1));
    }
    maxmin = rmax / (rmin > 1e-3 ? rmin : 1e-3);
    hr = 60.0 / meanRR;
  }

  // beat-aligned average template
  for (int i = 0; i < TL; i++) tm[i] = 0.0;
  int nb = 0;
  for (int k = 0; k < npk; k++) {
    int r = rp[k];
    if (r - PRE >= 0 && r + POST <= n) {
      for (int i = 0; i < TL; i++) tm[i] += (double)x[r - PRE + i];
      nb++;
    }
  }
  if (nb) {
    double base = 0;
    for (int i = 0; i < TL; i++) tm[i] /= nb;
    for (int i = 0; i < 5; i++) base += tm[i];
    base /= 5.0;
    for (int i = 0; i < TL; i++) tm[i] -= base;
  }

  // rhythm scalars
  f[0] = (float)((npk < 15 ? npk : 15) / 15.0);
  f[1] = (float)((hr < 300.0 ? hr : 300.0) / 200.0);
  if (hr > 0) {
    double ratio = hr / lerpTable(HR_TAB, 11, age);
    f[2] = (float)(ratio < 4.0 ? ratio : 4.0);
  }
  f[3] = (float)(meanRR < 6.0 ? meanRR : 6.0);
  f[4] = (float)(cv < 1.0 ? cv : 1.0);
  if (meanRR > 0) { double q = rmssd / meanRR; f[5] = (float)(q < 1.0 ? q : 1.0); }
  f[6] = (float)((maxmin < 5.0 ? maxmin : 5.0) / 5.0);
  f[7] = (float)(nb / 10.0);
  for (int i = 0; i < TL; i++) f[8 + i] = (float)tm[i];

  // spectrum: Goertzel on a Hann-windowed, mean-removed copy
  static float  hann[WIN];
  static bool   hannReady = false;
  if (!hannReady) {
    for (int i = 0; i < n; i++) hann[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (float)(n - 1));
    hannReady = true;
  }
  double pw[NSPEC];
  double tot = 0;
  for (int j = 0; j < NSPEC; j++) {
    float coeff = 2.0f * cosf(2.0f * (float)M_PI * SPEC_F[j] / (float)FS);
    float s1 = 0, s2 = 0;
    for (int i = 0; i < n; i++) {
      float y = (x[i] - (float)mu) * hann[i];
      float s0 = y + coeff * s1 - s2;
      s2 = s1; s1 = s0;
    }
    double p = (double)s1 * s1 + (double)s2 * s2 - (double)coeff * s1 * s2;
    if (p < 0) p = 0;
    pw[j] = p; tot += p;
  }
  for (int j = 0; j < NSPEC; j++) f[8 + TL + j] = (float)sqrt(pw[j] / (tot + 1e-12));

  const int o = 8 + TL + NSPEC;
  f[o]     = (float)sd;
  f[o + 1] = (float)p2p;
  f[o + 2] = maxAbsD * (float)FS / 50.0f;
  f[o + 3] = logf(1.0f + age) / logf(101.0f);
  f[o + 4] = h / 200.0f;
  f[o + 5] = w / 120.0f;

  if (info) { info->hr = (float)hr; info->beats = npk; info->cv = (float)cv; }
}

// Two hidden layers (ReLU) + softmax. `f` is the raw feature vector.
inline void predict(const float* f, float* probs) {
  static float a0[MODEL_N_IN], a1[MODEL_N_H1], a2[MODEL_N_H2], z[MODEL_N_OUT];
  for (int i = 0; i < MODEL_N_IN; i++) a0[i] = (f[i] - MODEL_MEAN[i]) / MODEL_STD[i];

  for (int j = 0; j < MODEL_N_H1; j++) a1[j] = MODEL_B1[j];
  for (int i = 0; i < MODEL_N_IN; i++) {
    const float v = a0[i];
    const float* row = &MODEL_W1[i * MODEL_N_H1];
    for (int j = 0; j < MODEL_N_H1; j++) a1[j] += v * row[j];
  }
  for (int j = 0; j < MODEL_N_H1; j++) if (a1[j] < 0) a1[j] = 0;

  for (int j = 0; j < MODEL_N_H2; j++) a2[j] = MODEL_B2[j];
  for (int i = 0; i < MODEL_N_H1; i++) {
    const float v = a1[i];
    const float* row = &MODEL_W2[i * MODEL_N_H2];
    for (int j = 0; j < MODEL_N_H2; j++) a2[j] += v * row[j];
  }
  for (int j = 0; j < MODEL_N_H2; j++) if (a2[j] < 0) a2[j] = 0;

  for (int j = 0; j < MODEL_N_OUT; j++) z[j] = MODEL_B3[j];
  for (int i = 0; i < MODEL_N_H2; i++) {
    const float v = a2[i];
    const float* row = &MODEL_W3[i * MODEL_N_OUT];
    for (int j = 0; j < MODEL_N_OUT; j++) z[j] += v * row[j];
  }
  float m = z[0];
  for (int j = 1; j < MODEL_N_OUT; j++) if (z[j] > m) m = z[j];
  float s = 0;
  for (int j = 0; j < MODEL_N_OUT; j++) { probs[j] = expf(z[j] - m); s += probs[j]; }
  for (int j = 0; j < MODEL_N_OUT; j++) probs[j] /= s;
}

}  // namespace ecgml

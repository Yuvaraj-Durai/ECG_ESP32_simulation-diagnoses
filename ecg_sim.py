"""
ecg_sim.py  -  Python port of the browser ECG simulator + the feature extractor
used by the ESP32 firmware.

* simulate()          -> synthetic lead-II ECG (mV) for one of 10 diagnoses,
                         scaled to a patient's age / height / weight.
* extract_features()  -> EXACTLY the same feature vector that ecg_ml.h computes
                         on the ESP32 (125 Hz, 750-sample = 6 s window).

Only needs numpy.
"""
import numpy as np

TAU = 2 * np.pi

DX_IDS = ['nsr', 'brady', 'tachy', 'afib', 'flutter', 'avb1', 'lbbb', 'stemi', 'vt', 'vf']
DX_NAMES = ['Normal sinus rhythm', 'Sinus bradycardia', 'Sinus tachycardia',
            'Atrial fibrillation', 'Atrial flutter', 'First-degree AV block',
            'Left bundle branch block', 'ST-elevation MI',
            'Ventricular tachycardia', 'Ventricular fibrillation']
DX_SHORT = ['Normal sinus', 'Sinus brady', 'Sinus tachy', 'Atrial fib', 'Atrial flutter',
            '1st-deg AV blk', 'LBBB', 'STEMI', 'V-tach', 'V-fib']

# ------------------------------------------------------------------ age tables
HR_T   = [[0.03, 145], [0.5, 130], [1, 120], [3, 105], [5, 98], [8, 88], [12, 80],
          [16, 74], [30, 72], [65, 70], [100, 66]]
PR_T   = [[0, .10], [1, .11], [3, .12], [8, .13], [12, .14], [16, .15], [25, .16], [80, .17], [100, .18]]
QRS_T  = [[0, .06], [1, .065], [5, .07], [12, .08], [16, .085], [25, .09], [80, .095], [100, .10]]
H_T    = [[0, 50], [1, 75], [2, 87], [5, 108], [8, 128], [10, 138], [12, 149], [14, 163],
          [16, 170], [18, 172], [25, 172], [80, 168], [100, 165]]
W_T    = [[0, 3.4], [1, 9.5], [2, 12], [5, 18], [8, 26], [10, 32], [12, 42], [14, 52],
          [16, 60], [18, 66], [25, 70], [80, 70], [100, 68]]
AMP_T  = [[0, 1.35], [2, 1.35], [6, 1.28], [12, 1.15], [18, 1.05], [30, 1.0], [60, 0.92], [100, 0.8]]
FLUT_T = [[0, 6.7], [1, 6.0], [12, 5.4], [16, 5.0]]


def lerp(tab, x):
    return float(np.interp(x, [a for a, _ in tab], [b for _, b in tab]))


def typical_for(age):
    return round(lerp(H_T, age)), round(lerp(W_T, age) * 10) / 10


def patient_model(age, h, w):
    th, tw = typical_for(age)
    bmi, bmi_exp = w / (h / 100) ** 2, tw / (th / 100) ** 2
    bsa, bsa_exp = np.sqrt(h * w / 3600), np.sqrt(th * tw / 3600)
    size = float(np.clip(bsa / bsa_exp, 0.7, 1.4))
    ratio = bmi / bmi_exp
    amp = float(np.clip(lerp(AMP_T, age) * np.clip(1 - 0.35 * (ratio - 1), 0.6, 1.3), 0.5, 2.0))
    return dict(age=age, h=h, w=w, bmi=bmi, bsa=bsa, size=size, amp=amp,
                hrN=lerp(HR_T, age), prN=lerp(PR_T, age), qrsN=lerp(QRS_T, age),
                qrs=lerp(QRS_T, age) * (1 + 0.15 * (size - 1)),
                rsa=0.06 if age < 18 else (0.035 if age < 65 else 0.018),
                qtc=0.40 + (0.02 if age < 0.5 else 0.0))


def make_profile(dx, P):
    age = P['age']
    pf = dict(id=dx, mode='regular', hr=P['hrN'], pr=P['prN'], qrs=P['qrs'], hasP=True,
              pDur=0.07 if age < 12 else 0.09, pAmp=0.13 * np.sqrt(P['amp']),
              rAmp=P['amp'], qAmp=-0.09, sAmp=-0.14, tAmp=0.28 * P['amp'], stAmp=0.0,
              notch=False, qtc=P['qtc'], rsa=P['rsa'], fAmp=0.0, flutHz=0.0, block=0)
    if dx == 'brady':
        pf['hr'] = P['hrN'] * 0.62
    elif dx == 'tachy':
        pf['hr'] = min(P['hrN'] * 1.65, 215)
    elif dx == 'afib':
        pf.update(mode='afib', hasP=False, hr=min(P['hrN'] * 1.5, 200), fAmp=0.05 * np.sqrt(P['amp']))
    elif dx == 'flutter':
        pf.update(mode='flutter', hasP=False, flutHz=lerp(FLUT_T, age),
                  block=2 if age < 1 else (3 if age < 12 else 4), fAmp=0.11 * np.sqrt(P['amp']))
        pf['hr'] = 60 * pf['flutHz'] / pf['block']
    elif dx == 'avb1':
        pf['pr'] = min(P['prN'] + 0.14, 0.42)
    elif dx == 'lbbb':
        pf.update(qrs=max(P['qrs'] + 0.055, 0.125), notch=True, qAmp=0, sAmp=0,
                  rAmp=P['amp'] * 0.9, tAmp=-0.3 * P['amp'], stAmp=-0.12 * P['amp'])
    elif dx == 'stemi':
        pf.update(hr=P['hrN'] * 1.15, stAmp=0.4 * P['amp'], tAmp=0.5 * P['amp'], qAmp=-0.3, sAmp=-0.03)
    elif dx == 'vt':
        pf.update(mode='vt', hasP=False, hr=float(np.clip(P['hrN'] * 2.3, 150, 240)), rAmp=1.1 * P['amp'])
    elif dx == 'vf':
        pf.update(mode='vf', hasP=False, rAmp=0.7 * P['amp'])
    return pf


def apply_jitter(pf, dx, rng):
    """Small random variation so the ML model does not over-fit exact parameter values."""
    u = rng.uniform
    hr_rng = dict(nsr=(0.85, 1.25), avb1=(0.85, 1.25), lbbb=(0.85, 1.25), brady=(0.85, 1.15),
                  tachy=(0.88, 1.15), afib=(0.8, 1.2), stemi=(0.9, 1.15), vt=(0.9, 1.15))
    if dx in hr_rng:
        pf['hr'] = max(20.0, pf['hr'] * u(*hr_rng[dx]))
    if dx == 'flutter':
        pf['flutHz'] *= u(0.95, 1.05)
        pf['hr'] = 60 * pf['flutHz'] / pf['block']
    pf['qrs'] *= u(0.92, 1.08)
    pf['pr'] *= u(0.92, 1.08)
    pf['rAmp'] *= u(0.85, 1.15)
    pf['tAmp'] *= u(0.85, 1.15)
    if dx == 'stemi':
        pf['stAmp'] *= u(0.6, 1.2)
    return pf


# ------------------------------------------------------------------ waveform
def _g(x, c, s, a):
    return a * np.exp(-0.5 * ((x - c) / s) ** 2)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _next_rr(pf, t, rng):
    base = 60.0 / pf['hr']
    m = pf['mode']
    if m == 'afib':
        return max(0.28, base * (0.55 + 0.9 * rng.random()))
    if m == 'flutter':
        return pf['block'] / pf['flutHz']
    if m == 'vt':
        return base * (1 + 0.01 * rng.standard_normal())
    return base * (1 + pf['rsa'] * np.sin(TAU * 0.28 * t) + 0.008 * rng.standard_normal())


def _schedule(pf, t_from, t_to, rng):
    beats = []
    if pf['mode'] == 'vf':
        return beats
    prev = pf['block'] / pf['flutHz'] if pf['mode'] == 'flutter' else 60.0 / pf['hr']
    t = t_from
    while t < t_to:
        qt = float(np.clip(pf['qtc'] * np.sqrt(np.clip(prev, 0.25, 1.6)), 0.18, 0.5))
        beats.append((t, qt, 1 + 0.03 * rng.standard_normal(), 1 + 0.05 * rng.standard_normal()))
        rr = _next_rr(pf, t, rng)
        prev, t = rr, t + rr
    return beats


def _beat(dt, qt, k, k2, pf):
    q, A = pf['qrs'], pf['rAmp'] * k
    if pf['mode'] == 'vt':
        return A * (_g(dt, 0, 0.045, 1) - 0.55 * _g(dt, 0.12, 0.06, 1))
    v, qOn = np.zeros_like(dt), -0.4 * q
    if pf['hasP']:
        v += _g(dt, qOn - pf['pr'] + pf['pDur'] / 2, pf['pDur'] / 4.5, pf['pAmp'] * k)
    if pf['notch']:
        v += _g(dt, -0.18 * q, 0.14 * q, 0.85 * A) + _g(dt, 0.18 * q, 0.14 * q, 0.95 * A)
    else:
        v += (_g(dt, -0.32 * q, 0.11 * q, pf['qAmp'] * A) + _g(dt, 0, 0.13 * q, A)
              + _g(dt, 0.30 * q, 0.12 * q, pf['sAmp'] * A))
    v += _g(dt, qOn + 0.72 * qt, 0.11 * qt, pf['tAmp'] * k2)
    if pf['stAmp']:
        v += pf['stAmp'] * (_sig((dt - 0.6 * q) / 0.008) - _sig((dt - (qOn + 0.98 * qt)) / 0.03))
    return v


def _background(t, pf, noise, t0):
    v, m = np.zeros_like(t), pf['mode']
    if m == 'afib':
        v += pf['fAmp'] * (np.sin(TAU * 5.1 * t + 0.3) * (0.7 + 0.3 * np.sin(TAU * 0.31 * t))
                           + 0.8 * np.sin(TAU * 6.3 * t + 1.7) * (0.7 + 0.3 * np.sin(TAU * 0.23 * t + 1))
                           + 0.6 * np.sin(TAU * 7.7 * t + 4.0) * (0.7 + 0.3 * np.sin(TAU * 0.41 * t + 2))) / 1.6
    elif m == 'flutter':
        ph = (t - t0) * pf['flutHz']
        s = sum(np.sin(TAU * k * ph) / k for k in range(1, 5))
        v += pf['fAmp'] * (2 / np.pi) * s
    elif m == 'vf':
        env = 0.6 + 0.4 * np.sin(TAU * 0.29 * t + 0.5)
        s2 = (np.sin(TAU * 4.1 * t + 1.5 * np.sin(TAU * 0.31 * t))
              + 0.8 * np.sin(TAU * 5.6 * t + 2.0 * np.sin(TAU * 0.47 * t + 1))
              + 0.6 * np.sin(TAU * 3.3 * t + 1.2 * np.sin(TAU * 0.23 * t + 2))
              + 0.4 * np.sin(TAU * 7.2 * t + 1.6 * np.sin(TAU * 0.61 * t + 3)))
        v += pf['rAmp'] * env * s2 / 2.4
    if noise:
        v += 0.025 * np.sin(TAU * 0.27 * t) + 0.012 * np.sin(TAU * 0.09 * t + 1.3)
        v += 0.008 * (np.sin(TAU * 43.7 * t) + np.sin(TAU * 57.3 * t + 2.1) + np.sin(TAU * 29.1 * t + 0.7))
    return v


def simulate(dx, age, h, w, dur=6.0, fs=250, rng=None, noise=True, jitter=False, t_start=None):
    """Return (signal_mV_float32 at `fs` Hz, profile, patient_model)."""
    rng = rng or np.random.default_rng()
    P = patient_model(age, h, w)
    pf = make_profile(dx, P)
    if jitter:
        apply_jitter(pf, dx, rng)
    t0 = rng.uniform(20, 2000) if t_start is None else t_start
    tb = t0 - 3.0
    beats = _schedule(pf, tb, t0 + dur + 1.0, rng)
    t = t0 + np.arange(int(round(dur * fs))) / fs
    v = np.zeros_like(t)
    for (tk, qt, k, k2) in beats:
        dt = t - tk
        m = (dt > -0.6) & (dt < 0.6)
        if m.any():
            v[m] += _beat(dt[m], qt, k, k2, pf)
    v += _background(t, pf, noise, tb)
    return v.astype(np.float32), pf, P


def decimate2(x):
    """250 Hz -> 125 Hz by averaging sample pairs (same as the firmware)."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x) // 2 * 2
    return ((x[0:n:2] + x[1:n:2]) * np.float32(0.5)).astype(np.float32)


# ------------------------------------------------------------------ features
FS, WIN, PRE, POST = 125, 750, 50, 75
TL = PRE + POST
REFRACT = 25
SPEC_F = [0.5 * k for k in range(1, 21)] + [12, 15, 20, 25, 30, 40]
NSPEC = len(SPEC_F)
NFEAT = 8 + TL + NSPEC + 3 + 3          # = 165


def extract_features(x, age, h, w):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    assert n == WIN
    x64 = x.astype(np.float64)
    mu = float(x64.sum() / n)
    sd = float(np.sqrt(((x64 - mu) ** 2).sum() / n))
    p2p = float(x.max() - x.min())

    d = np.zeros(n, dtype=np.float32)
    d[1:] = x[1:] - x[:-1]
    s = d.astype(np.float64) ** 2
    cs = np.concatenate([[0.0], np.cumsum(s)])
    idx = np.arange(n)
    env = cs[np.minimum(n, idx + 5)] - cs[np.maximum(0, idx - 5)]
    mx = float(env.max())
    thr = 0.30 * mx

    peaks = []
    if mx > 1e-6:
        last = -1
        for i in range(1, n - 1):
            e = env[i]
            if e > thr and e >= env[i - 1] and e > env[i + 1]:
                if last < 0 or i - last >= REFRACT:
                    peaks.append(i); last = i
                elif e > env[last]:
                    peaks[-1] = i; last = i
    # refine to the largest deflection from the mean
    r = []
    dev = np.abs(x64 - mu)
    for p in peaks:
        lo, hi = max(0, p - 8), min(n - 1, p + 8)
        best, bv = lo, dev[lo]
        for i in range(lo + 1, hi + 1):
            if dev[i] > bv:
                bv, best = dev[i], i
        r.append(best)

    f = np.zeros(NFEAT, dtype=np.float32)
    npk = len(r)
    mean_rr = cv = rmssd = maxmin = hr = 0.0
    if npk >= 2:
        rr = np.diff(np.array(r)) / float(FS)
        mean_rr = float(rr.mean())
        cv = float(np.sqrt(((rr - mean_rr) ** 2).mean()) / mean_rr)
        if len(rr) >= 2:
            rmssd = float(np.sqrt((np.diff(rr) ** 2).mean()))
        maxmin = float(rr.max() / max(rr.min(), 1e-3))
        hr = 60.0 / mean_rr
    # beat-aligned template
    tm = np.zeros(TL, dtype=np.float64)
    nb = 0
    for rp in r:
        if rp - PRE >= 0 and rp + POST <= n:
            tm += x64[rp - PRE: rp + POST]
            nb += 1
    if nb:
        tm /= nb
        tm -= tm[:5].mean()

    f[0] = min(npk, 15) / 15.0
    f[1] = min(hr, 300.0) / 200.0
    f[2] = min(hr / lerp(HR_T, age), 4.0) if hr > 0 else 0.0
    f[3] = min(mean_rr, 6.0)
    f[4] = min(cv, 1.0)
    f[5] = min(rmssd / mean_rr, 1.0) if mean_rr > 0 else 0.0
    f[6] = min(maxmin, 5.0) / 5.0
    f[7] = nb / 10.0
    f[8:8 + TL] = tm
    # spectrum (Hann window, power fractions, sqrt)
    win = 0.5 - 0.5 * np.cos(TAU * np.arange(n) / (n - 1))
    y = (x64 - mu) * win
    k = np.arange(n)
    pw = np.zeros(NSPEC)
    for j, fr in enumerate(SPEC_F):
        om = TAU * fr / FS
        re, im = (y * np.cos(om * k)).sum(), (y * np.sin(om * k)).sum()
        pw[j] = re * re + im * im
    f[8 + TL: 8 + TL + NSPEC] = np.sqrt(pw / (pw.sum() + 1e-12))
    o = 8 + TL + NSPEC
    f[o] = sd
    f[o + 1] = p2p
    f[o + 2] = float(np.abs(d).max()) * FS / 50.0
    f[o + 3] = np.log(1 + age) / np.log(101)
    f[o + 4] = h / 200.0
    f[o + 5] = w / 120.0
    return f

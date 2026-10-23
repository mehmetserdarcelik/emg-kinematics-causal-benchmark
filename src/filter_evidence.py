"""Supplementary S2: evidence from the data that the provided kinematics are zero-phase (forward-backward) filtered.

1) Spectral slope: second (20-40 Hz) and first (8-20 Hz) difference of the elbow angle, Welch + DPSS (NW = 8),
   median over trials; compared with white noise filtered once (lfilter) or forward-backward (filtfilt) with the
   stated 4th-order 3-Hz Butterworth filter.
2) Phase: peak lag of the cross-correlation between the provided velocity and the inverse-dynamics torque in SJ
   (+-300 ms); negative values mean that the velocity leads. Control: an additional causal 3-Hz pass on the velocity.
   The spectral slope of the torque (10-40 Hz, per participant) shows that the torque was not filtered at 3 Hz.
3) Leakage: share of the energy and absolute weight of the filtfilt impulse response that lies in the future.
Output: results/extra/filter_evidence.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from data import FS_KIN, ROOT, SUBJECTS, load_subject  # noqa: E402

B, A = signal.butter(4, 3, fs=FS_KIN)


def psd(x, nperseg=512):
    win = signal.windows.dpss(nperseg, 8)
    return signal.welch(x, fs=FS_KIN, window=win, nperseg=nperseg, noverlap=nperseg // 2)


def slope(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return 10 * np.polyfit(np.log10(f[m]), np.log10(p[m]), 1)[0]   # dB/dekad


def spectral(qs):
    out = {}
    for name, k, (lo, hi) in (("diff2_20_40", 2, (20, 40)), ("diff1_8_20", 1, (8, 20))):
        P = [psd(np.diff(q, k))[1] for q in qs]
        f = psd(np.diff(qs[0], k))[0]
        out[name] = slope(f, np.median(P, 0), lo, hi)
    return out


def simulated(n_trials=200, n=2940, seed=0):
    rng = np.random.default_rng(seed)
    w = [rng.standard_normal(n + 2000) for _ in range(n_trials)]
    one = [signal.lfilter(B, A, x)[1000:-1000] for x in w]
    two = [signal.filtfilt(B, A, x)[1000:-1000] for x in w]
    return spectral(one), spectral(two)


def lag_ms(x, y, max_ms=300):
    x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    L = max_ms * FS_KIN // 1000
    lags = np.arange(-L, L + 1)
    c = [np.mean(x[max(0, -l):len(x) - max(0, l)] * y[max(0, l):len(y) - max(0, -l)]) for l in lags]
    return lags[int(np.argmax(c))] * 1000 / FS_KIN   # >0: y, x'ten sonra gelir


def leak():
    imp = np.zeros(2001)
    imp[1000] = 1
    h = signal.filtfilt(B, A, imp)
    t = (np.arange(len(h)) - 1000) * 1000 / FS_KIN
    e, a = h ** 2, np.abs(h)
    res = {f"energy_after_{ms}ms": e[t > ms].sum() / e.sum() * 100 for ms in (50, 100, 150, 200)}
    res.update({f"weight_after_{ms}ms": a[t > ms].sum() / a.sum() * 100 for ms in (50, 100, 150, 200)})
    order = np.argsort(np.abs(t))
    cum = np.cumsum(a[order]) / a.sum()
    res["weight95_halfwidth_ms"] = float(np.abs(t[order][np.searchsorted(cum, 0.95)]))
    return res


def main():
    rows = []
    qs, lags, lags_extra, tslopes = [], [], [], []
    for s in SUBJECTS:
        d = load_subject(s)
        per = []
        per_x = []
        for tr in d["trials"]["SJ"]:
            qs.append(tr.q[:, 0])
            qd, tau = tr.qd[:, 0], tr.tau[:, 0]
            per.append(lag_ms(qd, tau))                      # >0: torque after velocity -> velocity leads
            per_x.append(lag_ms(signal.lfilter(B, A, qd), tau))
        lags.append(-np.median(per))                         # velocity - torque lag (negative: velocity leads)
        lags_extra.append(-np.median(per_x))
        f = psd(d["trials"]["SJ"][0].tau[:, 0])[0]
        tslopes.append(slope(f, np.median([psd(tr.tau[:, 0])[1] for tr in d["trials"]["SJ"]], 0), 10, 40))
    data = spectral(qs)
    one, two = simulated()
    for k in data:
        rows.append(dict(measure=f"slope_{k}_dB_per_decade", data=data[k], sim_one_pass=one[k], sim_two_pass=two[k]))
    rows.append(dict(measure="velocity_minus_torque_lag_ms_median", data=float(np.median(lags)),
                     sim_one_pass=np.nan, sim_two_pass=np.nan))
    rows.append(dict(measure="velocity_minus_torque_lag_ms_range", data=f"{min(lags):.0f} to {max(lags):.0f}",
                     sim_one_pass=np.nan, sim_two_pass=np.nan))
    rows.append(dict(measure="lag_after_extra_causal_pass_ms_median", data=float(np.median(lags_extra)),
                     sim_one_pass=np.nan, sim_two_pass=np.nan))
    rows.append(dict(measure="torque_slope_10_40_dB_per_decade_range", data=f"{min(tslopes):.0f} to {max(tslopes):.0f}",
                     sim_one_pass=np.nan, sim_two_pass=np.nan))
    for k, v in leak().items():
        rows.append(dict(measure=k, data=v, sim_one_pass=np.nan, sim_two_pass=np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "extra" / "filter_evidence.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

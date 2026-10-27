"""Break-even delay d*: delay at which the median R² of the kinematics-only ridge falls to that of the EMG-only model.

qd(t + 100 ms); kinematics: k_checks_sweep 'kin' (d = 0-160 ms). EMG-only: TCN (wsweep_all) and ridge
(k_checks_sweep 'emg_only'), all windows. Participant-level bootstrap (10 000); d* = inf without a crossing within
160 ms. Confidence intervals INCLUDE the inf values (unconditional percentiles). Output: results/k_checks/K2_breakeven_v2.csv
Additionally, "EMG-TCN, common samples": the late-fusion kinematics-only ridge and EMG-only TCNs (W = 25, 50,
200 ms; seed 0), evaluated on the same samples (t >= d) at every delay (results/latefusion/S*.csv); used in the
main text and Fig. 4c.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
J3 = [("SJ", "elbow_flexion"), ("MJ", "shoulder_elv"), ("MJ", "elbow_flexion")]
WINDOWS = [25, 50, 75, 100, 150, 200, 250]
rng = np.random.default_rng(0)


def crossing(delays, kin_med, emg_med):
    """emg_med: scalar or one value per delay; zero crossing of kin - emg (linear interpolation)."""
    g = np.asarray(kin_med, float) - np.broadcast_to(emg_med, np.shape(kin_med))
    for i in range(1, len(delays)):
        if g[i] <= 0:
            return delays[i - 1] + g[i - 1] / (g[i - 1] - g[i]) * (delays[i] - delays[i - 1])
    return np.inf


def pct(x, q):
    v = np.percentile(np.where(np.isfinite(x), x, 1e9), q)
    return np.inf if v > 1e8 else v


def main(n_boot=10000):
    sw = pd.concat([pd.read_csv(p) for p in glob.glob(str(R / "k_checks_sweep" / "S*.csv"))])
    ws = pd.read_csv(R / "extra" / "wsweep_all.csv")
    rows = []
    for c, j in J3:
        kin = sw[(sw.kind == "kin") & (sw.condition == c) & (sw.joint == j)].pivot_table(
            index="subject", columns="delay_ms", values="r2")
        delays = kin.columns.values.astype(float)
        for W in WINDOWS:
            emg = {"EMG-TCN": ws[(ws.window_ms == W) & (ws.h == 100) & (ws.target == "qd")
                                 & (ws.condition == c) & (ws.joint == j)].set_index("subject").r2,
                   "EMG-ridge": sw[(sw.kind == "emg_only") & (sw.W == W) & (sw.condition == c)
                                   & (sw.joint == j)].set_index("subject").r2}
            for name, e in emg.items():
                subj = kin.index.intersection(e.index)
                K, E = kin.loc[subj].values, e.loc[subj].values
                d = crossing(delays, np.median(K, 0), np.median(E))
                bs = np.empty(n_boot)
                for b in range(n_boot):
                    i = rng.integers(0, len(subj), len(subj))
                    bs[b] = crossing(delays, np.median(K[i], 0), np.median(E[i]))
                rows.append(dict(condition=c, joint=j, W=W, emg_model=name, n=len(subj), emg_r2=np.median(E),
                                 d_star=d, ci_lo=pct(bs, 2.5), ci_hi=pct(bs, 97.5),
                                 frac_beyond_160=np.mean(~np.isfinite(bs))))
    # common samples: kinematics-only and EMG-only scores of the late-fusion files on the same samples t >= d
    lf = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(R / "latefusion" / "S[0-9]*.csv")))])
    for c, j in J3:
        x = lf[(lf.condition == c) & (lf.joint == j)]
        kin = x[x.arm == "kin"].pivot_table(index="subject", columns="delay_ms", values="r2")
        delays = kin.columns.values.astype(float)
        for W in (25, 50, 200):
            emg = x[x.arm == f"emg_w{W}"].pivot_table(index="subject", columns="delay_ms", values="r2")[kin.columns]
            K, E = kin.values, emg.loc[kin.index].values
            d = crossing(delays, np.median(K, 0), np.median(E, 0))
            bs = np.empty(n_boot)
            for b in range(n_boot):
                i = rng.integers(0, len(K), len(K))
                bs[b] = crossing(delays, np.median(K[i], 0), np.median(E[i], 0))
            rows.append(dict(condition=c, joint=j, W=W, emg_model="EMG-TCN, common samples", n=len(K),
                             emg_r2=np.median(E[:, list(delays).index(150)]), d_star=d, ci_lo=pct(bs, 2.5),
                             ci_hi=pct(bs, 97.5), frac_beyond_160=np.mean(~np.isfinite(bs))))
    out = pd.DataFrame(rows)
    out.to_csv(R / "k_checks" / "K2_breakeven_v2.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(main().round(3).to_string())

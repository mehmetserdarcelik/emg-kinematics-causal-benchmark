"""Anchoring the nominal delay axis: one additional causal 3-Hz pass on the provided kinematics (Section 2.7).

The inputs [q, qd] of the kinematics-only ridge are filtered once more, forward only, with the stated 4th-order 3-Hz
Butterworth filter (state initialised from the first sample); the target qd(t + 100 ms) is unchanged. This adds the
delay of a causal filter to the zero-phase signal (it does not remove its future information). The kinematics-only
ridge and late fusion with the W = 25 ms EMG-only TCN are computed for every delay d.
Output: results/latefusion/causal_kin.csv, causal_kin_summary.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from data import FS_KIN, JOINTS, ROOT, SUBJECTS  # noqa: E402

OUT = ROOT / "results" / "latefusion"
K = 10
DELAYS = [0, 20, 40, 50, 60, 80, 100, 120, 140, 150, 160]
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
B, A = signal.butter(4, 3, fs=FS_KIN)


def causal(x):
    zi = signal.lfilter_zi(B, A)[:, None] * x[0][None, :]
    return signal.lfilter(B, A, x, axis=0, zi=zi)[0]


def r2(y, yh):
    return 1 - np.sum((yh - y) ** 2) / np.sum((y - y.mean()) ** 2)


def run_subject(subject):
    from k_checks import A_KIN, LAGS, Ridge, _stack
    from run_v2 import FOLDS, add_history, features
    P = np.load(OUT / f"preds_{subject}.npz")
    F = features(subject)
    rows = []
    for cond in ("SJ", "MJ"):
        J = JOINTS[cond]
        nJ = len(J)
        qd = {i: F[f"{cond}_{i}_qd"] for i in range(10)}
        kc = {i: causal(np.hstack([F[f"{cond}_{i}_q"], F[f"{cond}_{i}_qd"]])) for i in range(10)}
        kinH = {i: add_history(kc[i], LAGS) for i in range(10)}
        for d_ms in DELAYS:
            d = d_ms // 10
            idx = {i: np.arange(d, len(qd[i]) - K) for i in range(10)}
            Xk = {i: kinH[i][idx[i] - d] for i in range(10)}
            ref = {i: kc[i][idx[i] - d, nJ:2 * nJ] for i in range(10)}
            y = {i: qd[i][idx[i] + K] for i in range(10)}
            inc = {i: y[i] - ref[i] for i in range(10)}
            out = {"kin_causal": {}, "late_causal_w25": {}}
            for f, (test, val) in enumerate(FOLDS):
                tr = [i for i in range(10) if i not in test and i != val]
                kin = Ridge().fit(_stack(Xk, tr), _stack(inc, tr), Xk[val], inc[val], A_KIN)
                kp = {i: ref[i] + kin.predict(Xk[i]) for i in (val, *test)}
                ep = {i: P[f"{cond}|w25|f{f}|t{i}"][d:, nJ:2 * nJ] for i in (val, *test)}
                a = np.array([GRID[int(np.argmin([np.mean((y[val][:, j] - ((1 - g) * kp[val][:, j] + g * ep[val][:, j])) ** 2)
                                                   for g in GRID]))] for j in range(nJ)])
                for i in test:
                    out["kin_causal"][i] = (y[i], kp[i])
                    out["late_causal_w25"][i] = (y[i], (1 - a) * kp[i] + a * ep[i])
            for arm, D in out.items():
                Y = np.vstack([D[i][0] for i in range(10)])
                Yh = np.vstack([D[i][1] for i in range(10)])
                for j, joint in enumerate(J):
                    rows.append(dict(subject=subject, condition=cond, joint=joint, delay_ms=d_ms, arm=arm,
                                     r2=r2(Y[:, j], Yh[:, j])))
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:   # summary only: from the saved per-participant results (no data needed)
        df = pd.read_csv(OUT / "causal_kin.csv")
    else:
        with ProcessPoolExecutor(max_workers=3) as ex:
            df = pd.DataFrame([r for rs in ex.map(run_subject, SUBJECTS) for r in rs])
        df.to_csv(OUT / "causal_kin.csv", index=False)
    piv = df.pivot_table(index=["condition", "joint", "subject", "delay_ms"], columns="arm", values="r2")
    s = pd.DataFrame({"kin_causal": piv["kin_causal"].groupby(level=[0, 1, 3]).median(),
                      "late_causal_w25": piv["late_causal_w25"].groupby(level=[0, 1, 3]).median(),
                      "gain": (piv["late_causal_w25"] - piv["kin_causal"]).groupby(level=[0, 1, 3]).median(),
                      "wins": (piv["late_causal_w25"] - piv["kin_causal"]).gt(0).groupby(level=[0, 1, 3]).sum()})
    s.reset_index().to_csv(OUT / "causal_kin_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string())

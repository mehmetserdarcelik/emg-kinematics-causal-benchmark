"""Latency diagnostic: does the advantage of short windows come from their lower latency? (Section 2.8)

EMG-only ridge (0-500 ms history), q and qd, D in {0, 100} ms. Compared: W = 25, W = 200 and W = 200
advanced by 50/90/120 ms (non-causal, for diagnosis only). Output: results/extra/check_lag_ridge.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import JOINTS, SUBJECTS  # noqa: E402

VARIANTS = {"W25": (25, 0), "W200": (200, 0), "W200+50": (200, 5), "W200+90": (200, 9),
            "W200+120": (200, 12)}


def one(s):
    from featw import emg_feats
    from k_checks import A_EMG, Ridge
    from run_v2 import FOLDS, add_history, features
    F = features(s)
    LAGS = list(range(0, 51, 5))
    E = {W: emg_feats(s, W) for W in (25, 200)}
    rows = []
    for cond in ("SJ", "MJ"):
        J = JOINTS[cond]
        for h in (0, 100):
            k = h // 10
            P = {v: {} for v in VARIANTS}
            for test, val in FOLDS:
                tr = [i for i in range(10) if i not in test and i != val]
                for v, (W, adv) in VARIANTS.items():
                    St = np.vstack([E[W][f"{cond}_{i}_emg"] for i in tr])
                    mu, sd = np.nanmean(St, 0), np.nanstd(St, 0) + 1e-8
                    X, Y = {}, {}
                    for i in range(10):
                        z = np.nan_to_num((E[W][f"{cond}_{i}_emg"] - mu) / sd)
                        if adv:
                            z = np.vstack([z[adv:], np.repeat(z[-1:], adv, 0)])
                        n = len(F[f"{cond}_{i}_q"])
                        X[i] = add_history(z, LAGS)[:n - k]
                        Y[i] = np.hstack([F[f"{cond}_{i}_q"], F[f"{cond}_{i}_qd"]])[k:]
                    m = Ridge().fit(np.vstack([X[i] for i in tr]), np.vstack([Y[i] for i in tr]),
                                    X[val], Y[val], A_EMG)
                    for i in test:
                        P[v][i] = (Y[i], m.predict(X[i]))
            for v in VARIANTS:
                Yy = np.vstack([P[v][i][0] for i in range(10)])
                Yh = np.vstack([P[v][i][1] for i in range(10)])
                for ti, t in enumerate(("q", "qd")):
                    for j, jn in enumerate(J):
                        c = ti * len(J) + j
                        e = Yh[:, c] - Yy[:, c]
                        rows.append(dict(subject=s, condition=cond, joint=jn, target=t, h=h, variant=v,
                                         r2=1 - np.sum(e ** 2) / np.sum((Yy[:, c] - Yy[:, c].mean()) ** 2)))
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as ex:
        rows = [r for rs in ex.map(one, SUBJECTS) for r in rs]
    a = pd.DataFrame(rows)
    a.to_csv(Path(__file__).resolve().parents[1] / "results" / "extra" / "check_lag_ridge.csv", index=False)
    pd.set_option("display.width", 200)
    print(a.pivot_table(index=["h", "condition", "joint", "target"], columns="variant", values="r2",
                        aggfunc="median")[list(VARIANTS)].round(3).to_string())

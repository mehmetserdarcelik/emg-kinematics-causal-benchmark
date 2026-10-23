"""Late fusion (CPU): kinematics-only ridge + EMG-only TCN, qd(t + 100 ms) (Section 2.7).

For each (participant, condition, delay d, fold) the kinematic ridge (as in k_checks: [q, qd](t - d), 0.5 s of
history, increment target) predicts the validation and test trials. EMG-TCN predictions come from run_latefusion.py.
Fusion: y = (1 - a) * y_kin + a * y_EMG, a in {0, 0.05, ..., 1} selected per joint on the validation trial
(a = 0 equals the kinematic model). Arms: W = 25/50/200 ms and the negative control shift5_w200.
Output: results/latefusion/SXX.csv, summary.csv, tests.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from data import JOINTS, ROOT, SUBJECTS  # noqa: E402

OUT = ROOT / "results" / "latefusion"
K = 10
DELAYS = [0, 20, 40, 50, 60, 80, 100, 120, 140, 150, 160]
ARMS = ["w25", "w50", "w200", "shift5_w200"]
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def r2(y, yh):
    return 1 - np.sum((yh - y) ** 2) / np.sum((y - y.mean()) ** 2)


def run_subject(subject):
    from k_checks import A_KIN, LAGS, Ridge, _stack
    from run_v2 import FOLDS, add_history, features
    pfile = OUT / f"preds_{subject}.npz"
    if not pfile.exists():
        return []
    P = np.load(pfile)
    F = features(subject)
    rows = []
    for cond in ("SJ", "MJ"):
        J = JOINTS[cond]
        nJ = len(J)
        q = {i: F[f"{cond}_{i}_q"] for i in range(10)}
        qd = {i: F[f"{cond}_{i}_qd"] for i in range(10)}
        kinH = {i: add_history(np.hstack([q[i], qd[i]]), LAGS) for i in range(10)}
        for d_ms in DELAYS:
            d = d_ms // 10
            idx = {i: np.arange(d, len(qd[i]) - K) for i in range(10)}
            Xk = {i: kinH[i][idx[i] - d] for i in range(10)}
            ref = {i: qd[i][idx[i] - d] for i in range(10)}
            y = {i: qd[i][idx[i] + K] for i in range(10)}
            inc = {i: y[i] - ref[i] for i in range(10)}
            test_out = {}   # arm -> {trial: (y, yh)}, weights
            weights = {a: [] for a in ARMS}
            for f, (test, val) in enumerate(FOLDS):
                tr = [i for i in range(10) if i not in test and i != val]
                kin = Ridge().fit(_stack(Xk, tr), _stack(inc, tr), Xk[val], inc[val], A_KIN)
                kp = {i: ref[i] + kin.predict(Xk[i]) for i in (val, *test)}
                for i in test:
                    test_out.setdefault("kin", {})[i] = (y[i], kp[i])
                for arm in ARMS:
                    ep = {i: P[f"{cond}|{arm}|f{f}|t{i}"][d:, nJ:2 * nJ] for i in (val, *test)}
                    a = np.array([GRID[int(np.argmin([np.mean((y[val][:, j] - ((1 - g) * kp[val][:, j]
                                                                                + g * ep[val][:, j])) ** 2)
                                                       for g in GRID]))] for j in range(nJ)])
                    weights[arm].append(a)
                    for i in test:
                        test_out.setdefault(f"emg_{arm}", {})[i] = (y[i], ep[i])
                        test_out.setdefault(f"late_{arm}", {})[i] = (y[i], (1 - a) * kp[i] + a * ep[i])
            for arm_name, D in test_out.items():
                Y = np.vstack([D[i][0] for i in range(10)])
                Yh = np.vstack([D[i][1] for i in range(10)])
                for j, joint in enumerate(J):
                    e = Yh[:, j] - Y[:, j]
                    w = np.mean([a[j] for a in weights[arm_name.split("_", 1)[1]]]) if arm_name.startswith("late") else np.nan
                    rows.append(dict(subject=subject, condition=cond, joint=joint, delay_ms=d_ms, arm=arm_name,
                                     r2=r2(Y[:, j], Yh[:, j]), rmse_degs=float(np.degrees(np.sqrt(np.mean(e ** 2)))),
                                     weight=w))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"{subject}.csv", index=False)
    return rows


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1, run)
    return adj


def summarise(df):
    rng = np.random.default_rng(0)
    piv = df.pivot_table(index=["condition", "joint", "subject", "delay_ms"], columns="arm", values="r2")
    summ = df.groupby(["condition", "joint", "delay_ms", "arm"]).agg(
        r2=("r2", "median"), rmse=("rmse_degs", "median"), weight=("weight", "median")).reset_index()
    tests = []
    for W in ("w25", "w50", "w200"):
        fam = []
        for (c, j), s in piv.groupby(level=[0, 1]):
            for d in (0, 50, 100, 150):
                x = s.xs(d, level="delay_ms")
                for comp, a, b in (("late_vs_kin", f"late_{W}", "kin"),
                                   ("late_vs_shift", f"late_{W}", "late_shift5_w200")):
                    dd = (x[a] - x[b]).dropna()
                    bs = np.median(rng.choice(dd.values, (10000, len(dd))), axis=1)
                    fam.append(dict(W=W, comparison=comp, condition=c, joint=j, delay_ms=d, n=len(dd),
                                    wins=int((dd > 0).sum()), median=dd.median(), ci_lo=np.percentile(bs, 2.5),
                                    ci_hi=np.percentile(bs, 97.5),
                                    p=wilcoxon(dd).pvalue if (dd != 0).any() else 1.0))
        f = pd.DataFrame(fam)
        f["p_holm"] = f.groupby("comparison").p.transform(lambda s: holm(s.values))
        f["pos"] = (f["median"] >= 0.01) & (f.p_holm < 0.05)
        tests.append(f)
    return summ, pd.concat(tests, ignore_index=True)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as ex:
        rows = [r for rs in ex.map(run_subject, SUBJECTS) for r in rs]
    df = pd.DataFrame(rows)
    n = df.subject.nunique()
    summ, tests = summarise(df)
    summ.to_csv(OUT / "summary.csv", index=False)
    tests.to_csv(OUT / "tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{n} participants")
    print(summ[summ.delay_ms.isin([0, 50, 100, 150])].pivot_table(
        index=["condition", "joint", "delay_ms"], columns="arm", values="r2").round(3).to_string())
    print(tests[tests.comparison == "late_vs_kin"].round(4).to_string(index=False))

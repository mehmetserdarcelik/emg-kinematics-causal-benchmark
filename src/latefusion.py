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
ARMS2 = ["shift5_w25", "seed1_w25", "seed1_w200"]   # matched control and second seed (preds2_SXX.npz)
KT_ARMS = ["kintcn", "kintcn0"]
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def r2(y, yh):
    return 1 - np.sum((yh - y) ** 2) / np.sum((y - y.mean()) ** 2)


def run_subject(subject):
    from k_checks import A_KIN, LAGS, Ridge, _stack
    from run_v2 import FOLDS, add_history, features
    pfile = OUT / f"preds_{subject}.npz"
    if not pfile.exists():
        return []
    P = dict(np.load(pfile))
    p2 = OUT / f"preds2_{subject}.npz"
    arms = list(ARMS)
    if p2.exists():
        P.update(dict(np.load(p2)))
        arms += ARMS2
    KT = {}   # ensemble control (kinematics-only TCN): kintcn (input noise 0.1), kintcn0 (no input noise)
    for kt in KT_ARMS:
        pk = OUT / f"{kt}_{subject}.npz"
        if pk.exists():
            KT.update(dict(np.load(pk)))
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
            weights = {a: [] for a in list(arms) + KT_ARMS}
            for f, (test, val) in enumerate(FOLDS):
                tr = [i for i in range(10) if i not in test and i != val]
                kin = Ridge().fit(_stack(Xk, tr), _stack(inc, tr), Xk[val], inc[val], A_KIN)
                kp = {i: ref[i] + kin.predict(Xk[i]) for i in (val, *test)}
                for i in test:
                    test_out.setdefault("kin", {})[i] = (y[i], kp[i])
                arm_list = list(arms) + [kt for kt in KT_ARMS if f"{cond}|{kt}_d{d_ms}|f{f}|t{val}" in KT]
                for arm in arm_list:
                    if arm in KT_ARMS:
                        ep = {i: KT[f"{cond}|{arm}_d{d_ms}|f{f}|t{i}"][:, nJ:2 * nJ] for i in (val, *test)}
                    else:
                        ep = {i: P[f"{cond}|{arm}|f{f}|t{i}"][d:, nJ:2 * nJ] for i in (val, *test)}
                    a = np.array([GRID[int(np.argmin([np.mean((y[val][:, j] - ((1 - g) * kp[val][:, j]
                                                                                + g * ep[val][:, j])) ** 2)
                                                       for g in GRID]))] for j in range(nJ)])
                    weights[arm].append(a)
                    if arm == "w25":   # oracle upper bound: weight chosen on the test trials (descriptive only)
                        yt = np.vstack([y[i] for i in test])
                        kt = np.vstack([kp[i] for i in test])
                        et = np.vstack([ep[i] for i in test])
                        ao = np.array([GRID[int(np.argmin([np.mean((yt[:, j] - ((1 - g) * kt[:, j] + g * et[:, j])) ** 2)
                                                           for g in GRID]))] for j in range(nJ)])
                        for i in test:
                            test_out.setdefault("oracle_w25", {})[i] = (y[i], (1 - ao) * kp[i] + ao * ep[i])
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


def cross(ds, v, thr=0.01):
    for i in range(1, len(ds)):
        if v[i - 1] < thr <= v[i]:
            return ds[i - 1] + (thr - v[i - 1]) * (ds[i] - ds[i - 1]) / (v[i] - v[i - 1])
    return np.inf


def crossings(df, arms=("late_w25", "late_w50", "late_w200", "late_seed1_w25", "late_seed1_w200"), n_boot=10000):
    """Nominal delay at which the median late-fusion gain reaches 0.01, with participant-level bootstrap 95 % CI."""
    rng = np.random.default_rng(0)
    piv = df.pivot_table(index=["condition", "joint", "subject", "delay_ms"], columns="arm", values="r2")
    rows = []
    for (c, j), s in piv.groupby(level=[0, 1]):
        for arm in [a for a in arms if a in s]:
            g = (s[arm] - s["kin"]).droplevel([0, 1]).unstack("delay_ms")
            ds = g.columns.values.astype(float)
            G = g.values
            est = cross(ds, np.median(G, 0))
            bs = np.array([cross(ds, np.median(G[rng.integers(0, len(G), len(G))], 0)) for _ in range(n_boot)])
            fin = np.where(np.isfinite(bs), bs, 1e9)
            rows.append(dict(condition=c, joint=j, arm=arm, d001=est, ci_lo=np.percentile(fin, 2.5),
                             ci_hi=np.percentile(fin, 97.5), frac_no_cross=np.mean(~np.isfinite(bs))))
    return pd.DataFrame(rows)


def summarise(df):
    rng = np.random.default_rng(0)
    piv = df.pivot_table(index=["condition", "joint", "subject", "delay_ms"], columns="arm", values="r2")
    summ = df.groupby(["condition", "joint", "delay_ms", "arm"]).agg(
        r2=("r2", "median"), rmse=("rmse_degs", "median"), weight=("weight", "median")).reset_index()
    # each window against its own shifted control (W = 50: nearest, W = 25); the windows form one Holm family
    ctrl = {"w25": "late_shift5_w25", "w50": "late_shift5_w25", "w200": "late_shift5_w200",
            "seed1_w25": "late_shift5_w25", "seed1_w200": "late_shift5_w200"}
    tests = []
    for W in [w for w in ("w25", "w50", "w200", "seed1_w25", "seed1_w200") if f"late_{w}" in piv]:
        fam = []
        for (c, j), s in piv.groupby(level=[0, 1]):
            for d in (0, 50, 100, 150):
                x = s.xs(d, level="delay_ms")
                for comp, a, b in (("late_vs_kin", f"late_{W}", "kin"),
                                   ("late_vs_shift", f"late_{W}", ctrl[W] if ctrl[W] in x else "late_shift5_w200")):
                    dd = (x[a] - x[b]).dropna()
                    bs = np.median(rng.choice(dd.values, (10000, len(dd))), axis=1)
                    fam.append(dict(W=W, comparison=comp, condition=c, joint=j, delay_ms=d, n=len(dd),
                                    wins=int((dd > 0).sum()), median=dd.median(), ci_lo=np.percentile(bs, 2.5),
                                    ci_hi=np.percentile(bs, 97.5),
                                    p=wilcoxon(dd).pvalue if (dd != 0).any() else 1.0))
        tests.append(pd.DataFrame(fam))
    t = pd.concat(tests, ignore_index=True)
    t["family"] = np.where(t.W.str.startswith("seed1"), "seed1", "seed0")
    t["p_holm"] = t.groupby(["family", "comparison"]).p.transform(lambda s: holm(s.values))
    t["pos"] = (t["median"] >= 0.01) & (t.p_holm < 0.05)
    return summ, t


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:   # summary only: from the per-participant S*.csv files (no data needed)
        df = pd.concat([pd.read_csv(OUT / f"{s}.csv") for s in SUBJECTS], ignore_index=True)
    else:
        with ProcessPoolExecutor(max_workers=2) as ex:
            rows = [r for rs in ex.map(run_subject, SUBJECTS) for r in rs]
        df = pd.DataFrame(rows)
    n = df.subject.nunique()
    summ, tests = summarise(df)
    summ.to_csv(OUT / "summary.csv", index=False)
    crossings(df).to_csv(OUT / "crossings.csv", index=False)
    tests.to_csv(OUT / "tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{n} participants")
    print(summ[summ.delay_ms.isin([0, 50, 100, 150])].pivot_table(
        index=["condition", "joint", "delay_ms"], columns="arm", values="r2").round(3).to_string())
    print(tests[tests.comparison == "late_vs_kin"].round(4).to_string(index=False))

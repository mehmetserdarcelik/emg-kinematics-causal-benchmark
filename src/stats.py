"""Statistics: participant-level bootstrap 95 % CI (median R²), paired Wilcoxon tests,
Holm correction within each comparison family (Section 2.9). Output: results/extra/stats_*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "extra"
TGT = [("SJ", "elbow_flexion", "q"), ("SJ", "elbow_flexion", "qd"), ("MJ", "shoulder_elv", "q"),
       ("MJ", "shoulder_elv", "qd"), ("MJ", "elbow_flexion", "q"), ("MJ", "elbow_flexion", "qd")]
rng = np.random.default_rng(0)


def boot_ci(x, n=10000):
    x = np.asarray(x)
    b = np.median(rng.choice(x, (n, len(x))), axis=1)
    return np.median(x), np.percentile(b, 2.5), np.percentile(b, 97.5)


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired(df, a, b, family, label):
    rows = []
    for c, j, t in TGT:
        s = df[(df.condition == c) & (df.joint == j) & (df.target == t)]
        w = s.pivot_table(index="subject", columns="model", values="r2")
        if a not in w or b not in w:
            continue
        d = (w[a] - w[b]).dropna()
        m, lo, hi = boot_ci(d)
        rows.append(dict(family=family, comparison=label, condition=c, joint=j, target=t,
                         n=len(d), wins=int((d > 0).sum()), median_diff=m, ci_lo=lo, ci_hi=hi,
                         p=wilcoxon(d).pvalue))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p.values)
    return out


if __name__ == "__main__":
    v = pd.read_csv(R / "v2" / "summary_by_subject.csv")  # averaged over seeds
    v = v[v.target != "tau"]
    # 1) median R² and 95 % CI
    rows = []
    for (c, j, t, m), s in v.groupby(["condition", "joint", "target", "model"]):
        med, lo, hi = boot_ci(s.r2)
        rows.append(dict(condition=c, joint=j, target=t, model=m, median_r2=med, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).to_csv(OUT / "stats_within_ci.csv", index=False)
    # 2) paired comparisons (each a family of 6 targets, Holm)
    fams = [paired(v, "tcn", "ridge25", "A", "TCN − Ridge"),
            paired(v, "tcn", "lstm", "B", "TCN − LSTM"),
            paired(v, "tcn", "tau_tcn", "C", "EMG-TCN − torque-TCN"),
            paired(v, "emgtau_tcn", "tau_tcn", "D", "(EMG + torque) − torque"),
            paired(v, "tcn_aux", "tcn", "E", "torque auxiliary output − none")]
    # 3) LOSO: effect of calibration (averaged over seeds, including additional seeds if present)
    l = pd.concat([pd.read_csv(p) for p in sorted((R / "v2_loso").glob("metrics_S*.csv"))])
    ex = sorted((R / "extra").glob("loso_seeds_S*.csv"))
    if ex:
        l = pd.concat([l, pd.concat([pd.read_csv(p) for p in ex])])
    lt = l[l.model == "tcn"].groupby(["subject", "condition", "joint", "target", "variant"],
                                     as_index=False).r2.mean()
    lt = lt.rename(columns={"variant": "model"})
    fams.append(paired(lt, "calib30s", "none", "F", "LOSO: 30-s calibration − uncalibrated"))
    st = pd.concat(fams)
    st.to_csv(OUT / "stats_paired.csv", index=False)
    pd.set_option("display.width", 220)
    print(st.round(4).to_string(index=False))

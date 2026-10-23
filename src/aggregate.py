"""Builds the intermediate summary tables from the per-participant result files.

Outputs (under OUT):
  horizon/all_with_h0.csv          horizon analysis + D = 0 (run_v2, seed 0)
  extra/wsweep_all.csv             TCN window sweep (W = 200: run_v2 / run_horizon, seed 0)
  extra/wsweep_tests.csv           W = 25/50 vs W = 200, D = 0/100; Holm family of 6 tests per (D, W)
  extra/delay_all.csv              TCN early fusion, d = 0/50/100/150 (hbase2-4)
  extra/stats_delay.csv            early fusion: EMG+kin - kin, Holm family of 12 tests per delay
  k_checks/K3B_tcn.csv             early fusion, qd D = 100, W = 50/100/200; Holm per window
  k_checks_sweep/sweep_summary.csv summary of the ridge window sweep (including d0.01)
  k_checks/K3A_slope_threshold001.csv  d0.01 and slope b (bootstrap 2000)
Usage: python src/aggregate.py [OUT]   (default OUT: results)
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from stats import holm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
J3 = [("SJ", "elbow_flexion"), ("MJ", "shoulder_elv"), ("MJ", "elbow_flexion")]
KEY = ["subject", "condition", "joint", "target"]


def cat(pattern):
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def all_with_h0():
    h = cat("horizon/metrics_S*.csv")
    v = cat("v2/metrics_S*.csv")
    v0 = v[v.model.isin(["tcn", "tau_tcn"]) & (v.seed == 0)].assign(horizon_ms=0)
    return pd.concat([v0, h], ignore_index=True)[["subject", "condition", "joint", "target", "model", "seed", "r",
                                                  "r2", "nrmse", "rmse_deg", "bias_deg", "horizon_ms"]]


def wsweep_all():
    w = cat("extra/wsweep_S*.csv")
    k = cat("extra/k3b_S*.csv")
    k = k[k.model.str.startswith("emg_only")]
    v = cat("v2/metrics_S*.csv")
    v = v[(v.model == "tcn") & (v.seed == 0)].assign(model="tcn_w200", horizon_ms=0, window_ms=200)
    h = cat("horizon/metrics_S*.csv")
    h = h[(h.model == "tcn") & (h.horizon_ms == 100)].assign(model="emg_only_w200", window_ms=200)
    a = pd.concat([w, k, v, h], ignore_index=True)
    a["h"] = a.horizon_ms
    return a


def wsweep_tests(w):
    rows = []
    for h in (0, 100):
        for W in (25, 50):
            fam = []
            for (c, j, t), s in w[w.h == h].groupby(["condition", "joint", "target"]):
                A = s[s.window_ms == W].set_index("subject").r2
                B = s[s.window_ms == 200].set_index("subject").r2
                dd, p = paired(A, B)
                fam.append(dict(h=h, W=W, condition=c, joint=j, target=t, median=dd.median(),
                                wins=int((dd > 0).sum()), p=p))
            f = pd.DataFrame(fam)
            f["p_holm"] = holm(f.p.values)
            rows.append(f)
    return pd.concat(rows, ignore_index=True)


def delay_all():
    return pd.concat([cat(f"extra/{m}_S*.csv") for m in ("hbase2", "hbase3", "hbase4")], ignore_index=True)


def paired(A, B):
    d = (A - B).dropna()
    return d, (wilcoxon(d).pvalue if (d != 0).any() else 1.0)


def stats_delay(da):
    rows = []
    for d in (0, 50, 100, 150):
        fam = []
        for (c, j, t, h), s in da.groupby(["condition", "joint", "target", "horizon_ms"]):
            A = s[s.model == f"emgkin_inc_d{d}"].set_index("subject").r2
            B = s[s.model == f"kin_inc_d{d}"].set_index("subject").r2
            dd, p = paired(A, B)
            fam.append(dict(condition=c, joint=j, target=t, horizon_ms=h, delay_ms=d, median_diff=dd.median(),
                            wins=int((dd > 0).sum()), p=p, kin=B.median(), emgkin=A.median()))
        f = pd.DataFrame(fam)
        f["p_holm"] = holm(f.p.values)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def k3b(da):
    k = cat("extra/k3b_S*.csv")
    rows = []
    for W in (50, 100, 200):
        fam = []
        for d in (0, 50, 100, 150):
            for c, j in J3:
                sel = lambda x: x[(x.condition == c) & (x.joint == j) & (x.target == "qd") & (x.horizon_ms == 100)]  # noqa: E731
                B = sel(da)[sel(da).model == f"kin_inc_d{d}"].set_index("subject").r2
                src, name = (sel(da), f"emgkin_inc_d{d}") if W == 200 else (sel(k), f"emgkin_inc_d{d}_w{W}")
                A = src[src.model == name].set_index("subject").r2
                dd, p = paired(A, B)
                fam.append(dict(W=W, d=d, condition=c, joint=j, median=dd.median(), wins=int((dd > 0).sum()), p=p))
        f = pd.DataFrame(fam)
        f["p_holm"] = holm(f.p.values)
        f["pos"] = (f["median"] >= 0.01) & (f.p_holm < 0.05)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def cross(ds, vals, thr=0.01):
    """First upward crossing of thr by the median dR2(d) curve (linear interpolation)."""
    ds, vals = np.asarray(ds, float), np.asarray(vals, float)
    if vals[0] >= thr:
        return -np.inf
    for i in range(1, len(ds)):
        if vals[i - 1] < thr <= vals[i]:
            return ds[i - 1] + (thr - vals[i - 1]) * (ds[i] - ds[i - 1]) / (vals[i] - vals[i - 1])
    return np.inf


def ridge_sweep():
    sw = cat("k_checks_sweep/S*.csv")
    rng = np.random.default_rng(0)
    summ, slope = [], []
    for c, j in J3:
        s = sw[(sw.condition == c) & (sw.joint == j)]
        kin = s[s.kind == "kin"].pivot_table(index="subject", columns="delay_ms", values="r2")
        curves = {}
        for W in sorted(s[s.kind == "fused"].W.unique()):
            fu = s[(s.kind == "fused") & (s.W == W)].pivot_table(index="subject", columns="delay_ms", values="r2")
            curves[W] = fu - kin
            med = curves[W].median()
            summ.append(dict(condition=c, joint=j, W=W,
                             emg_only=s[(s.kind == "emg_only") & (s.W == W)].r2.median(),
                             dR2_d50=med[50], dR2_d100=med[100], dR2_d150=med[150],
                             d001=cross(med.index, med.values), win150=int((curves[W][150] > 0).sum())))
        d0 = {W: cross(curves[W].columns, curves[W].median().values) for W in (50, 100, 200)}
        subj = curves[200].index.values
        bs = []
        for _ in range(2000):
            idx = rng.choice(subj, len(subj))
            z = {W: cross(curves[W].columns, curves[W].loc[idx].median().values) for W in (50, 200)}
            if np.isfinite(z[200]) and np.isfinite(z[50]):
                bs.append((z[200] - z[50]) / 75)
        slope.append(dict(condition=c, joint=j, d01_W50=d0[50], d01_W100=d0[100], d01_W200=d0[200],
                          b=(d0[200] - d0[50]) / 75, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    return pd.DataFrame(summ), pd.DataFrame(slope)


def main(out=R):
    out = Path(out)
    for sub in ("horizon", "extra", "k_checks", "k_checks_sweep"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    all_with_h0().to_csv(out / "horizon" / "all_with_h0.csv", index=False)
    w = wsweep_all()
    w.to_csv(out / "extra" / "wsweep_all.csv", index=False)
    wsweep_tests(w).to_csv(out / "extra" / "wsweep_tests.csv", index=False)
    da = delay_all()
    da.to_csv(out / "extra" / "delay_all.csv", index=False)
    stats_delay(da).to_csv(out / "extra" / "stats_delay.csv", index=False)
    k3b(da).to_csv(out / "k_checks" / "K3B_tcn.csv", index=False)
    summ, slope = ridge_sweep()
    summ.to_csv(out / "k_checks_sweep" / "sweep_summary.csv", index=False)
    slope.to_csv(out / "k_checks" / "K3A_slope_threshold001.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else R)

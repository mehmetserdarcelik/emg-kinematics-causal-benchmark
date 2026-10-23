"""Application of the K3/K4 decision rules (Supplementary S3).

This script was written BEFORE the K3-A/K4 results were seen (2026-10-14). Two details
left open by the plan were fixed here, before the results:
  (1) Per-target decisions are turned into an overall decision with an "at least 2 of
      3 targets" criterion (consistent with the 2/3 rule of K3).
  (2) The validation-gate condition "dR2 <= 0 at d = 0" is interpreted as "NOT
      significantly positive at d = 0 (median dR2 < 0.01 or Holm p >= 0.05)". Because
      residual fusion can select c = 0 on validation, ~0 is expected at d = 0.
"Significantly positive": median paired dR2 >= 0.01 AND Holm p < 0.05.
Primary targets: qd, D = 100 ms; SJ elbow, MJ shoulder, MJ elbow.
Outputs: results/k_checks/K4_tests.csv, K3A_slope.csv (zero-crossing version; see
src/aggregate.py for the 0.01-threshold version used in the paper).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
TG = [("SJ", "elbow_flexion"), ("MJ", "shoulder_elv"), ("MJ", "elbow_flexion")]
rng = np.random.default_rng(0)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1, run)
    return adj


def boot(x, n=5000):
    x = np.asarray(x)
    b = np.median(rng.choice(x, (n, len(x))), axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def paired(df, a, b, family):
    """a, b: (kind, W, d) triplets; df: k_checks rows (qd only)."""
    rows = []
    for c, j in TG:
        s = df[(df.condition == c) & (df.joint == j)]
        A = s[(s.kind == a[0]) & (s.W == a[1]) & (s.delay_ms == a[2])].set_index("subject").r2
        B = s[(s.kind == b[0]) & (s.W == b[1]) & (s.delay_ms == b[2])].set_index("subject").r2
        dlt = (A - B).dropna()
        lo, hi = boot(dlt)
        rows.append(dict(family=family, a=f"{a[0]}_W{a[1]}_d{a[2]}", b=f"{b[0]}_W{b[1]}_d{b[2]}",
                         condition=c, joint=j, n=len(dlt), wins=int((dlt > 0).sum()),
                         median=dlt.median(), ci_lo=lo, ci_hi=hi,
                         p=wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0))
    return rows


def finalize(rows):
    df = pd.DataFrame(rows)
    df["p_holm"] = df.groupby("family").p.transform(lambda s: holm(s.values))
    df["pos"] = (df["median"] >= 0.01) & (df.p_holm < 0.05)
    return df


def zero_cross(ds, vals):
    """First negative-to-positive crossing of the median dR2(d) curve (linear interpolation)."""
    ds, vals = np.asarray(ds, float), np.asarray(vals, float)
    if vals[0] > 0:
        return -np.inf
    for i in range(1, len(ds)):
        if vals[i - 1] <= 0 < vals[i]:
            return ds[i - 1] + (0 - vals[i - 1]) * (ds[i] - ds[i - 1]) / (vals[i] - vals[i - 1])
    return np.inf


def k3a_slope(df):
    out = []
    for c, j in TG:
        s = df[(df.condition == c) & (df.joint == j)]
        kin = s[s.kind == "kin"].pivot_table(index="subject", columns="delay_ms", values="r2")
        d0 = {}
        curves = {}
        for W in (50, 100, 200):
            fu = s[(s.kind == "fused") & (s.W == W)].pivot_table(index="subject", columns="delay_ms",
                                                                  values="r2")
            curves[W] = fu - kin
            d0[W] = zero_cross(curves[W].columns, curves[W].median().values)
        b = (d0[200] - d0[50]) / 75 if np.isfinite(d0[200]) and np.isfinite(d0[50]) else np.nan
        bs = []
        subj = curves[200].index.values
        for _ in range(2000):
            idx = rng.choice(subj, len(subj))
            z = {W: zero_cross(curves[W].columns, curves[W].loc[idx].median().values)
                 for W in (50, 200)}
            if np.isfinite(z[200]) and np.isfinite(z[50]):
                bs.append((z[200] - z[50]) / 75)
        out.append(dict(condition=c, joint=j, d0_W50=d0[50], d0_W100=d0[100], d0_W200=d0[200], b=b,
                        b_ci_lo=np.percentile(bs, 2.5) if bs else np.nan,
                        b_ci_hi=np.percentile(bs, 97.5) if bs else np.nan, n_boot_valid=len(bs)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(p) for p in sorted((R / "k_checks").glob("S*.csv"))])
    out = R / "k_checks"
    pd.set_option("display.width", 220)

    # ---------------- K4 ----------------
    rows = []
    for d in (0, 50, 150):
        rows += paired(df, ("fused", 200, d), ("kin", 0, d), "K4a_fused_vs_kin")
    for S in (5, 7, 9):
        rows += paired(df, ("fused", 200, 150), (f"shift{S}", 200, 150), "K4b_fused_vs_shift")
        rows += paired(df, (f"shift{S}", 200, 150), ("kin", 0, 150), "K4c_shift_vs_kin")
    k4 = finalize(rows)
    k4.to_csv(out / "K4_tests.csv", index=False)
    print("=== K4 ===")
    print(k4[["family", "a", "b", "condition", "joint", "wins", "median", "ci_lo", "ci_hi", "p_holm", "pos"]]
          .round(4).to_string(index=False))
    # decision: capacity artefact (per target) -> the delay finding is withdrawn if present in >= 2/3 targets
    retract_t = []
    for c, j in TG:
        sk = k4[(k4.family == "K4c_shift_vs_kin") & (k4.condition == c) & (k4.joint == j)]
        fs = k4[(k4.family == "K4b_fused_vs_shift") & (k4.condition == c) & (k4.joint == j)]
        cap = (sk["median"] >= 0.03).any() or (fs["median"] < 0.01).any()
        specific = fs.pos.all()
        retract_t.append(dict(condition=c, joint=j, capacity_artifact=bool(cap),
                              time_specific=bool(specific)))
    rt = pd.DataFrame(retract_t)
    print(rt.to_string(index=False))
    n_art = int(rt.capacity_artifact.sum())
    print("K4 DECISION (capacity):", "DELAY FINDING WITHDRAWN" if n_art >= 2 else
          ("TIME-SPECIFIC ADDITIONAL INFORMATION" if rt.time_specific.sum() >= 2 else "PARTIAL / INCONCLUSIVE"))
    low = k4[(k4.family == "K4a_fused_vs_kin") & k4.b.str.contains("_d0|_d50")]
    low = low[low.a.str.contains("_d0$|_d50$", regex=True)]
    print("K4 DECISION (d = 0/50 residual fusion):",
          "CONTRIBUTION PRESENT, MASKED BY OVERFITTING IN THE TCN" if low.pos.sum() >= 2 else
          "NO EMG CONTRIBUTION DETECTED AT DELAYS <= 50 ms")

    # ---------------- K3-A ----------------
    sl = k3a_slope(df)
    sl.to_csv(out / "K3A_slope.csv", index=False)
    print("\n=== K3-A (ridge) ===")
    print(sl.round(3).to_string(index=False))
    # validation gates
    g1 = finalize(paired(df, ("fused", 200, 150), ("kin", 0, 150), "g150")
                  + paired(df, ("fused", 200, 0), ("kin", 0, 0), "g0"))
    gate1 = (g1[g1.family == "g150"]["median"] > 0).sum() >= 2 and (~g1[g1.family == "g0"].pos).sum() >= 2
    eo = df[df.kind == "emg_only"].pivot_table(index=["condition", "joint", "subject"], columns="W", values="r2")
    gap = (eo[200] - eo[50]).groupby(level=[0, 1]).median()
    gate2 = (gap <= 0.05).sum() >= 2
    print("Gate 1 (ridge reproduces the TCN pattern):", gate1, "| Gate 2 (W50 EMG-only loss <= 0.05):", gate2,
          "| W200 - W50 EMG-only median differences:", gap.round(3).to_dict())
    print("Note: the overall K3 decision is taken together with the TCN (K3-B) results: b >= 0.6 and TCN W50 d50"
          " positive in >= 2/3 targets; b <= 0.25 and no positive TCN W50 d50; otherwise in between.")

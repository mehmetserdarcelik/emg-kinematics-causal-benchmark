"""Paper tables -> paper/tables.md. Table 1 data audit, Table 2 main results, Table 3 delayed feedback."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from stats import boot_ci, holm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "tables.md"
T6 = [("SJ", "elbow_flexion", "q", "SJ elbow angle"), ("SJ", "elbow_flexion", "qd", "SJ elbow velocity"),
      ("MJ", "shoulder_elv", "q", "MJ shoulder angle"), ("MJ", "shoulder_elv", "qd", "MJ shoulder velocity"),
      ("MJ", "elbow_flexion", "q", "MJ elbow angle"), ("MJ", "elbow_flexion", "qd", "MJ elbow velocity")]
J3 = [("SJ", "elbow_flexion", "SJ elbow"), ("MJ", "shoulder_elv", "MJ shoulder"), ("MJ", "elbow_flexion", "MJ elbow")]


def mci(x, nd=3):
    _, lo, hi = boot_ci(np.asarray(x))
    return f"{np.median(x):.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fp(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"{p / 10 ** e:.1f}·10{sup}"
    return f"{p:.3f}"


T1 = """**Table 1.** Issues identified in the audit of the open dataset and how they were handled.

| Issue | Evidence | Handling |
|---|---|---|
| MJ angular velocity units | Ratio to the numerical derivative of the angle 100.2–100.5 (r = 0.998); SJ ratio 1.0 | Multiplied by 100 (rad·sample⁻¹ at 100 Hz → rad·s⁻¹) |
| MVC recordings | 0.00 % of power above 20 Hz in all 34 files (trial EMG: 63–96 %); consistent with a processed envelope | MVC normalisation not used; z-scoring with training statistics |
| Wireless sensor dropouts | Runs of ≥ 20 identical raw samples in 1 491 of 3 230 trial–channel pairs (≈ 30 000 segments, mostly 10–20 ms, up to 7 s; 744 s in total); genuine EMG ≤ 2–3 identical consecutive samples | Online detection (≥ 20 identical samples), causal forward fill of features; median 17 % of samples per participant and condition (range 1–78 %) within an affected 200-ms window |
| Zero-phase kinematic filtering | Preprint statement of bidirectional filtering; −290 dB/decade roll-off (two-pass simulation −293, one-pass −132); velocity leads SJ torque by a median of 65 ms | Targets used as provided; nominal kinematic delays interpreted with caution (filtering may bias both kinematic and EMG-based predictions) |
| MJ elbow angle saturation | Plateau near 130° in some participants (≤ 4.7 % of samples) | Excluded in sensitivity analysis (median R² change ≤ 0.015) |
"""


def table2():
    v = pd.read_csv(R / "v2" / "summary_by_subject.csv")
    L = pd.concat([pd.read_csv(p) for p in glob.glob(str(R / "v2_loso" / "metrics_S*.csv"))]
                  + [pd.read_csv(p) for p in glob.glob(str(R / "extra" / "loso_seeds_S*.csv"))])
    L = L[L.model == "tcn"].groupby(["subject", "condition", "joint", "target", "variant"], as_index=False).r2.mean()
    st = pd.read_csv(R / "extra" / "stats_paired.csv")
    rows = []
    for c, j, t, lab in T6:
        sel = lambda d, m: d[(d.model == m) & (d.condition == c) & (d.joint == j) & (d.target == t)]  # noqa: E731
        tcn = sel(v, "tcn")
        rm = tcn.rmse_deg.median()
        lo = {k: L[(L.variant == k) & (L.condition == c) & (L.joint == j) & (L.target == t)].r2.values
              for k in ("none", "calib30s")}
        pf = st[(st.family == "F") & (st.condition == c) & (st.joint == j) & (st.target == t)].iloc[0]
        rows.append(f"| {lab} | {np.median(sel(v, 'ridge25').r2):.3f} | {np.median(sel(v, 'lstm').r2):.3f} | "
                    f"**{mci(tcn.r2)}** | {rm:.1f} | {mci(lo['none'])} | {mci(lo['calib30s'])} | "
                    f"{int(pf.wins)}/17, {fp(pf.p_holm)} |")
    head = ("**Table 2.** Decoding accuracy (R², median across 17 participants; 95 % bootstrap CI for the TCN and "
            "LOSO). Within-subject: rotating trial-level folds; LOSO: leave-one-subject-out, TCN averaged over three "
            "seeds. Last column: participants improved by calibration and Holm-corrected Wilcoxon p. "
            "RMSE in ° (angle) or °·s⁻¹ (velocity). W = 200 ms.\n\n"
            "| Target | Ridge | LSTM | TCN | TCN RMSE | LOSO uncalibrated | LOSO 30-s calibration | Calibration effect |\n"
            "|---|---|---|---|---|---|---|---|\n")
    return head + "\n".join(rows) + "\n"


def table3():
    sw = pd.concat([pd.read_csv(p) for p in glob.glob(str(R / "k_checks_sweep" / "S*.csv"))])
    idx = ["condition", "joint", "subject", "delay_ms"]
    kin = sw[sw.kind == "kin"].set_index(idx)
    out = []
    for W in (200, 50):
        fu = sw[(sw.kind == "fused") & (sw.W == W)].set_index(idx)
        sh = sw[(sw.kind == "shift5") & (sw.W == W)].set_index(idx) if W == 200 else None
        recs = []
        for c, j, lab in J3:
            for d in (0, 50, 100, 150):
                k = kin.loc[(c, j)].xs(d, level="delay_ms")
                f = fu.loc[(c, j)].xs(d, level="delay_ms")
                diff = (f.r2 - k.r2).dropna()
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                rmse = f"{k.rmse_degs.median():.1f} → {f.rmse_degs.median():.1f}"
                recs.append(dict(lab=lab, d=d, kin=np.median(k.r2), diff=diff, p=p, rmse=rmse))
        ph = holm(np.array([r["p"] for r in recs]))
        for r, q in zip(recs, ph):
            dm = np.median(r["diff"])
            pos = "yes" if (dm >= 0.01 and q < 0.05) else "no"
            _, lo, hi = boot_ci(r["diff"].values)
            out.append(f"| {W} | {r['lab']} | {r['d']} | {r['kin']:.3f} | {dm:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                       f"{int((r['diff'] > 0).sum())}/17 | {fp(q)} | {r['rmse']} | {pos} |")
    head = ("**Table 3.** Added value of EMG over delayed kinematic feedback for velocity prediction "
            "q̇(t + 100 ms). Kinematics-only model: ridge on [q, q̇](t − d); EMG added by residual fusion. "
            "ΔR²: median paired difference (fused − kinematics) with 95 % bootstrap CI; p: Wilcoxon, Holm-corrected "
            "within each window (12 tests). ΔR² > 0: participants with a positive difference. Meets criterion: median ΔR² ≥ 0.01 and p_Holm < 0.05 (fixed before the analysis). "
            "Time-shifted EMG controls (5, 7, 9 s) at d = 150 ms, W = 200 ms: ΔR² between −0.0006 and −0.0001, "
            "1–6/17 participants positive.\n\n"
            "| W (ms) | Target | d (ms) | R² kinematics | ΔR² (EMG added) | ΔR² > 0 | p_Holm | RMSE (°·s⁻¹) | "
            "Meets criterion |\n|---|---|---|---|---|---|---|---|---|\n")
    return head + "\n".join(out) + "\n"


if __name__ == "__main__":
    txt = "# Tables\n\n> Generated by src/paper_tables.py.\n\n" + T1 + "\n" + table2() + "\n" + table3()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(txt, encoding="utf-8")
    print(txt)

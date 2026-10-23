"""Supplementary material -> paper/supplementary.md. All tables are generated from results/ and cache/."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from data import CACHE_DIR, MUSCLES, SUBJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "supplementary.md"
T6 = [("SJ", "elbow_flexion", "q", "SJ elbow angle"), ("SJ", "elbow_flexion", "qd", "SJ elbow velocity"),
      ("MJ", "shoulder_elv", "q", "MJ shoulder angle"), ("MJ", "shoulder_elv", "qd", "MJ shoulder velocity"),
      ("MJ", "elbow_flexion", "q", "MJ elbow angle"), ("MJ", "elbow_flexion", "qd", "MJ elbow velocity")]
LAB = {(c, j, t): lab for c, j, t, lab in T6}
J3 = [("SJ", "elbow_flexion", "SJ elbow"), ("MJ", "shoulder_elv", "MJ shoulder"), ("MJ", "elbow_flexion", "MJ elbow")]
MODEL = {"ridge25": "Ridge (2.5 s)", "lstm": "LSTM", "tcn": "TCN", "tcn_aux": "TCN + torque output",
         "tau_ridge": "Torque → ridge", "tau_tcn": "Torque → TCN", "emgtau_tcn": "EMG + torque → TCN"}


def md(df, nd=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: "—" if pd.isna(x) else ("> 160" if np.isinf(x) else f"{x:.{nd}f}"))
    return df.to_markdown(index=False)


def lab(df):
    df = df.copy()
    df.insert(0, "Target", [LAB[(c, j, t)] for c, j, t in zip(df.condition, df.joint, df.target)])
    return df.drop(columns=["condition", "joint", "target"])


# ------------------------------------------------------------------ S1
def s1():
    rows, pw, sat = [], [], []
    for s in SUBJECTS:
        z = np.load(CACHE_DIR / f"{s}.npz")
        for i in range(10):
            q, qd = z[f"MJ_{i}_q"], z[f"MJ_{i}_qd"]
            for jj, jn in enumerate(("shoulder", "elbow")):
                g = np.gradient(q[:, jj]) * 100
                rows.append(dict(subject=s, joint=jn, ratio=np.dot(g, qd[:, jj]) / np.dot(qd[:, jj], qd[:, jj]),
                                 r=np.corrcoef(g, qd[:, jj])[0, 1]))
            sat.append(dict(subject=s, frac=np.mean(np.degrees(q[:, 1]) > 129.5)))
            qs, qds = z[f"SJ_{i}_q"][:, 0], z[f"SJ_{i}_qd"][:, 0]
            gs = np.gradient(qs) * 100
            rows.append(dict(subject=s, joint="SJ elbow", ratio=np.dot(gs, qds) / np.dot(qds, qds),
                             r=np.corrcoef(gs, qds)[0, 1]))

        def hf(x):
            f, p = signal.welch(x - x.mean(0), fs=2000, nperseg=1024, axis=0)
            return 100 * p[f > 20].sum() / p.sum()
        pw.append(dict(subject=s, mvc=max(hf(z["mvc_elbow"]), hf(z["mvc_shoulder"])),
                       trial=float(np.median([hf(z[f"SJ_{i}_emg_raw"].astype(float)) for i in (0, 5)]))))
    v = pd.DataFrame(rows).groupby("joint").agg(ratio_min=("ratio", "min"), ratio_median=("ratio", "median"),
                                                ratio_max=("ratio", "max"), r_median=("r", "median")).reset_index()
    v["joint"] = v.joint.replace({"shoulder": "MJ shoulder (as provided)", "elbow": "MJ elbow (as provided)"})
    pw = pd.DataFrame(pw)
    sat = pd.DataFrame(sat).groupby("subject").frac.mean().mul(100)
    k = pd.read_csv(R / "extra" / "dropouts.csv").rename(columns={"condition": "cond"})
    sens = pd.read_csv(R / "extra" / "cpu_sensitivity.csv")
    sens = sens[sens.subset == "dropout_free"].drop_duplicates(["subject", "condition"])
    aff = sens.assign(pct=100 * (1 - sens.frac_kept)).pivot(index="subject", columns="condition", values="pct")
    dk = k.groupby(["subject", "cond"]).agg(n=("sec", "size"), runs=("n_runs", "sum"), sec=("sec", "sum")).unstack(
        "cond").fillna(0)
    d = pd.DataFrame({"Participant": SUBJECTS})
    for c in ("SJ", "MJ"):
        d[f"{c} trial–channels"] = [int(dk[("n", c)].get(s, 0)) for s in SUBJECTS]
        d[f"{c} segments"] = [int(dk[("runs", c)].get(s, 0)) for s in SUBJECTS]
        d[f"{c} duration (s)"] = [round(float(dk[("sec", c)].get(s, 0)), 1) for s in SUBJECTS]
        d[f"{c} affected samples (%)"] = [round(float(aff[c].get(s, 0)), 1) for s in SUBJECTS]
    d["MJ elbow > 129.5° (%)"] = [round(float(sat.get(s, 0)), 1) for s in SUBJECTS]
    musc = k.groupby(["cond", "muscle"]).sec.sum().sort_values(ascending=False).head(6).round(1).reset_index()
    txt = f"""# S1. Data audit

All checks used the files of Zenodo record 11209324 (MD5 of the archive `ea2dd9d26ec61a5870b6455b8abe5918`). Each participant contributed ten SJ and ten MJ trials of ≈ 29.4 s; raw EMG (2 kHz) and kinematics/torque (100 Hz) had consistent lengths (ratio 20.0) and no missing values.

**S1.1 Velocity units.** For each trial, the provided angular velocity was regressed on the numerical derivative of the provided angle (central differences × 100 Hz). Ratio = slope of derivative on provided velocity (Table S1.1). The MJ ratios of ≈ 100 indicate rad·sample⁻¹ units; MJ velocities were multiplied by 100.

**Table S1.1.** Ratio between the numerical derivative of the angle and the provided velocity (all participants and trials).

{md(v)}

**S1.2 MVC recordings.** The share of spectral power above 20 Hz (Welch, 1024-sample segments) was computed for both MVC files of each participant and for SJ trials 0 and 5. MVC files: {pw.mvc.min():.2f}–{pw.mvc.max():.2f} % (all 34 files); trial EMG: {pw.trial.min():.0f}–{pw.trial.max():.0f} %. The MVC files have a positive mean and an autocorrelation ≈ 1 between successive samples, consistent with a processed envelope rather than raw EMG, so MVC normalisation was not used.

**S1.3 Wireless sensor dropouts and S1.4 angle saturation.** A dropout was defined as a run of ≥ 20 identical raw samples (10 ms), the criterion of the online detector; runs of genuine EMG never exceeded 2–3 identical samples (amplitude resolution 0.1007). Over all analysed channels (SJ: 7 elbow muscles; MJ: 12 muscles), {k.n_runs.sum():,} segments were found in {len(k):,} of 3,230 trial–channel pairs, with a total duration of {k.sec.sum():.0f} s. Most segments were brief (10–20 ms), but {int((k.longest_s >= 1).sum())} trial–channel pairs contained segments of 1–{k.longest_s.max():.1f} s. Because every brief segment invalidates all feature windows that contain it, the fraction of affected samples is much larger than the dropout duration itself. Table S1.2 lists, per participant, the number of affected trial–channel pairs and segments, their total duration, the percentage of 100-Hz samples whose 200-ms feature window contained a dropout in any channel, and the percentage of MJ elbow samples above 129.5°. Channels with the longest total dropout duration: {", ".join(f"{r.muscle} ({r.cond}, {r.sec:.0f} s)" for r in musc.itertuples())}.

**Table S1.2.** Dropouts and angle saturation per participant.

{d.to_markdown(index=False)}
"""
    return txt


# ------------------------------------------------------------------ S2
S2 = """# S2. Evidence that the provided kinematics are zero-phase filtered

The source publication states a 4th-order, 3-Hz Butterworth low-pass filter for the kinematics but not its direction [QuesadaIJRR2026]. Four independent indications support zero-phase (forward–backward) filtering.

1. **Statement in the first preprint.** The first bioRxiv version of the source study (posted 12 January 2024, Section II.D.2) states that "all filters were bidirectionally applied" [QuesadaBioRxiv2024v1]. The sentence appears in the EMG subsection; later versions restrict the statement to EMG and do not specify the direction for the kinematics.
2. **Spectral roll-off.** Marker noise is approximately white at high frequency, so after filtering the spectrum follows |H|² for a single pass and |H|⁴ for a forward–backward pass. The slope of the median power spectral density (Welch, DPSS window, NW = 8) of the second difference of the elbow angle between 20 and 40 Hz was compared with simulations of white noise filtered by the stated filter (Table S2.1).
3. **Phase relative to the torque.** In SJ, the reference torque is dominated by the viscous interaction torque measured by the F/T sensor; its spectrum is broadband (slope −28 to −44 dB/decade), indicating that it was not low-pass filtered at 3 Hz. A causal 3-Hz filter would delay the velocity by ≈ 140–146 ms (group delay at 0.3–1 Hz). Instead, the provided velocity *led* the torque by a median of 65 ms (range across participants 40–90 ms; 17/17 participants). Applying an additional causal 3-Hz pass to the velocity shifted this lag by ≈ +150 ms (e.g. S02 trial 0: −80 → +70 ms), confirming the sensitivity of the test.
4. **Practice of the same laboratory.** Other studies from the same group with the same exoskeleton explicitly filter position data with zero-phase Butterworth filters [Verdel2023].

**Table S2.1.** Spectral slope of the angle signal (dB/decade).

| Signal | 2nd difference, 20–40 Hz | 1st difference, 8–20 Hz |
|---|---|---|
| Simulation, single pass (lfilter) | −132 | −72 |
| Simulation, forward–backward (filtfilt) | −293 | −163 |
| **Provided elbow angle** | **−290** | **−192** |

**Future information in the provided kinematics.** For a forward–backward 4th-order 3-Hz Butterworth filter, the impulse response extends into the future. Its energy beyond +50, +100, +150 and +200 ms is 16.4, 3.1, 1.2 and 1.0 % (absolute weight: 27.2, 13.8, 9.0 and 7.7 %); 95 % of the absolute weight lies within ±320 ms. Kinematic inputs delayed by a nominal d therefore still contain some information from after t − d, particularly for d ≤ 50 ms.

Because raw kinematics are not available, the filtering cannot be undone, and re-filtering the provided signals causally would not remove the future information they already contain. The nominal delays are interpreted with this limitation (Sections 2.7 and 4.7).
"""

# ------------------------------------------------------------------ S3
S3 = """# S3. Decision rules fixed before the control analyses, outcomes and deviations

The rules below were written and frozen on 14 October 2026, after the main analyses (within-subject, LOSO, torque reference, horizon, and the TCN early-fusion delay analysis at d = 0 and 150 ms) but before any of the control analyses were run. They were documented internally and were not publicly registered; they fix the interpretation of the control analyses only.

**Primary outcome.** q̇(t + 100 ms) for the SJ elbow, MJ shoulder and MJ elbow; nominal kinematic delays d = 0, 50, 100 and 150 ms; one Holm family of 12 tests; participant-level median paired ΔR², 95 % bootstrap CI and RMSE. *Positive* contribution: median ΔR² ≥ 0.01 and p_Holm < 0.05; *negligible*: CI entirely within ±0.01; otherwise *inconclusive*.

**Table S3.1.** Control analyses, pre-specified rules and outcomes.

| Control | Design | Pre-specified rule | Outcome | Conclusion applied |
|---|---|---|---|---|
| K1 timing diagnostic | Cross-correlation of a causal 25-ms EMG envelope with q̇ and torque | Three-way rule on EMG-to-kinematics lead L | **Cancelled before running**: the filter direction was established independently (S2) | Delays reported as nominal |
| K4 specificity and overfitting | Ridge residual fusion; EMG circularly shifted by 5, 7 and 9 s | Shifted EMG gains as much as aligned EMG (difference < 0.01) or ΔR² ≥ 0.03 → withdraw the delay finding. Aligned EMG beats kinematics and all shifted controls at d = 150 → "time-specific information". Residual fusion at d = 0/50 ≤ 0 or inconclusive → "no EMG contribution detected at ≤ 50 ms" | Shifted controls ΔR² −0.0006 to −0.0001; aligned EMG beat all controls in 17/17 participants for all targets; d = 0/50: ΔR² ≤ +0.0006 (significant, but CIs within ±0.01) | Contribution depends on temporal alignment; negligible at d ≤ 50 ms (worded in the manuscript as "statistically detectable but negligible") |
| K3 window ablation | Ridge: W = 50/100/200 ms, d = 0–160 ms; TCN: W = 50/100 ms, d = 0/50/100/150 | b ≥ 0.6 and TCN (W = 50, d = 50) positive in ≥ 2/3 targets → "mainly window latency"; b ≤ 0.25 and no positive TCN result → "insensitive to window"; otherwise "partly window-dependent" | b = 0.28–0.34 (95 % CI 0.15–0.43); TCN W = 50, d = 50 positive in 2/3 targets | "Partly window-dependent" |
| K2 break-even | Crossing of median EMG-only and kinematics-only curves | d* reported only within the tested range and with CI | See Table S5.4 | Reported within 0–160 ms with CIs |

**Validation gates of K3.** (i) The ridge analysis at W = 200 ms had to reproduce the TCN pattern (ΔR² > 0 at d = 150 ms; ΔR² ≤ 0 at d = 0). At d = 150 ms the gate was met (+0.013 to +0.021). Before the analysis was run, the condition at d = 0 was operationalised in the analysis script as "not significantly positive" (median ΔR² < 0.01 or p_Holm ≥ 0.05), because residual fusion can select a zero gain and is therefore expected to yield ≈ 0 rather than negative values. The median ΔR² at d = 0 was +0.00002 to +0.00006, so the gate was met under this definition. (ii) EMG-only R² at W = 50 ms had to be no more than 0.05 below that at W = 200 ms; it was higher (Table S5.2).

**Deviations and post hoc additions.**
1. *Crossing criterion.* K3 defined d₀ as the zero crossing of the median ΔR² curve. Under residual fusion, ΔR² was never negative, so d₀ was undefined (−∞) for all targets. The crossing of the 0.01 criterion (d₀.₀₁) was used instead, with the same formula for b.
2. *Break-even baseline.* K2 planned the TCN kinematics-only baseline. Because it was much weaker than the ridge baseline (R² 0.54–0.65 vs 0.75–0.81 at d = 150 ms), d* was computed against the ridge baseline (strongest kinematics-only model). After review, the bootstrap CIs were recomputed with 10 000 resamples that include resamples without a crossing (earlier CIs excluded them), and d* was computed for all window lengths.
3. *Delays 50 and 100 ms* (TCN early fusion) were added after inspecting the d = 0 and 150 ms results.
4. *Window sweep.* The full TCN sweep (W = 25–250 ms) and the ridge sweep with W = 25, 75, 150 and 250 ms were added after the control analyses, as were the diagnostic analyses in S5.
5. *Equivalence test.* The TOST for EMG + torque versus torque (S4) was added after review.
"""


# ------------------------------------------------------------------ S4
def s4():
    v = pd.read_csv(R / "v2" / "summary_by_subject.csv")
    m = v.groupby(["model", "condition", "joint", "target"]).agg(
        R2=("r2", "median"), r=("r", "median"), RMSE=("rmse_deg", "median"), nRMSE=("nrmse", "median"),
        bias=("bias_deg", "median")).reset_index()
    m["Model"] = m.model.map(MODEL)
    m = lab(m)
    m["o"] = m.Target.map({t[3]: i for i, t in enumerate(T6)})
    m["mo"] = m.Model.map({v_: i for i, v_ in enumerate(MODEL.values())})
    m = m.sort_values(["o", "mo"])[["Target", "Model", "R2", "r", "RMSE", "nRMSE", "bias"]]
    m = m.rename(columns={"R2": "R²", "RMSE": "RMSE (°, °/s)", "nRMSE": "nRMSE (%)", "bias": "Bias (°, °/s)"})
    st = pd.read_csv(R / "extra" / "stats_paired.csv")
    fam = {"A": "TCN − ridge", "B": "TCN − LSTM", "C": "EMG→TCN − torque→TCN", "D": "(EMG + torque) − torque",
           "E": "TCN + torque output − TCN", "F": "LOSO: 30-s calibration − uncalibrated"}
    st["Comparison"] = st.family.map(fam)
    st = lab(st)[["Comparison", "Target", "wins", "median_diff", "ci_lo", "ci_hi", "p_holm"]]
    st = st.rename(columns={"wins": "Positive (n/17)", "median_diff": "Median ΔR²", "ci_lo": "CI low",
                            "ci_hi": "CI high", "p_holm": "p_Holm"})
    st["p_Holm"] = st["p_Holm"].map(lambda p: f"{p:.2g}")
    to = lab(pd.read_csv(R / "extra" / "tost.csv"))[["Target", "median_diff", "ci90_lo", "ci90_hi", "p_tost"]]
    to["p_tost"] = to.p_tost.map(lambda p: f"{p:.2g}")
    to = to.rename(columns={"median_diff": "Median ΔR²", "ci90_lo": "90 % CI low", "ci90_hi": "90 % CI high",
                            "p_tost": "p_TOST (±0.01)"})
    L = pd.concat([pd.read_csv(p) for p in glob.glob(str(R / "v2_loso" / "metrics_S*.csv"))]
                  + [pd.read_csv(p) for p in glob.glob(str(R / "extra" / "loso_seeds_S*.csv"))])
    L = L.groupby(["model", "subject", "condition", "joint", "target", "variant"], as_index=False)[
        ["r2", "rmse_deg"]].mean()
    lo = L.groupby(["model", "variant", "condition", "joint", "target"]).agg(
        R2=("r2", "median"), R2_min=("r2", "min"), RMSE=("rmse_deg", "median")).reset_index()
    lo["Model"] = lo.model.map({"ridge25": "Ridge", "tcn": "TCN (3 seeds)"})
    lo["Normalisation"] = lo.variant.map({"none": "uncalibrated", "calib30s": "30-s calibration"})
    lo = lab(lo)[["Target", "Model", "Normalisation", "R2", "R2_min", "RMSE"]].rename(
        columns={"R2": "Median R²", "R2_min": "Min R²", "RMSE": "Median RMSE"})
    sv = pd.read_csv(R / "extra" / "cpu_seed_variability.csv")
    sv = lab(sv.groupby(["condition", "joint", "target"]).agg(SD_median=("std", "median"),
                                                              SD_max=("std", "max")).reset_index())
    se = pd.read_csv(R / "extra" / "cpu_sensitivity.csv")
    se = se.groupby(["subset", "condition", "joint", "target"]).r2.median().unstack("subset").reset_index()
    se = lab(se).rename(columns={"all": "All samples", "dropout_free": "Dropout-free",
                                 "dropout_and_saturation_free": "Dropout- and saturation-free"})
    se = se[["Target", "All samples", "Dropout-free", "Dropout- and saturation-free"]]
    return f"""# S4. Additional metrics, statistics and robustness

**Table S4.1.** Within-subject decoding (W = 200 ms, Δ = 0): medians across 17 participants of R², Pearson r, RMSE, range-normalised RMSE and bias. TCN models: three seeds (TCN) or one seed (others). Torque → ridge was an additional reference not discussed in the main text.

{md(m)}

**Table S4.2.** All paired comparisons (Wilcoxon signed-rank, Holm correction within each comparison family of six targets; 95 % bootstrap CI of the median paired difference).

{md(st)}

**Table S4.3.** Equivalence test (two one-sided Wilcoxon tests, margin ±0.01 ΔR²) for EMG + torque versus torque, with 90 % bootstrap CI. Equivalence is shown only where p_TOST < 0.05.

{md(to)}

**Table S4.4.** Leave-one-subject-out results.

{md(lo)}

**Table S4.5.** Standard deviation of R² across the three TCN seeds (per participant; median and maximum across participants).

{md(sv)}

**Table S4.6.** Sensitivity of median TCN R² (seed 0) to excluding samples whose feature window contained a dropout, and additionally MJ elbow samples above 128°.

{md(se)}
"""


# ------------------------------------------------------------------ S5
def s5():
    w = pd.read_csv(R / "extra" / "wsweep_all.csv")
    t = w.groupby(["h", "condition", "joint", "target", "window_ms"]).r2.median().unstack("window_ms").reset_index()
    t = lab(t)
    t.insert(0, "Δ (ms)", t.pop("h"))
    t.columns = [str(c) if not isinstance(c, (int, np.integer)) else f"W={c}" for c in t.columns]
    sw = pd.read_csv(R / "k_checks_sweep" / "sweep_summary.csv")
    sw = sw.assign(Target=[x[2] for c_, j_ in zip(sw.condition, sw.joint) for x in J3 if x[:2] == (c_, j_)])
    sw = sw[["Target", "W", "emg_only", "dR2_d50", "dR2_d100", "dR2_d150", "d001", "win150"]].rename(columns={
        "emg_only": "EMG-only R² (ridge)", "dR2_d50": "ΔR² d=50", "dR2_d100": "ΔR² d=100", "dR2_d150": "ΔR² d=150",
        "d001": "d₀.₀₁ (ms)", "win150": "Positive at d=150 (n/17)"})
    sl = pd.read_csv(R / "k_checks" / "K3A_slope_threshold001.csv")
    sl = sl.assign(Target=[x[2] for c_, j_ in zip(sl.condition, sl.joint) for x in J3 if x[:2] == (c_, j_)])
    sl = sl[["Target", "d01_W50", "d01_W100", "d01_W200", "b", "ci_lo", "ci_hi"]]
    cl = pd.read_csv(R / "extra" / "check_lag_ridge.csv")
    cl = cl.groupby(["h", "condition", "joint", "target", "variant"]).r2.median().unstack("variant").reset_index()
    cl = lab(cl)
    cl.insert(0, "Δ (ms)", cl.pop("h"))
    cl = cl[["Δ (ms)", "Target", "W25", "W200", "W200+50", "W200+90", "W200+120"]]
    # dropout burden
    sens = pd.read_csv(R / "extra" / "cpu_sensitivity.csv")
    sens = sens[sens.subset == "dropout_free"].drop_duplicates(["subject", "condition"])
    burden = sens.assign(b=1 - sens.frac_kept).set_index(["subject", "condition"]).b
    g = w[w.h == 0].pivot_table(index=["subject", "condition", "joint", "target"], columns="window_ms",
                                values="r2")
    gain = (g[25] - g[200]).rename("gain").reset_index()
    rows = []
    for c, j, tg, lb in T6:
        x = gain[(gain.condition == c) & (gain.joint == j) & (gain.target == tg)].copy()
        x["burden"] = [burden[(s, c)] for s in x.subject]
        rho = spearmanr(x.burden, x.gain).statistic
        low = x.nsmallest(8, "burden")
        rows.append({"Target": lb, "Median gain, all 17": x.gain.median(), "Spearman ρ (gain vs burden)": rho,
                     "Median gain, 8 lowest-dropout": low.gain.median(),
                     "Positive, 8 lowest-dropout": f"{int((low.gain > 0).sum())}/8",
                     "Max burden in subgroup (%)": 100 * low.burden.max()})
    db = pd.DataFrame(rows)
    br = pd.read_csv(R / "k_checks" / "K2_breakeven_v2.csv")
    br = br.assign(Target=[x[2] for c_, j_ in zip(br.condition, br.joint) for x in J3 if x[:2] == (c_, j_)])
    br = br[["Target", "emg_model", "W", "emg_r2", "d_star", "ci_lo", "ci_hi", "frac_beyond_160"]].rename(
        columns={"emg_model": "EMG-only model", "emg_r2": "EMG-only R²", "d_star": "d* (ms)", "ci_lo": "CI low",
                 "ci_hi": "CI high", "frac_beyond_160": "Resamples > 160 ms"})
    return f"""# S5. Window length: sweep, diagnostics and break-even latency

**Table S5.1.** TCN EMG-only decoding as a function of the window W (median R² across 17 participants; single seed).

{md(t)}

**Table S5.2.** Ridge delayed-feedback analysis as a function of W: EMG-only ridge R² for q̇(t + 100 ms), median gain from residual fusion at d = 50, 100 and 150 ms, interpolated delay at which the gain reaches 0.01 (d₀.₀₁), and number of participants with a positive gain at d = 150 ms.

{md(sw, 4)}

**Table S5.3.** Dependence of d₀.₀₁ on the window (W = 50, 100 and 200 ms) and slope b = [d₀.₀₁(200) − d₀.₀₁(50)] / 75 ms with 95 % bootstrap CI (2 000 resamples). b = 1 would correspond to a pure effect of mean window age; b = 0 to no window effect.

{md(sl)}

**Table S5.4.** Latency diagnostic (ridge EMG decoder, 0.5 s history; median R²). W200+k: 200-ms features advanced by k ms (non-causal, diagnosis only).

{md(cl)}

**Table S5.5.** Dropout burden and the short-window gain (TCN, Δ = 0; gain = R²(W = 25) − R²(W = 200)). Burden: fraction of samples whose 200-ms window contained a dropout in any channel. Subgroup: the eight participants with the lowest burden in the respective condition.

{md(db)}

**Table S5.6.** Break-even latency d* (q̇, Δ = 100 ms; kinematics-only ridge baseline) for all windows and both EMG-only models. 95 % CIs from 10 000 participant-level bootstrap resamples, including resamples without a crossing within 160 ms ("> 160").

{md(br, 3)}
"""


# ------------------------------------------------------------------ S6
def s6():
    hz = pd.read_csv(R / "horizon" / "all_with_h0.csv")
    hb = pd.read_csv(R / "extra" / "cpu_horizon_baselines.csv")
    a = hz[hz.model.isin(["tcn", "tau_tcn"])].groupby(["model", "condition", "joint", "target", "horizon_ms"]).r2.median()
    b = hb[hb.model == "persist"].groupby(["model", "condition", "joint", "target", "horizon_ms"]).r2.median()
    h = pd.concat([a, b]).unstack("horizon_ms").reset_index()
    h["Model"] = h.model.map({"tcn": "EMG → TCN", "tau_tcn": "Torque → TCN", "persist": "Persistence"})
    h = lab(h.drop(columns="model"))
    h = h[["Target", "Model"] + [c for c in h.columns if isinstance(c, (int, np.integer))]]
    h.columns = [f"Δ={c}" if isinstance(c, (int, np.integer)) else c for c in h.columns]
    sw = pd.concat([pd.read_csv(p) for p in glob.glob(str(R / "k_checks_sweep" / "S*.csv"))])
    idx = ["condition", "joint", "subject", "delay_ms"]
    kin = sw[sw.kind == "kin"].set_index(idx).r2
    rows = []
    for c, j, lb in J3:
        for d in sorted(sw.delay_ms.dropna().unique()):
            r = {"Target": lb, "d (ms)": int(d), "Kinematics R²": kin.loc[(c, j)].xs(d, level="delay_ms").median()}
            for W in (25, 50, 100, 200):
                f = sw[(sw.kind == "fused") & (sw.W == W)].set_index(idx).r2
                r[f"ΔR² W={W}"] = (f - kin).loc[(c, j)].xs(d, level="delay_ms").median()
            rows.append(r)
    rf = pd.DataFrame(rows)
    k4 = pd.read_csv(R / "k_checks" / "K4_tests.csv")
    k4 = k4.assign(Target=[x[2] for c_, j_ in zip(k4.condition, k4.joint) for x in J3 if x[:2] == (c_, j_)])
    k4 = k4[["family", "a", "b", "Target", "wins", "median", "ci_lo", "ci_hi", "p_holm"]]
    k4["family"] = k4.family.map({"K4a_fused_vs_kin": "Aligned EMG vs kinematics",
                                  "K4b_fused_vs_shift": "Aligned vs shifted EMG",
                                  "K4c_shift_vs_kin": "Shifted EMG vs kinematics"})
    k4["p_holm"] = k4.p_holm.map(lambda p: f"{p:.2g}")
    k4 = k4.rename(columns={"family": "Comparison", "a": "Arm A", "b": "Arm B", "wins": "A > B (n/17)",
                            "median": "Median ΔR²", "ci_lo": "CI low", "ci_hi": "CI high", "p_holm": "p_Holm"})
    ef = lab(pd.read_csv(R / "extra" / "stats_delay.csv").sort_values(
        ["condition", "joint", "target", "horizon_ms", "delay_ms"], ascending=[False, True, True, True, True]))
    ef = ef[["Target", "horizon_ms", "delay_ms", "kin", "emgkin", "median_diff", "wins", "p_holm"]].rename(columns={
        "horizon_ms": "Δ (ms)", "delay_ms": "d (ms)", "kin": "Kinematics-only TCN R²", "emgkin": "EMG + kinematics TCN R²",
        "median_diff": "Median ΔR²", "wins": "Positive (n/17)", "p_holm": "p_Holm"})
    ef["p_Holm"] = ef["p_Holm"].map(lambda p: f"{p:.2g}")
    k3 = pd.read_csv(R / "k_checks" / "K3B_tcn.csv")
    k3 = k3.assign(Target=[x[2] for c_, j_ in zip(k3.condition, k3.joint) for x in J3 if x[:2] == (c_, j_)])
    k3 = k3[["W", "d", "Target", "median", "wins", "p_holm", "pos"]].rename(columns={
        "d": "d (ms)", "median": "Median ΔR²", "wins": "Positive (n/17)", "pos": "Meets criterion"})
    k3["p_holm"] = k3.p_holm.map(lambda p: f"{p:.2g}")
    return f"""# S6. Prediction horizon and delayed kinematic feedback

**Table S6.1.** Median R² as a function of the prediction horizon Δ (ms) for the EMG-only TCN, the torque-only TCN and persistence (ŷ(t + Δ) = y(t)). W = 200 ms.

{md(h)}

**Table S6.2.** Residual fusion (ridge) for q̇(t + 100 ms) at every tested delay: median R² of the kinematics-only model and median paired gain from adding EMG for four windows.

{md(rf, 4)}

**Table S6.3.** Specificity controls at d = 0, 50 and 150 ms (W = 200 ms): aligned versus circularly shifted EMG (5, 7, 9 s). Holm correction within each comparison family.

{md(k4, 5)}

**Table S6.4.** TCN early fusion (EMG and delayed kinematics as joint inputs; W = 200 ms; single seed): gain of adding EMG to a kinematics-only TCN, six targets × Δ = 100 and 300 ms. The kinematics-only TCN is considerably weaker than the ridge baseline, so these gains are not used as primary estimates.

{md(ef, 4)}

**Table S6.5.** TCN early fusion for q̇(t + 100 ms) with W = 50, 100 and 200 ms (single seed). Meets criterion: median ΔR² ≥ 0.01 and p_Holm < 0.05.

{md(k3, 4)}
"""


if __name__ == "__main__":
    head = ("# Supplementary material\n\n> Generated by src/supplement.py from the files in results/ and cache/. "
            "Participants: n = 17. Targets: q, angle; q̇, velocity.\n\n")
    txt = head + "\n\n".join([s1(), S2, S3, s4(), s5(), s6()])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(txt, encoding="utf-8")
    print(len(txt.split()), "words")

"""Paper figures -> paper/figures/FigN.png (+ .pdf)."""
import glob
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
                     "figure.dpi": 150, "legend.frameon": False, "font.family": "DejaVu Sans"})
T6 = [("SJ", "elbow_flexion", "q", "SJ elbow\nangle"), ("SJ", "elbow_flexion", "qd", "SJ elbow\nvelocity"),
      ("MJ", "shoulder_elv", "q", "MJ shoulder\nangle"), ("MJ", "shoulder_elv", "qd", "MJ shoulder\nvelocity"),
      ("MJ", "elbow_flexion", "q", "MJ elbow\nangle"), ("MJ", "elbow_flexion", "qd", "MJ elbow\nvelocity")]
J3 = [(("SJ", "elbow_flexion"), "SJ elbow", BLUE), (("MJ", "shoulder_elv"), "MJ shoulder", ORANGE),
      (("MJ", "elbow_flexion"), "MJ elbow", AQUA)]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def title(ax, s):
    ax.set_title(s, loc="left", fontsize=9.5, color=INK, fontweight="bold")


def dotplot(ax, data, series, ylabel, ylim=None, w=0.24):
    n = len(series)
    for si, (key, label, col) in enumerate(series):
        for ti in range(len(T6)):
            v = data.get((key, ti))
            if v is None or len(v) == 0:
                continue
            x = ti + (si - (n - 1) / 2) * w
            jit = np.random.default_rng(ti * 13 + si).uniform(-0.05, 0.05, len(v))
            ax.scatter(x + jit, v, s=7, color=col, alpha=0.35, linewidths=0)
            ax.scatter([x], [np.median(v)], s=48, color=col, edgecolors="white", linewidths=1.5, zorder=3,
                       label=label if ti == 0 else None)
    ax.set_xticks(range(len(T6)), [t[3] for t in T6])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


# ------------------------------------------------------------------ Fig 1
def fig1():
    from data import MUSCLES, load_subject
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1], hspace=0.55, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    title(ax, "a  Causal decoding pipeline")
    bw, step = 0.135, 0.157
    boxes = [(t_, 0.012 + i * step) for i, t_ in enumerate(
        ["Raw EMG\n2 kHz", "Notch +\nband-pass", "Dropout\ndetection", "6 features/ch\nwindow W",
         "z-score\n(train stats)", "Causal TCN\n2.53 s RF"])]
    for txt, x in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.42), bw, 0.4, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc="#eef3fb", ec=BLUE, lw=1, transform=ax.transAxes, clip_on=False))
        ax.text(x + bw / 2, 0.62, txt, ha="center", va="center", fontsize=7, transform=ax.transAxes)
    for _, x in boxes:
        end = 0.968 if x == boxes[-1][1] else x + step - 0.006
        ax.annotate("", xy=(end, 0.62), xytext=(x + bw + 0.008, 0.62), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.text(0.975, 0.62, "q̂, q̂̇\n(t + Δ)", ha="left", va="center", fontsize=8, transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((0.47, 0.0), 0.3, 0.24, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#fdf1ea", ec=ORANGE, lw=1, transform=ax.transAxes, clip_on=False))
    ax.text(0.62, 0.12, "Measured kinematics q, q̇ (t − d)\n(zero-phase filtered; delayed-feedback analyses only)", ha="center",
            va="center", fontsize=7.2, transform=ax.transAxes)
    ax.text(0.79, -0.02, "residual fusion (ridge) or early fusion (TCN)", ha="left", va="center", fontsize=6.3,
            color=ORANGE, transform=ax.transAxes)
    ax.annotate("", xy=(0.975, 0.50), xytext=(0.775, 0.20), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1, connectionstyle="arc3,rad=0.25"))
    # b: dropout example
    d = load_subject("S06")
    t = d["trials"]["SJ"][7]
    k = MUSCLES.index("BICLong")
    x = t.emg_raw[:9000, k].astype(float)
    tt = np.arange(len(x)) / 2000
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(tt, x, color=INK2, lw=0.4)
    eq = np.r_[False, x[1:] == x[:-1]]
    run = np.zeros(len(x), int)
    for i in range(1, len(x)):
        run[i] = run[i - 1] + 1 if eq[i] else 0
    m = run >= 19
    lo, hi = np.percentile(x[~m], [0.5, 99.5]) if (~m).sum() > 100 else (x.min(), x.max())
    pad = 0.3 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.fill_between(tt, lo - pad, hi + pad, where=m, color=ORANGE, alpha=0.18, lw=0, label="detected dropout")
    title(ax, "b  Wireless dropout (raw EMG)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EMG (a.u.)")
    ax.legend(loc="lower left", fontsize=7)
    # c, d: example predictions (median participant, SJ, seed 0)
    v = pd.read_csv(R / "v2" / "summary_by_subject.csv")
    s = v[(v.model == "tcn") & (v.condition == "SJ") & (v.target == "q")].sort_values("r2")
    subj = s.iloc[len(s) // 2].subject
    from run_v2 import features
    F = features(subj)
    P = np.load(R / "v2" / f"preds_{subj}.npz")["SJ|tcn|0"]
    lens = [len(F[f"SJ_{i}_q"]) for i in range(10)]
    e = np.cumsum([0] + lens)
    tr = 3
    y = np.hstack([F[f"SJ_{tr}_q"], F[f"SJ_{tr}_qd"]])
    p = P[e[tr]:e[tr + 1]]
    tt = np.arange(len(y)) / 100
    for col_i, (lab, unit, pos) in enumerate([("angle", "deg", gs[1, 1]), ("velocity", "deg/s", gs[2, :])]):
        ax = fig.add_subplot(pos)
        ax.plot(tt, np.degrees(y[:, col_i]), color=INK, lw=1, label="measured")
        ax.plot(tt, np.degrees(p[:, col_i]), color=BLUE, lw=1, label="TCN (causal)")
        title(ax, f"{'c' if col_i == 0 else 'd'}  SJ elbow {lab} ({subj}, test trial)")
        ax.set_ylabel(f"{lab.capitalize()} ({unit})")
        ax.set_xlabel("Time (s)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7, ncol=2, borderaxespad=0.2)
    save(fig, "Fig1")


# ------------------------------------------------------------------ Fig 2
def fig2():
    v = pd.read_csv(R / "v2" / "summary_by_subject.csv")
    series = [("ridge25", "Ridge (2.5 s)", AQUA), ("lstm", "LSTM", YELLOW), ("tcn", "TCN", BLUE)]
    data = {(k, i): v[(v.model == k) & (v.condition == c) & (v.joint == j) & (v.target == t)].r2.values
            for k, _, _ in series for i, (c, j, t, _) in enumerate(T6)}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    dotplot(ax, data, series, "R² (test, per participant)", (0.3, 1.0))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    save(fig, "Fig2")


# ------------------------------------------------------------------ Fig 3
def fig3():
    v = pd.read_csv(R / "v2" / "summary_by_subject.csv")
    series = [("tcn", "EMG", BLUE), ("tau_tcn", "Reference torque", ORANGE), ("emgtau_tcn", "EMG + torque", AQUA)]
    data = {(k, i): v[(v.model == k) & (v.condition == c) & (v.joint == j) & (v.target == t)].r2.values
            for k, _, _ in series for i, (c, j, t, _) in enumerate(T6)}
    st = pd.read_csv(R / "extra" / "stats_paired.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[2.0, 1]))
    dotplot(axs[0], data, series, "R² (TCN input)", (0.3, 1.0))
    axs[0].tick_params(axis="x", labelsize=6.2)
    axs[0].legend(loc="lower left", ncol=3, fontsize=7.5)
    title(axs[0], "a  Decoding from EMG, torque, or both")
    ax = axs[1]
    s = st[st.family == "D"]
    for i, (c, j, t, lab) in enumerate(T6):
        r = s[(s.condition == c) & (s.joint == j) & (s.target == t)].iloc[0]
        col = BLUE if r.p_holm < 0.05 else INK2
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=2)
        ax.scatter([r.median_diff], [i], color=col, s=30, zorder=3)
        ax.text(0.335, i, f"{int(r.wins)}/17", va="center", fontsize=6.8, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(6), [t[3].replace("\n", " ") for t in T6], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 0.38)
    ax.set_xlabel("ΔR² = (EMG+torque) − torque")
    title(ax, "b  EMG added to torque")
    fig.tight_layout()
    save(fig, "Fig3")


# ------------------------------------------------------------------ Fig 4
def fig4():
    hb = pd.read_csv(R / "extra" / "cpu_horizon_baselines.csv")
    hz = pd.read_csv(R / "horizon" / "all_with_h0.csv")
    kc = pd.concat([pd.read_csv(p) for p in glob.glob(str(R / "k_checks" / "S*.csv"))])
    sw = pd.concat([pd.read_csv(p) for p in glob.glob(str(R / "k_checks_sweep" / "S*.csv"))])
    br = pd.read_csv(R / "k_checks" / "K2_breakeven_v2.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax = axs[0]
    for (c, j), lab, col in J3:
        e = hz[(hz.model == "tcn") & (hz.condition == c) & (hz.joint == j) & (hz.target == "qd")].groupby(
            "horizon_ms").r2.median()
        p = hb[(hb.model == "persist") & (hb.condition == c) & (hb.joint == j) & (hb.target == "qd")].groupby(
            "horizon_ms").r2.median()
        ax.plot(e.index, e.values, color=col, lw=1.6, marker="o", ms=3.5, label=f"{lab} EMG")
        ax.plot(p.index, p.values, color=col, lw=1, ls="--")
    ax.plot([], [], color=INK2, ls="--", lw=1, label="persistence")
    ax.set_xlabel("Horizon Δ (ms)")
    ax.set_ylabel("Velocity R² (median)")
    ax.set_ylim(-0.2, 1.0)
    ax.legend(fontsize=6.3, loc="lower left")
    title(ax, "a  EMG vs persistence")
    ax = axs[1]
    kin = sw[sw.kind == "kin"].set_index(["condition", "joint", "subject", "delay_ms"]).r2
    lf = pd.concat([pd.read_csv(p) for p in glob.glob(str(R / "latefusion" / "S*.csv"))])
    lp = lf.pivot_table(index=["condition", "joint", "subject", "delay_ms"], columns="arm", values="r2")
    for (c, j), lab, col in J3:
        x = lp.loc[(c, j)]
        for arm, ls, lw in (("late_w25", "-", 1.6), ("late_w200", "--", 1.2)):
            d = (x[arm] - x["kin"]).groupby(level=1).median()
            ax.plot(d.index, d.values, color=col, lw=lw, ls=ls)
        fu = sw[(sw.kind == "fused") & (sw.W == 200)].set_index(["condition", "joint", "subject", "delay_ms"]).r2
        d = (fu - kin).loc[(c, j)].groupby(level=1).median()
        ax.plot(d.index, d.values, color=col, lw=0.9, ls=":")
        sd = (x["late_shift5_w200"] - x["kin"]).groupby(level=1).median()
        sd = sd[sd.index.isin([0, 50, 100, 150])]
        off = {"SJ elbow": -4, "MJ shoulder": 0, "MJ elbow": 4}[lab]
        ax.scatter(sd.index + off, sd.values, color=col, marker="x", s=14)
    ax.axhline(0.01, color=INK2, lw=0.7, ls="-.")
    ax.plot([], [], color=INK2, lw=1.6, label="late fusion, W = 25 ms")
    ax.plot([], [], color=INK2, lw=1.2, ls="--", label="late fusion, W = 200 ms")
    ax.plot([], [], color=INK2, lw=0.9, ls=":", label="linear residual fusion")
    ax.scatter([], [], color=INK2, marker="x", s=14, label="time-shifted EMG")
    ax.set_xlabel("Kinematic delay d (ms)")
    ax.set_ylabel("ΔR² (EMG added)")
    ax.legend(fontsize=5.8, loc="upper left")
    title(ax, "b  Added value vs delay")
    ax = axs[2]
    BW = (25, 50, 200)
    for i, ((c, j), lab, col) in enumerate(J3):
        for k, W in enumerate(BW):
            r = br[(br.condition == c) & (br.joint == j) & (br.W == W) & (br.emg_model == "EMG-TCN")].iloc[0]
            y = i * 3.5 + k
            al = 0.45 if W == 200 else 0.9
            lo = r.ci_lo if np.isfinite(r.ci_lo) else 160
            hi = r.ci_hi if np.isfinite(r.ci_hi) else 165
            ax.plot([lo, hi], [y, y], color=col, lw=2, alpha=al)
            if not np.isfinite(r.ci_hi):
                ax.scatter([165], [y], color=col, marker=">", s=22, alpha=al, zorder=3)
            fin = np.isfinite(r.d_star)
            ax.scatter([r.d_star if fin else 165], [y], color=col, s=26, marker="o" if fin else ">", alpha=al,
                       zorder=3)
    ax.axvline(160, color=INK2, lw=0.7, ls=":")
    ax.set_yticks([i * 3.5 + k for i in range(3) for k in range(3)],
                  [f"{x[1]}, W={W}" for x in J3 for W in BW], fontsize=6.3)
    ax.set_xlim(98, 168)
    ax.invert_yaxis()
    ax.set_xlabel("Break-even delay d* (ms)")
    title(ax, "c  Break-even latency")
    fig.tight_layout()
    save(fig, "Fig4")


# ------------------------------------------------------------------ Fig 5
def fig5():
    a = pd.read_csv(R / "extra" / "wsweep_all.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True)
    for ax, (h, t, ttl) in zip(axs, [(0, "q", "a  Angle, Δ = 0"), (0, "qd", "b  Velocity, Δ = 0"),
                                     (100, "qd", "c  Velocity, Δ = 100 ms")]):
        for (c, j), lab, col in J3:
            s = a[(a.h == h) & (a.condition == c) & (a.joint == j) & (a.target == t)].groupby("window_ms").r2
            m = s.median()
            ax.fill_between(m.index, s.quantile(0.25), s.quantile(0.75), color=col, alpha=0.12, lw=0)
            ax.plot(m.index, m.values, color=col, lw=1.6, marker="o", ms=3.5, label=lab)
        ax.axvline(200, color=INK2, ls=":", lw=0.8)
        ax.set_xticks([25, 50, 100, 150, 200, 250])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("Window W (ms)")
        title(ax, ttl)
    axs[0].set_ylabel("R² (median, IQR)")
    axs[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "Fig5")


# ------------------------------------------------------------------ Fig 6
def fig6():
    L = pd.concat([pd.read_csv(p) for p in glob.glob(str(R / "v2_loso" / "metrics_S*.csv"))]
                  + [pd.read_csv(p) for p in glob.glob(str(R / "extra" / "loso_seeds_S*.csv"))])
    L = L[L.model == "tcn"].groupby(["subject", "condition", "joint", "target", "variant"], as_index=False).r2.mean()
    w = pd.read_csv(R / "v2" / "summary_by_subject.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    series = [("none", "LOSO, uncalibrated", AQUA), ("calib30s", "LOSO, 30-s unlabelled calibration", ORANGE),
              ("within", "Within-subject", BLUE)]
    for si, (key, lab, col) in enumerate(series):
        for ti, (c, j, t, _) in enumerate(T6):
            if key == "within":
                v = w[(w.model == "tcn") & (w.condition == c) & (w.joint == j) & (w.target == t)].r2.values
            else:
                v = L[(L.variant == key) & (L.condition == c) & (L.joint == j) & (L.target == t)].r2.values
            x = ti + (si - 1) * 0.26
            vp = ax.violinplot([v], positions=[x], widths=0.24, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col)
                b.set_alpha(0.22)
                b.set_edgecolor("none")
            jit = np.random.default_rng(ti * 7 + si).uniform(-0.04, 0.04, len(v))
            ax.scatter(x + jit, v, s=5, color=col, alpha=0.55, linewidths=0)
            ax.scatter([x], [np.median(v)], color=col, s=30, edgecolors="white", linewidths=1.2, zorder=3,
                       label=lab if ti == 0 else None)
    ax.set_xticks(range(6), [t[3] for t in T6])
    ax.set_ylabel("R² (TCN, per participant)")
    ax.set_ylim(-0.25, 1.0)
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, "Fig6")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print(f.__name__, "ok")

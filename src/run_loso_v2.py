"""Leave-one-subject-out evaluation with causal features (Section 2.5).

Trials T1-T9 of the held-out participant are tested (identical for all variants).
Training: the other 16 participants, T0-T8 for training, T9 for early stopping /
alpha selection.

Normalisation variants (feature z-scores):
  none     : uncalibrated - all participants are normalised with the pooled statistics
             of the training participants; no data of the test participant are used.
  calib30s : each participant is normalised with the unlabelled EMG of their own
             trial T0 (~29 s), including the test participant (T0 is not tested).

Usage: python src/run_loso_v2.py [n_workers]
Additional TCN seeds: python src/run_extra.py loso_seeds
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import JOINTS, ROOT, SUBJECTS  # noqa: E402
from run_v2 import RIDGE_LAGS, ALPHAS, add_history, deep_fp, features, metrics, targets  # noqa: E402

OUT = ROOT / "results" / "v2_loso"
TG = ["q", "qd"]
TEST = list(range(1, 10))
VARIANTS = ["none", "calib30s"]
MODELS = {"ridge25": ("ridge", [0]), "tcn": ("tcn", [0])}


def ridge_big(Xtr, Ytr, Xva, Yva, Xte):
    """Ridge that accumulates the Gram matrix trial by trial (large LOSO training set)."""
    H = lambda x: add_history(x, RIDGE_LAGS)  # noqa: E731
    n = sum(len(x) for x in Xtr)
    sx = sum(H(x).sum(0) for x in Xtr) / n
    sxx = sum((H(x) ** 2).sum(0) for x in Xtr) / n
    mx, sdx = sx, np.sqrt(np.maximum(sxx - sx ** 2, 0)) + 1e-8
    B = np.vstack(Ytr)
    my, sy = B.mean(0), B.std(0) + 1e-8
    D = len(mx)
    G, b = np.zeros((D, D)), np.zeros((D, B.shape[1]))
    for x, y in zip(Xtr, Ytr):
        Z = (H(x) - mx) / sdx
        G += Z.T @ Z
        b += Z.T @ ((y - my) / sy)
    lam, V = np.linalg.eigh(G)
    Vb = V.T @ b
    Zva = np.vstack([(H(x) - mx) / sdx for x in Xva]) @ V
    Tva = (np.vstack(Yva) - my) / sy
    best, W = np.full(B.shape[1], np.inf), np.zeros((D, B.shape[1]))
    for a in ALPHAS * (n / 20000):  # alpha scaled with the amount of data
        C = Vb / (lam + a)[:, None]
        err = np.mean((Zva @ C - Tva) ** 2, axis=0)
        upd = err < best
        best[upd] = err[upd]
        W[:, upd] = (V @ C)[:, upd]
    return [((H(x) - mx) / sdx) @ W * sy + my for x in Xte]


def run_subject(s):
    out_csv = OUT / f"metrics_{s}.csv"
    if out_csv.exists():
        return s, "skipped"
    t0 = time.time()
    Fall = {o: features(o) for o in SUBJECTS}
    rows = []
    for cond in ("SJ", "MJ"):
        J = JOINTS[cond]
        others = [o for o in SUBJECTS if o != s]
        raw = {o: {i: Fall[o][f"{cond}_{i}_emg"] for i in range(10)} for o in SUBJECTS}
        for var in VARIANTS:
            if var == "none":
                St = np.vstack([raw[o][i] for o in others for i in range(9)])
                g = (np.nanmean(St, 0), np.nanstd(St, 0) + 1e-8)
                norm = {o: g for o in SUBJECTS}
            else:
                norm = {o: (np.nanmean(raw[o][0], 0), np.nanstd(raw[o][0], 0) + 1e-8)
                        for o in SUBJECTS}
            N = lambda o, i: np.nan_to_num((raw[o][i] - norm[o][0]) / norm[o][1])  # noqa: E731
            Xtr = [N(o, i) for o in others for i in range(9)]
            Ytr = [targets(Fall[o], cond, i, TG) for o in others for i in range(9)]
            Xva = [N(o, 9) for o in others]
            Yva = [targets(Fall[o], cond, 9, TG) for o in others]
            Xte = [N(s, i) for i in TEST]
            Y = np.vstack([targets(Fall[s], cond, i, TG) for i in TEST])
            for name, (kind, seeds) in MODELS.items():
                for seed in seeds:
                    if kind == "ridge":
                        P = ridge_big(Xtr, Ytr, Xva, Yva, Xte)
                    else:
                        P = deep_fp(kind, seed, Xtr, Ytr, Xva, Yva, Xte)
                    Yh = np.vstack(P)
                    for ti, t in enumerate(TG):
                        for j, joint in enumerate(J):
                            c = ti * len(J) + j
                            rows.append(dict(subject=s, condition=cond, joint=joint, target=t,
                                             model=name, seed=seed, variant=var,
                                             **metrics(Y[:, c], Yh[:, c])))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return s, f"{(time.time() - t0) / 60:.1f} min"


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s, msg in ex.map(run_subject, SUBJECTS):
            print(s, msg, flush=True)
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("metrics_S*.csv"))])
    s = df.groupby(["condition", "joint", "target", "model", "variant"]).agg(
        r2_med=("r2", "median"), r2_mean=("r2", "mean"), rmse_deg=("rmse_deg", "median"))
    pd.set_option("display.width", 200)
    print(s.round(3).to_string())

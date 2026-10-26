"""Within-subject evaluation with the frozen, causal pipeline (Sections 2.3-2.6).

Protocol (each participant and condition): 5 rotating folds; in each fold 2 trials
are tested, 1 trial is used for validation (ridge alpha selection / early stopping)
and 7 for training. Every trial is tested exactly once; metrics are computed on the
concatenation of the 10 test trials of each participant. All models use identical splits.

Models (targets q and qd unless stated otherwise):
  ridge25     EMG, last 2.5 s (26 lags spaced by 10 samples), alpha per output
  tcn         EMG, causal TCN (receptive field ~2.5 s), 3 seeds
  tcn_aux     EMG -> q, qd, tau (torque as auxiliary output), ablation
  lstm        EMG, LSTM
  tau_ridge   REFERENCE: inverse-dynamics torque as input, ridge 2.5 s
  tau_tcn     REFERENCE: inverse-dynamics torque as input, TCN
  emgtau_tcn  REFERENCE: EMG + inverse-dynamics torque as input, TCN
The torque-input models are not real-time (the torque is computed from the
kinematics); they are analytical references for "does EMG carry information beyond
the joint torque?".

Usage: python src/run_v2.py [n_workers] [model,model,...]
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import JOINTS, MUSCLES, ROOT, SJ_MUSCLES, SUBJECTS, load_subject  # noqa: E402
from emgfeat import FEAT_VERSION, extract  # noqa: E402

FEAT_DIR = ROOT / "cache" / f"feats_{FEAT_VERSION}"
OUT = ROOT / "results" / "v2"
FOLDS = [((0, 1), 2), ((2, 3), 4), ((4, 5), 6), ((6, 7), 8), ((8, 9), 0)]
RIDGE_LAGS = list(range(0, 251, 10))
ALPHAS = np.logspace(-1, 5, 13)

MODELS = {
    "ridge25":    dict(kind="ridge", inp="emg", tg=["q", "qd"], seeds=[0]),
    "tcn":        dict(kind="tcn", inp="emg", tg=["q", "qd"], seeds=[0, 1, 2]),
    "tcn_aux":    dict(kind="tcn", inp="emg", tg=["q", "qd", "tau"], seeds=[0]),
    "lstm":       dict(kind="lstm", inp="emg", tg=["q", "qd"], seeds=[0]),
    "tau_ridge":  dict(kind="ridge", inp="tau", tg=["q", "qd"], seeds=[0]),
    "tau_tcn":    dict(kind="tcn", inp="tau", tg=["q", "qd"], seeds=[0]),
    "emgtau_tcn": dict(kind="tcn", inp="emg+tau", tg=["q", "qd"], seeds=[0]),
}


# ---------------------------------------------------------------- features
def features(subject: str) -> dict:
    path = FEAT_DIR / f"{subject}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d = load_subject(subject)
    out = {}
    for cond in ("SJ", "MJ"):
        cols = [MUSCLES.index(m) for m in (SJ_MUSCLES if cond == "SJ" else MUSCLES)]
        for t in d["trials"][cond]:
            n = len(t.q)
            out[f"{cond}_{t.index}_emg"] = extract(t.emg_raw[:, cols], n)
            for tg in ("q", "qd", "tau"):
                out[f"{cond}_{t.index}_{tg}"] = getattr(t, tg)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def add_history(X: np.ndarray, lags: list[int]) -> np.ndarray:
    cols = []
    for l in lags:
        Xl = np.empty_like(X)
        Xl[l:] = X[:len(X) - l] if l else X
        Xl[:l] = X[0]
        cols.append(Xl)
    return np.hstack(cols)


def inputs(F, cond, i, inp):
    parts = []
    if "emg" in inp:
        parts.append(F[f"{cond}_{i}_emg"])
    if "tau" in inp:
        parts.append(F[f"{cond}_{i}_tau"])
    if "kin" in inp:  # measured kinematics (as from exoskeleton encoders): q and qd
        parts += [F[f"{cond}_{i}_q"], F[f"{cond}_{i}_qd"]]
    return np.hstack(parts)


def targets(F, cond, i, tg):
    return np.hstack([F[f"{cond}_{i}_{t}"] for t in tg])


# ---------------------------------------------------------------- models
def ridge_fp(Xtr, Ytr, Xva, Yva, Xte):
    H = lambda L: np.vstack([add_history(x, RIDGE_LAGS) for x in L])  # noqa: E731
    A, B = H(Xtr), np.vstack(Ytr)
    mx, sx = A.mean(0), A.std(0) + 1e-8
    my, sy = B.mean(0), B.std(0) + 1e-8
    Z, T = (A - mx) / sx, (B - my) / sy
    lam, V = np.linalg.eigh(Z.T @ Z)
    Vb = V.T @ (Z.T @ T)
    Zva, Tva = (H(Xva) - mx) / sx, (np.vstack(Yva) - my) / sy
    ZvV = Zva @ V
    best = np.full(T.shape[1], np.inf)
    W = np.zeros((Z.shape[1], T.shape[1]))
    for a in ALPHAS:
        C = Vb / (lam + a)[:, None]
        err = np.mean((ZvV @ C - Tva) ** 2, axis=0)
        upd = err < best
        best[upd] = err[upd]
        W[:, upd] = (V @ C)[:, upd]
    return [((add_history(x, RIDGE_LAGS) - mx) / sx) @ W * sy + my for x in Xte]


def deep_fp(kind, seed, Xtr, Ytr, Xva, Yva, Xte, noise=0.1):
    from deep import predict, train
    B = np.vstack(Ytr)
    my, sy = B.mean(0), B.std(0) + 1e-8
    m = train(kind, Xtr, [(y - my) / sy for y in Ytr], Xva, [(y - my) / sy for y in Yva],
              seed=seed, noise=noise)
    return [p * sy + my for p in predict(m, Xte)]


# ---------------------------------------------------------------- metrics
def metrics(y, yh):
    e = yh - y
    return dict(r=np.corrcoef(y, yh)[0, 1],
                r2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
                nrmse=np.sqrt(np.mean(e ** 2)) / (y.max() - y.min()) * 100,
                rmse_deg=np.degrees(np.sqrt(np.mean(e ** 2))),
                bias_deg=np.degrees(e.mean()))


# ---------------------------------------------------------------- participant
def run_subject(args):
    subject, model_names = args
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    out_csv = OUT / f"metrics_{subject}.csv"
    if out_csv.exists() and (OUT / f"preds_{subject}.npz").exists():
        return subject, "skipped"
    t0 = time.time()
    F = features(subject)
    rows, preds_store = [], {}
    for cond in ("SJ", "MJ"):
        J = JOINTS[cond]
        # preds[(model, seed)][trial] = (T, len(tg)) prediction
        preds = {}
        for test, val in FOLDS:
            train_idx = [i for i in range(10) if i not in test and i != val]
            for name in model_names:
                spec = MODELS[name]
                Xall = {i: inputs(F, cond, i, spec["inp"]) for i in range(10)}
                St = np.vstack([Xall[i] for i in train_idx])
                mu, sd = np.nanmean(St, 0), np.nanstd(St, 0) + 1e-8
                Xn = {i: np.nan_to_num((x - mu) / sd) for i, x in Xall.items()}
                Ytr = [targets(F, cond, i, spec["tg"]) for i in train_idx]
                Yva = [targets(F, cond, val, spec["tg"])]
                Xtr, Xva = [Xn[i] for i in train_idx], [Xn[val]]
                Xte = [Xn[i] for i in test]
                for seed in spec["seeds"]:
                    if spec["kind"] == "ridge":
                        P = ridge_fp(Xtr, Ytr, Xva, Yva, Xte)
                    else:
                        P = deep_fp(spec["kind"], seed, Xtr, Ytr, Xva, Yva, Xte)
                    for i, p in zip(test, P):
                        preds.setdefault((name, seed), {})[i] = p
        for (name, seed), pt in preds.items():
            tg = MODELS[name]["tg"]
            Yh = np.vstack([pt[i] for i in range(10)])
            Y = np.vstack([targets(F, cond, i, tg) for i in range(10)])
            preds_store[f"{cond}|{name}|{seed}"] = Yh.astype(np.float32)
            for ti, t in enumerate(tg):
                for j, joint in enumerate(J):
                    c = ti * len(J) + j
                    rows.append(dict(subject=subject, condition=cond, joint=joint, target=t,
                                     model=name, seed=seed, **metrics(Y[:, c], Yh[:, c])))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"preds_{subject}.npz", **preds_store)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return subject, f"{(time.time() - t0) / 60:.1f} min"


def summarize():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("metrics_S*.csv"))])
    df = df[df.target != "tau"]
    # average over seeds -> one value per participant
    d = df.groupby(["subject", "condition", "joint", "target", "model"], as_index=False).mean(
        numeric_only=True)
    s = d.groupby(["condition", "joint", "target", "model"]).agg(
        n=("r2", "size"), r2_med=("r2", "median"), r2_mean=("r2", "mean"),
        r=("r", "mean"), rmse_deg=("rmse_deg", "median"))
    pd.set_option("display.width", 200)
    print(s.round(3).to_string())
    d.to_csv(OUT / "summary_by_subject.csv", index=False)
    return d


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:   # summary only: metrics_S*.csv -> summary_by_subject.csv (no data needed)
        summarize()
        sys.exit()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(MODELS)
    OUT.mkdir(parents=True, exist_ok=True)
    # compute the features first, in parallel (CPU)
    with ProcessPoolExecutor(max_workers=8) as ex:
        list(ex.map(features, SUBJECTS))
    print("features ready", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s, msg in ex.map(run_subject, [(s, names) for s in SUBJECTS]):
            print(s, msg, flush=True)
    summarize()

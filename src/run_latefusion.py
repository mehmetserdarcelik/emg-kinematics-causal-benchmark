"""EMG-only TCN predictions for late fusion (GPU).

EMG-only TCN for q, qd (t + 100 ms) (same design as run_extra.run_wsweep, seed 0); predictions are saved for the
validation trial and the test trials of every fold (the fusion weight is selected on the validation trial).
Arms: W = 25, 50, 200 ms; negative control: W = 200 with the EMG circularly shifted by 5 s within each trial.
Output: results/latefusion/preds_SXX.npz, key "{cond}|{arm}|f{fold}|t{trial}" -> (T - 10, 2J) [q, qd].
Usage: python src/run_latefusion.py [round2]   (finished participants are skipped; round2: ARMS2 -> preds2_SXX.npz)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import ROOT, SUBJECTS  # noqa: E402
from run_v2 import FOLDS, deep_fp, features, targets  # noqa: E402

OUT = ROOT / "results" / "latefusion"
TG = ["q", "qd"]
K = 10                                   # Δ = 100 ms
ARMS = [("w25", 25, 0, 0), ("w50", 50, 0, 0), ("w200", 200, 0, 0), ("shift5_w200", 200, 5, 0)]
# matched negative control for the short window and a second seed -> preds2_SXX.npz
ARMS2 = [("shift5_w25", 25, 5, 0), ("seed1_w25", 25, 0, 1), ("seed1_w200", 200, 0, 1)]


def _norm(Xall, tr):
    St = np.vstack([Xall[i] for i in tr])
    mu, sd = np.nanmean(St, 0), np.nanstd(St, 0) + 1e-8
    return {i: np.nan_to_num((x - mu) / sd) for i, x in Xall.items()}


def run_subject(subject, arms=None, tag="preds"):
    arms = arms or ARMS
    out = OUT / f"{tag}_{subject}.npz"
    if out.exists():
        return subject, "skipped"
    from featw import emg_feats
    t0 = time.time()
    F0 = features(subject)
    store = {}
    for arm, W, shift_s, seed in arms:
        E = emg_feats(subject, W)
        for cond in ("SJ", "MJ"):
            n = {i: len(F0[f"{cond}_{i}_q"]) for i in range(10)}
            emg = {i: E[f"{cond}_{i}_emg"][:n[i]] for i in range(10)}
            if shift_s:
                emg = {i: np.roll(x, shift_s * 100, axis=0) for i, x in emg.items()}
            Xall = {i: emg[i][:n[i] - K] for i in range(10)}
            Yall = {i: targets(F0, cond, i, TG)[K:] for i in range(10)}
            for f, (test, val) in enumerate(FOLDS):
                tr = [i for i in range(10) if i not in test and i != val]
                Xn = _norm(Xall, tr)
                ids = [val, *test]
                P = deep_fp("tcn", seed, [Xn[i] for i in tr], [Yall[i] for i in tr],
                            [Xn[val]], [Yall[val]], [Xn[i] for i in ids])
                for i, p in zip(ids, P):
                    store[f"{cond}|{arm}|f{f}|t{i}"] = p.astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **store)
    return subject, f"{(time.time() - t0) / 60:.1f} min"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    second = len(sys.argv) > 1 and sys.argv[1] == "round2"
    jobs = [(s, ARMS2, "preds2") if second else (s, ARMS, "preds") for s in SUBJECTS]
    with ProcessPoolExecutor(max_workers=1) as ex:
        for s, msg in ex.map(run_subject, *zip(*jobs)):
            print("latefusion", s, msg, flush=True)

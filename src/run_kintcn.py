"""Kinematics-only TCN predictions for the ensemble control of late fusion (GPU).

Input [q, qd](t - d), target increment y(t + 100 ms) - y(t - d) (same design as run_extra.run_hbase2 'kin_inc', seed 0);
absolute predictions (y(t - d) + increment) are saved for the validation and test trials of every fold. d in {100, 150} ms.
This TCN is fused with the ridge in place of the EMG-only TCN: a gain of ~0 shows that the EMG gain is not an ensemble effect.
Output: results/latefusion/kintcn_SXX.npz, key "{cond}|kintcn_d{d}|f{fold}|t{trial}" -> (T - d - 10, 2J) [q, qd].
Usage: python src/run_kintcn.py
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import ROOT, SUBJECTS  # noqa: E402
from run_v2 import FOLDS, deep_fp, features  # noqa: E402

OUT = ROOT / "results" / "latefusion"
K = 10
DELAYS = [100, 150]


def _norm(Xall, tr):
    St = np.vstack([Xall[i] for i in tr])
    mu, sd = np.nanmean(St, 0), np.nanstd(St, 0) + 1e-8
    return {i: np.nan_to_num((x - mu) / sd) for i, x in Xall.items()}


def run_subject(subject):
    out = OUT / f"kintcn_{subject}.npz"
    if out.exists():
        return subject, "skipped"
    t0 = time.time()
    F = features(subject)
    store = {}
    for cond in ("SJ", "MJ"):
        for d_ms in DELAYS:
            d = d_ms // 10
            B = {}
            for i in range(10):
                kin = np.hstack([F[f"{cond}_{i}_q"], F[f"{cond}_{i}_qd"]])
                t = np.arange(d, len(kin) - K)
                B[i] = (kin[t - d], kin[t - d], kin[t + K])
            for f, (test, val) in enumerate(FOLDS):
                tr = [i for i in range(10) if i not in test and i != val]
                Xn = _norm({i: B[i][0] for i in range(10)}, tr)
                inc = {i: B[i][2] - B[i][1] for i in range(10)}
                ids = [val, *test]
                P = deep_fp("tcn", 0, [Xn[i] for i in tr], [inc[i] for i in tr],
                            [Xn[val]], [inc[val]], [Xn[i] for i in ids])
                for i, p in zip(ids, P):
                    store[f"{cond}|kintcn_d{d_ms}|f{f}|t{i}"] = (B[i][1] + p).astype(np.float32)
    np.savez_compressed(out, **store)
    return subject, f"{(time.time() - t0) / 60:.1f} min"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=1) as ex:
        for s, msg in ex.map(run_subject, SUBJECTS):
            print("kintcn", s, msg, flush=True)

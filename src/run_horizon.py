"""Prediction horizon: q(t + D) and qd(t + D) from EMG (and the reference torque).

D in {50, 100, 200, 300} ms (D = 0 results are in results/v2, same folds).
Protocol as in run_v2 (5 rotating folds, 7 training / 1 validation / 2 test).
The last D samples of each trial are discarded for the shifted target.

Usage: python src/run_horizon.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import JOINTS, ROOT, SUBJECTS  # noqa: E402
from run_v2 import FOLDS, MODELS, deep_fp, features, inputs, metrics, targets  # noqa: E402

OUT = ROOT / "results" / "horizon"
HORIZONS_MS = [50, 100, 200, 300]
MODEL_NAMES = ["tcn", "tau_tcn"]


def run_subject(subject: str) -> str:
    out_csv = OUT / f"metrics_{subject}.csv"
    if out_csv.exists():
        return "skipped"
    t0 = time.time()
    F = features(subject)
    rows = []
    for cond in ("SJ", "MJ"):
        J = JOINTS[cond]
        for h_ms in HORIZONS_MS:
            k = h_ms // 10
            for name in MODEL_NAMES:
                spec = MODELS[name]
                preds = {}
                for test, val in FOLDS:
                    tr = [i for i in range(10) if i not in test and i != val]
                    Xall = {i: inputs(F, cond, i, spec["inp"])[:-k] for i in range(10)}
                    Yall = {i: targets(F, cond, i, spec["tg"])[k:] for i in range(10)}
                    St = np.vstack([Xall[i] for i in tr])
                    mu, sd = np.nanmean(St, 0), np.nanstd(St, 0) + 1e-8
                    Xn = {i: np.nan_to_num((x - mu) / sd) for i, x in Xall.items()}
                    P = deep_fp(spec["kind"], 0, [Xn[i] for i in tr], [Yall[i] for i in tr],
                                [Xn[val]], [Yall[val]], [Xn[i] for i in test])
                    for i, p in zip(test, P):
                        preds[i] = p
                Yh = np.vstack([preds[i] for i in range(10)])
                Y = np.vstack([targets(F, cond, i, spec["tg"])[k:] for i in range(10)])
                for ti, t in enumerate(spec["tg"]):
                    for j, joint in enumerate(J):
                        c = ti * len(J) + j
                        rows.append(dict(subject=subject, condition=cond, joint=joint, target=t,
                                         model=name, horizon_ms=h_ms,
                                         **metrics(Y[:, c], Yh[:, c])))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return f"{(time.time() - t0) / 60:.1f} min"


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    # as in run_v2: computation in a separate process (loading torch and MKL together
    # in the main process can cause an OpenMP conflict on Windows)
    with ProcessPoolExecutor(max_workers=1) as ex:
        for s, msg in zip(SUBJECTS, ex.map(run_subject, SUBJECTS)):
            print(s, msg, flush=True)

"""CPU analyses (no training): horizon baselines, seed variability, sensitivity.

1) Prediction-horizon baselines (D = 50, 100, 200, 300 ms):
     persist   : y(t+D) = y(t)               (measured kinematics)
     emg_hold  : y(t+D) = y_EMG(t)           (EMG-TCN prediction at D = 0, seed 0)
     emg_extrap: q(t+D) = q_EMG(t) + D * qd_EMG(t)  (q only)
2) Variability across seeds (TCN, 3 seeds).
3) Sensitivity: TCN scores after excluding samples whose 200-ms window contains a dropout,
   and additionally MJ elbow samples above 128 deg.
Outputs: results/extra/cpu_*.csv
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import JOINTS, MUSCLES, ROOT, SJ_MUSCLES, SUBJECTS, load_subject  # noqa: E402
from emgfeat import DECIM, _rsum, dropout_mask  # noqa: E402
from run_v2 import features, metrics, targets  # noqa: E402

OUT = ROOT / "results" / "extra"
HS = [50, 100, 200, 300]


def _preds(subject, cond, model="tcn", seed=0):
    P = np.load(ROOT / "results" / "v2" / f"preds_{subject}.npz")
    return P[f"{cond}|{model}|{seed}"]


def horizon_baselines(subject):
    F = features(subject)
    rows = []
    for cond in ("SJ", "MJ"):
        J = JOINTS[cond]
        nJ = len(J)
        lens = [len(F[f"{cond}_{i}_q"]) for i in range(10)]
        e = np.cumsum([0] + lens)
        P = _preds(subject, cond)  # q(J), qd(J)
        for h in HS:
            k = h // 10
            Y, Ypers, Yhold, Yext = [], [], [], []
            for i in range(10):
                y = targets(F, cond, i, ["q", "qd"])
                p = P[e[i]:e[i + 1]]
                Y.append(y[k:])
                Ypers.append(y[:-k])
                Yhold.append(p[:-k])
                ext = p[:-k].copy()
                ext[:, :nJ] = p[:-k, :nJ] + (h / 1000) * p[:-k, nJ:]
                Yext.append(ext)
            Y = np.vstack(Y)
            for name, Yh in (("persist", np.vstack(Ypers)), ("emg_hold", np.vstack(Yhold)),
                             ("emg_extrap", np.vstack(Yext))):
                for ti, t in enumerate(["q", "qd"]):
                    if name == "emg_extrap" and t == "qd":
                        continue
                    for j, joint in enumerate(J):
                        c = ti * nJ + j
                        rows.append(dict(subject=subject, condition=cond, joint=joint, target=t,
                                         model=name, horizon_ms=h, **metrics(Y[:, c], Yh[:, c])))
    return rows


def sensitivity(subject):
    F = features(subject)
    d = load_subject(subject)
    rows = []
    w = int(0.2 * 2000)
    for cond in ("SJ", "MJ"):
        J = JOINTS[cond]
        nJ = len(J)
        cols = [MUSCLES.index(m) for m in (SJ_MUSCLES if cond == "SJ" else MUSCLES)]
        lens = [len(F[f"{cond}_{i}_q"]) for i in range(10)]
        e = np.cumsum([0] + lens)
        P = _preds(subject, cond)
        Y = np.vstack([targets(F, cond, i, ["q", "qd"]) for i in range(10)])
        bad = []
        for t in d["trials"][cond]:
            m = dropout_mask(t.emg_raw[:, cols])
            b = (_rsum(m.astype(float), w) > 0).any(axis=1)[::DECIM][:len(t.q)]
            if len(b) < len(t.q):
                b = np.r_[b, np.repeat(b[-1], len(t.q) - len(b))]
            bad.append(b)
        bad = np.concatenate(bad)
        sat = np.zeros(len(Y), bool)
        if cond == "MJ":
            sat = np.degrees(Y[:, J.index("elbow_flexion")]) > 128
        for mask_name, keep in (("all", np.ones(len(Y), bool)), ("dropout_free", ~bad),
                                ("dropout_and_saturation_free", ~bad & ~sat)):
            for ti, t in enumerate(["q", "qd"]):
                for j, joint in enumerate(J):
                    c = ti * nJ + j
                    rows.append(dict(subject=subject, condition=cond, joint=joint, target=t,
                                     subset=mask_name, frac_kept=keep.mean(),
                                     **metrics(Y[keep, c], P[keep, c])))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        hb = [r for rs in ex.map(horizon_baselines, SUBJECTS) for r in rs]
        se = [r for rs in ex.map(sensitivity, SUBJECTS) for r in rs]
    pd.DataFrame(hb).to_csv(OUT / "cpu_horizon_baselines.csv", index=False)
    pd.DataFrame(se).to_csv(OUT / "cpu_sensitivity.csv", index=False)
    # seed variability
    m = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "results" / "v2").glob("metrics_S*.csv"))])
    s = m[(m.model == "tcn") & (m.target != "tau")].groupby(
        ["subject", "condition", "joint", "target"]).r2.agg(["mean", "std"]).reset_index()
    s.to_csv(OUT / "cpu_seed_variability.csv", index=False)
    print("ok")

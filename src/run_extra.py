"""Additional GPU analyses. Protocol as in run_v2 (5 rotating folds).

loso_seeds : additional seeds (1, 2) of the LOSO TCN, both normalisation variants.
hbase2     : delayed kinematic feedback, TCN early fusion, d = 0 and 150 ms (Section 2.7).
hbase3     : as hbase2, d = 50 ms.
hbase4     : as hbase2, d = 100 ms.
k3b        : TCN early fusion with W = 50 and 100 ms; EMG-only TCN at D = 100 ms.
wsweep     : TCN window sweep, W in {25, 50, 75, 100, 150, 250} ms (Section 2.8).

Usage: python src/run_extra.py loso_seeds|hbase2|hbase3|hbase4|k3b|wsweep
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
from run_v2 import FOLDS, deep_fp, features, inputs, metrics, targets  # noqa: E402

OUT = ROOT / "results" / "extra"
TG = ["q", "qd"]
HBASE_MS = [100, 300]


def _rows(subject, cond, model, Y, Yh, tg, **extra):
    J = JOINTS[cond]
    out = []
    for ti, t in enumerate(tg):
        for j, joint in enumerate(J):
            c = ti * len(J) + j
            out.append(dict(subject=subject, condition=cond, joint=joint, target=t,
                            model=model, **extra, **metrics(Y[:, c], Yh[:, c])))
    return out


def _norm(Xall, tr):
    St = np.vstack([Xall[i] for i in tr])
    mu, sd = np.nanmean(St, 0), np.nanstd(St, 0) + 1e-8
    return {i: np.nan_to_num((x - mu) / sd) for i, x in Xall.items()}, (mu, sd)


def run_loso_seeds(subject):
    import run_loso_v2 as L
    rows = []
    Fall = {o: features(o) for o in SUBJECTS}
    for cond in ("SJ", "MJ"):
        others = [o for o in SUBJECTS if o != subject]
        raw = {o: {i: Fall[o][f"{cond}_{i}_emg"] for i in range(10)} for o in SUBJECTS}
        for var in L.VARIANTS:
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
            Xte = [N(subject, i) for i in L.TEST]
            Y = np.vstack([targets(Fall[subject], cond, i, TG) for i in L.TEST])
            for seed in (1, 2):
                Yh = np.vstack(deep_fp("tcn", seed, Xtr, Ytr, Xva, Yva, Xte))
                rows += _rows(subject, cond, "tcn", Y, Yh, TG, seed=seed, variant=var)
    return rows


HB2_DELAYS_MS = [0, 150]


def run_hbase2(subject, delays_ms=None):
    """Early fusion with an increment target and delayed kinematics.

    Measured kinematics arrive with a delay of d ms: input [EMG(t), q(t-d), qd(t-d)].
    Target increment: y(t+D) - y(t-d); prediction = y(t-d) + predicted increment.
    Also reports persistence from the delayed kinematics (persist_d).
    """
    F = features(subject)
    rows = []
    for cond in ("SJ", "MJ"):
        for h in HBASE_MS:
            k = h // 10
            for d_ms in (delays_ms or HB2_DELAYS_MS):
                d = d_ms // 10

                def build(i, use_emg):
                    kin = np.hstack([F[f"{cond}_{i}_q"], F[f"{cond}_{i}_qd"]])
                    t = np.arange(d, len(kin) - k)
                    parts = [F[f"{cond}_{i}_emg"][t]] if use_emg else []
                    parts.append(kin[t - d])
                    return np.hstack(parts), kin[t - d], kin[t + k]

                Y = np.vstack([build(i, False)[2] for i in range(10)])
                ref = np.vstack([build(i, False)[1] for i in range(10)])
                rows += _rows(subject, cond, f"persist_d{d_ms}", Y, ref, TG, horizon_ms=h)
                for name, use_emg in (("kin_inc", False), ("emgkin_inc", True)):
                    B = {i: build(i, use_emg) for i in range(10)}
                    preds = {}
                    for test, val in FOLDS:
                        tr = [i for i in range(10) if i not in test and i != val]
                        Xn, _ = _norm({i: B[i][0] for i in range(10)}, tr)
                        inc = {i: B[i][2] - B[i][1] for i in range(10)}
                        out = deep_fp("tcn", 0, [Xn[i] for i in tr], [inc[i] for i in tr],
                                      [Xn[val]], [inc[val]], [Xn[i] for i in test])
                        for i, p in zip(test, out):
                            preds[i] = B[i][1] + p
                    Yh = np.vstack([preds[i] for i in range(10)])
                    rows += _rows(subject, cond, f"{name}_d{d_ms}", Y, Yh, TG, horizon_ms=h)
    return rows


def run_hbase3(subject):
    """Intermediate delay: kinematics delayed by 50 ms (same design as hbase2)."""
    return run_hbase2(subject, delays_ms=[50])


def run_hbase4(subject):
    """Intermediate delay: kinematics delayed by 100 ms (same design as hbase2)."""
    return run_hbase2(subject, delays_ms=[100])


def run_k3b(subject):
    """Window ablation with the TCN (control K3-B, Supplementary S3).

    For W in {50, 100} ms (W = 200 results come from hbase* and run_horizon):
      emgkin_inc_d{d}_w{W}: same design as run_hbase2, D = 100 ms, d = 0/50/100/150
      emg_only_w{W}       : EMG only -> q, qd (t + 100 ms), same design as run_horizon
    The kinematics-only model (kin_inc) does not depend on W; the hbase* results are used.
    """
    from featw import emg_feats
    F0 = features(subject)
    rows = []
    h, k = 100, 10
    for W in (50, 100):
        E = emg_feats(subject, W)
        F = dict(F0)
        F.update(E)
        for cond in ("SJ", "MJ"):
            for d_ms in (0, 50, 100, 150):
                d = d_ms // 10

                def build(i):
                    kin = np.hstack([F[f"{cond}_{i}_q"], F[f"{cond}_{i}_qd"]])
                    t = np.arange(d, len(kin) - k)
                    return np.hstack([F[f"{cond}_{i}_emg"][t], kin[t - d]]), kin[t - d], kin[t + k]

                B = {i: build(i) for i in range(10)}
                preds = {}
                for test, val in FOLDS:
                    tr = [i for i in range(10) if i not in test and i != val]
                    Xn, _ = _norm({i: B[i][0] for i in range(10)}, tr)
                    inc = {i: B[i][2] - B[i][1] for i in range(10)}
                    out = deep_fp("tcn", 0, [Xn[i] for i in tr], [inc[i] for i in tr],
                                  [Xn[val]], [inc[val]], [Xn[i] for i in test])
                    for i, p in zip(test, out):
                        preds[i] = B[i][1] + p
                Y = np.vstack([B[i][2] for i in range(10)])
                Yh = np.vstack([preds[i] for i in range(10)])
                rows += _rows(subject, cond, f"emgkin_inc_d{d_ms}_w{W}", Y, Yh, TG,
                              horizon_ms=h, window_ms=W, delay_ms=d_ms)
            # EMG only, D = 100 ms
            preds = {}
            for test, val in FOLDS:
                tr = [i for i in range(10) if i not in test and i != val]
                Xall = {i: F[f"{cond}_{i}_emg"][:-k] for i in range(10)}
                Yall = {i: targets(F, cond, i, TG)[k:] for i in range(10)}
                Xn, _ = _norm(Xall, tr)
                out = deep_fp("tcn", 0, [Xn[i] for i in tr], [Yall[i] for i in tr],
                              [Xn[val]], [Yall[val]], [Xn[i] for i in test])
                for i, p in zip(test, out):
                    preds[i] = p
            Y = np.vstack([targets(F, cond, i, TG)[k:] for i in range(10)])
            Yh = np.vstack([preds[i] for i in range(10)])
            rows += _rows(subject, cond, f"emg_only_w{W}", Y, Yh, TG, horizon_ms=h, window_ms=W)
    return rows


def run_wsweep(subject):
    """Systematic EMG window sweep (TCN, seed 0), W in {25, 50, 75, 100, 150, 200, 250} ms.

    tcn_w{W}      : EMG -> q, qd (D = 0), same design as run_v2 "tcn"
    emg_only_w{W} : EMG -> q, qd (D = 100 ms), same design as run_horizon
    Already computed elsewhere and skipped here: D = 100 for W = 50/100 (k3b) and W = 200
    (run_horizon); D = 0 for W = 200 (run_v2, seed 0). See src/aggregate.py.
    """
    from featw import emg_feats
    F0 = features(subject)
    rows = []
    parts = OUT / "wsweep_parts"
    parts.mkdir(parents=True, exist_ok=True)
    for W in (25, 50, 75, 100, 150, 250):
        part = parts / f"{subject}_w{W}.csv"
        if part.exists():  # checkpoint: skip windows that are already finished
            rows += pd.read_csv(part).to_dict("records")
            continue
        n_before = len(rows)
        E = emg_feats(subject, W)
        F = dict(F0)
        F.update(E)
        horizons = [0] if W in (50, 100) else [0, 100]
        for cond in ("SJ", "MJ"):
            for h in horizons:
                k = h // 10
                preds = {}
                for test, val in FOLDS:
                    tr = [i for i in range(10) if i not in test and i != val]
                    Xall = {i: F[f"{cond}_{i}_emg"][:len(F[f"{cond}_{i}_q"]) - k] for i in range(10)}
                    Yall = {i: targets(F, cond, i, TG)[k:] for i in range(10)}
                    Xn, _ = _norm(Xall, tr)
                    out = deep_fp("tcn", 0, [Xn[i] for i in tr], [Yall[i] for i in tr],
                                  [Xn[val]], [Yall[val]], [Xn[i] for i in test])
                    for i, p in zip(test, out):
                        preds[i] = p
                Y = np.vstack([targets(F, cond, i, TG)[k:] for i in range(10)])
                Yh = np.vstack([preds[i] for i in range(10)])
                name = f"tcn_w{W}" if h == 0 else f"emg_only_w{W}"
                rows += _rows(subject, cond, name, Y, Yh, TG, horizon_ms=h, window_ms=W)
        pd.DataFrame(rows[n_before:]).to_csv(part, index=False)
    return rows


MODES = {"loso_seeds": run_loso_seeds, "hbase2": run_hbase2, "hbase3": run_hbase3,
         "hbase4": run_hbase4, "k3b": run_k3b, "wsweep": run_wsweep}


def run_one(args):
    mode, subject = args
    out = OUT / f"{mode}_{subject}.csv"
    if out.exists():
        return subject, "skipped"
    t0 = time.time()
    pd.DataFrame(MODES[mode](subject)).to_csv(out, index=False)
    return subject, f"{(time.time() - t0) / 60:.1f} min"


if __name__ == "__main__":
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    # computation in a subprocess to avoid the torch/MKL OpenMP conflict (see deep.py)
    with ProcessPoolExecutor(max_workers=1) as ex:
        for s, msg in ex.map(run_one, [(mode, s) for s in SUBJECTS]):
            print(mode, s, msg, flush=True)

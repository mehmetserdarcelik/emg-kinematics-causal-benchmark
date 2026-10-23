"""Controls K3-A (window x delay) and K4 (residual fusion + shifted EMG), ridge, CPU.

Design and decision rules: Supplementary S3 (fixed before the analysis was run).
Primary outcome: qd, D = 100 ms. 5 rotating folds (7 training / 1 validation / 2 test).

For each (condition, d, fold):
  kin      : ridge, input [q, qd](t-d) + history (0:50:500 ms) -> target qd(t+D) - qd(t-d)
  fused    : kin + c * EMG-ridge(residual). Residuals are obtained within the training
             trials by leave-one-trial-out cross-prediction; the EMG-ridge alpha and
             c in {0, .25, .5, 1} are selected on the validation trial. W in {50, 100, 200}.
  shiftS   : as fused (W = 200), with the EMG circularly shifted by S s within each trial;
             S in {5, 7, 9}; d in {0, 50, 150}.
  emg_only : EMG only (W) -> qd(t+D), absolute target (independent of d).
Output: results/k_checks/SXX.csv (participant x condition x joint x arm x W x d; R² and
RMSE over the concatenated 10 test trials).
With K_SWEEP=1: window sweep (7 windows) -> results/k_checks_sweep/SXX.csv (Section 2.8).
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
from featw import emg_feats  # noqa: E402
from run_v2 import FOLDS, add_history, features  # noqa: E402

OUT = ROOT / "results" / "k_checks"
K = 10                                                   # D = 100 ms
DELAYS = [0, 20, 40, 50, 60, 80, 100, 120, 140, 150, 160]  # ms
SHIFT_DELAYS = [0, 50, 150]
WINDOWS = [50, 100, 200]
if __import__("os").environ.get("K_SWEEP") == "1":
    # systematic window sweep (same design, 7 windows); also visible to worker processes
    WINDOWS = [25, 50, 75, 100, 150, 200, 250]
    OUT = ROOT / "results" / "k_checks_sweep"
LAGS = list(range(0, 51, 5))                             # 0..500 ms of history
A_KIN = np.logspace(-2, 4, 7)
A_EMG = np.logspace(0, 6, 7)
GAINS = [0.0, 0.25, 0.5, 1.0]
SHIFTS_S = [5, 7, 9]


class Ridge:
    """Standardised multi-output ridge; alpha selected per output on the validation set."""

    def fit(self, Xtr, Ytr, Xva, Yva, alphas):
        self.mx, self.sx = Xtr.mean(0), Xtr.std(0) + 1e-8
        self.my = Ytr.mean(0)
        Z = (Xtr - self.mx) / self.sx
        lam, V = np.linalg.eigh(Z.T @ Z)
        Vb = V.T @ (Z.T @ (Ytr - self.my))
        Zv = ((Xva - self.mx) / self.sx) @ V
        best = np.full(Ytr.shape[1], np.inf)
        self.W = np.zeros((Z.shape[1], Ytr.shape[1]))
        for a in alphas:
            C = Vb / (lam + a)[:, None]
            err = np.mean((Zv @ C + self.my - Yva) ** 2, axis=0)
            upd = err < best
            best[upd] = err[upd]
            self.W[:, upd] = (V @ C)[:, upd]
        return self

    def predict(self, X):
        return ((X - self.mx) / self.sx) @ self.W + self.my


def _stack(D, ids):
    return np.vstack([D[i] for i in ids])


def run_subject(subject):
    out_csv = OUT / f"{subject}.csv"
    if out_csv.exists():
        return subject, "skipped"
    t0 = time.time()
    F = features(subject)
    EW = {W: emg_feats(subject, W) for W in WINDOWS}
    agg = []
    for cond in ("SJ", "MJ"):
        J = JOINTS[cond]
        nJ = len(J)
        q = {i: F[f"{cond}_{i}_q"] for i in range(10)}
        qd = {i: F[f"{cond}_{i}_qd"] for i in range(10)}
        kinH = {i: add_history(np.hstack([q[i], qd[i]]), LAGS) for i in range(10)}
        preds = {}   # (arm, W, d) -> {trial: (y, yh)}

        def put(key, ti, y, yh):
            preds.setdefault(key, {})[ti] = (y, yh)

        for test, val in FOLDS:
            tr = [i for i in range(10) if i not in test and i != val]
            EH = {}
            for W in WINDOWS:
                St = np.vstack([EW[W][f"{cond}_{i}_emg"] for i in tr])
                mu, sd = np.nanmean(St, 0), np.nanstd(St, 0) + 1e-8
                Zs = {i: np.nan_to_num((EW[W][f"{cond}_{i}_emg"] - mu) / sd) for i in range(10)}
                EH[W] = {i: add_history(Zs[i], LAGS) for i in range(10)}
                if W == 200:
                    for S in SHIFTS_S:
                        EH[f"shift{S}"] = {i: add_history(np.roll(Zs[i], S * 100, axis=0), LAGS)
                                           for i in range(10)}
            # EMG only, absolute target
            for W in WINDOWS:
                Xe = {i: EH[W][i][:len(qd[i]) - K] for i in range(10)}
                Ye = {i: qd[i][K:] for i in range(10)}
                m = Ridge().fit(_stack(Xe, tr), _stack(Ye, tr), Xe[val], Ye[val], A_EMG)
                for ti in test:
                    put(("emg_only", W, 0), ti, Ye[ti], m.predict(Xe[ti]))
            for d_ms in DELAYS:
                d = d_ms // 10
                idx = {i: np.arange(d, len(qd[i]) - K) for i in range(10)}
                Xk = {i: kinH[i][idx[i] - d] for i in range(10)}
                ref = {i: qd[i][idx[i] - d] for i in range(10)}
                y = {i: qd[i][idx[i] + K] for i in range(10)}
                inc = {i: y[i] - ref[i] for i in range(10)}
                kin = Ridge().fit(_stack(Xk, tr), _stack(inc, tr), Xk[val], inc[val], A_KIN)
                kin_val = kin.predict(Xk[val])
                kin_te = {ti: kin.predict(Xk[ti]) for ti in test}
                for ti in test:
                    put(("kin", 0, d_ms), ti, y[ti], ref[ti] + kin_te[ti])
                # leave-one-trial-out residuals within the training trials
                res = {}
                for j in tr:
                    rest = [x for x in tr if x != j]
                    mj = Ridge().fit(_stack(Xk, rest), _stack(inc, rest), Xk[val], inc[val], A_KIN)
                    res[j] = inc[j] - mj.predict(Xk[j])
                res_val = inc[val] - kin_val
                arms = [("fused", W, W) for W in WINDOWS]
                if d_ms in SHIFT_DELAYS:
                    arms += [(f"shift{S}", 200, f"shift{S}") for S in SHIFTS_S]
                for kind, W, key in arms:
                    Xe = {i: EH[key][i][idx[i]] for i in range(10)}
                    em = Ridge().fit(_stack(Xe, tr), _stack(res, tr), Xe[val], res_val, A_EMG)
                    e_val = em.predict(Xe[val])
                    c = np.array([GAINS[int(np.argmin(
                        [np.mean((res_val[:, j] - g * e_val[:, j]) ** 2) for g in GAINS]))]
                        for j in range(nJ)])
                    for ti in test:
                        put((kind, W, d_ms), ti, y[ti], ref[ti] + kin_te[ti] + c * em.predict(Xe[ti]))
        for (kind, W, d_ms), P in preds.items():
            Y = np.vstack([P[i][0] for i in range(10)])
            Yh = np.vstack([P[i][1] for i in range(10)])
            for j, joint in enumerate(J):
                e = Yh[:, j] - Y[:, j]
                agg.append(dict(subject=subject, condition=cond, joint=joint, kind=kind, W=W,
                                delay_ms=d_ms,
                                r2=1 - np.sum(e ** 2) / np.sum((Y[:, j] - Y[:, j].mean()) ** 2),
                                rmse_degs=float(np.degrees(np.sqrt(np.mean(e ** 2))))))
    pd.DataFrame(agg).to_csv(out_csv, index=False)
    return subject, f"{(time.time() - t0) / 60:.1f} min"


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s, msg in ex.map(run_subject, SUBJECTS):
            print(s, msg, flush=True)

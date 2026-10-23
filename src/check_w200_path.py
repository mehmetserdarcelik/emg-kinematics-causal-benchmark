"""Code-path check: reruns the W = 200, D = 0 TCN with the window-sweep code (run_extra logic)
and compares it with the main analysis (run_v2, seed 0) in three participants."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
SUBJ = ["S02", "S09", "S14"]


def one(subject):
    from run_extra import FOLDS, TG, _norm, _rows, deep_fp, features, targets
    F = features(subject)
    rows = []
    for cond in ("SJ", "MJ"):
        preds = {}
        for test, val in FOLDS:
            tr = [i for i in range(10) if i not in test and i != val]
            Xall = {i: F[f"{cond}_{i}_emg"] for i in range(10)}
            Yall = {i: targets(F, cond, i, TG) for i in range(10)}
            Xn, _ = _norm(Xall, tr)
            out = deep_fp("tcn", 0, [Xn[i] for i in tr], [Yall[i] for i in tr],
                          [Xn[val]], [Yall[val]], [Xn[i] for i in test])
            for i, p in zip(test, out):
                preds[i] = p
        Y = np.vstack([targets(F, cond, i, TG) for i in range(10)])
        Yh = np.vstack([preds[i] for i in range(10)])
        rows += _rows(subject, cond, "tcn_w200_sweeppath", Y, Yh, TG)
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1) as ex:
        rows = [r for rs in ex.map(one, SUBJ) for r in rs]
    a = pd.DataFrame(rows)
    m = pd.concat([pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "v2" / f"metrics_{s}.csv")
                   for s in SUBJ])
    m = m[(m.model == "tcn") & (m.seed == 0) & (m.target != "tau")]
    j = a.merge(m, on=["subject", "condition", "joint", "target"], suffixes=("_sweep", "_v2"))
    j["diff"] = j.r2_sweep - j.r2_v2
    out = Path(__file__).resolve().parents[1] / "results" / "extra" / "check_w200_path.csv"
    j.to_csv(out, index=False)
    print(j[["subject", "condition", "joint", "target", "r2_v2", "r2_sweep", "diff"]].round(4).to_string(index=False))

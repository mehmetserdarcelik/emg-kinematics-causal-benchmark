"""Wireless dropout statistics with the detector used in the analysis (>= 20 identical samples).

For each trial-channel: number of runs, total duration (whole runs) and the longest run. Analysed channels only
(SJ: 7 elbow muscles; MJ: 12 muscles). Output: results/extra/dropouts.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import MUSCLES, SJ_MUSCLES, SUBJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MIN_RUN = 20


def one(subject):
    from data import load_subject
    d = load_subject(subject)
    rows = []
    for cond in ("SJ", "MJ"):
        names = SJ_MUSCLES if cond == "SJ" else MUSCLES
        for t in d["trials"][cond]:
            for m in names:
                x = t.emg_raw[:, MUSCLES.index(m)]
                change = np.r_[True, x[1:] != x[:-1]]
                starts = np.flatnonzero(change)
                lens = np.diff(np.r_[starts, len(x)])
                runs = lens[lens >= MIN_RUN]
                if len(runs):
                    rows.append(dict(subject=subject, condition=cond, trial=t.index, muscle=m, n_runs=len(runs),
                                     sec=runs.sum() / 2000, longest_s=runs.max() / 2000))
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as ex:
        rows = [r for rs in ex.map(one, SUBJECTS) for r in rs]
    a = pd.DataFrame(rows)
    a.to_csv(ROOT / "results" / "extra" / "dropouts.csv", index=False)
    print(len(a), "trial-channels;", a.n_runs.sum(), "runs;", round(a.sec.sum(), 1), "s")
    print(a.groupby("condition")[["n_runs", "sec"]].sum())
    print("run length: median", (a.sec / a.n_runs).median(), "longest", a.longest_s.max())

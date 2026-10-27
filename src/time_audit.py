"""Time-stamp audit (Supplementary S1): TIME columns of the EMG (2 kHz) and kinematics/torque (100 Hz) files.

For every trial: start times, uniformity of the sampling intervals, whether every 20th EMG sample coincides with a
kinematic time stamp, the number of additional EMG samples, and whether the angle, velocity, torque and filtered-EMG
files share one time base. Output: results/extra/time_audit.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import DATA_DIR, ROOT, SUBJECTS  # noqa: E402

KIN_FILES = ("humanPositions", "humanVelocities", "muscleTorque", "emgFilt")


def audit(subject):
    rows = []
    for cond in ("SJ", "MJ"):
        for t in sorted((DATA_DIR / subject / cond).iterdir(), key=lambda p: int(p.name[1:])):
            te = pd.read_csv(t / "emgRaw.csv").iloc[:, 0].to_numpy()
            tk = {f: np.abs(pd.read_csv(t / f"{f}.csv").iloc[:, 0].to_numpy()) for f in KIN_FILES}  # some files contain "-0"
            k = tk["humanPositions"]
            rows.append(dict(subject=subject, condition=cond, trial=int(t.name[1:]), n_emg=len(te), n_kin=len(k),
                             extra_emg=len(te) - 20 * len(k), t0_emg=te[0], t0_kin=k[0],
                             dt_emg_max_dev=np.abs(np.diff(te) - 1 / 2000).max(),
                             dt_kin_max_dev=np.abs(np.diff(k) - 1 / 100).max(),
                             emg_vs_kin_time_max=np.abs(te[::20][:len(k)] - k).max(),
                             shared_kin_base=all(len(v) == len(k) and np.allclose(v, k, rtol=0, atol=1e-9)
                                                 for v in tk.values())))
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as ex:
        df = pd.DataFrame([r for rs in ex.map(audit, SUBJECTS) for r in rs])
    df.to_csv(ROOT / "results" / "extra" / "time_audit.csv", index=False)
    print(len(df), "trials; additional EMG samples:", df.extra_emg.value_counts().to_dict(),
          "| start at 0:", bool((df.t0_emg == 0).all() and (df.t0_kin == 0).all()),
          "| largest time difference (s):", df[["dt_emg_max_dev", "dt_kin_max_dev", "emg_vs_kin_time_max"]].max().max(),
          "| shared kinematic time base:", bool(df.shared_kin_base.all()))

"""Equivalence test (TOST, Wilcoxon signed-rank): (EMG + torque) - torque, margin +-0.01 dR2.

For each target two one-sided tests: H0_1: diff <= -delta, H0_2: diff >= +delta. p_TOST = max(p1, p2).
Also a 90 % bootstrap CI (consistent with TOST). Output: results/extra/tost.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from stats import TGT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
DELTA = 0.01
MARGINS = (0.005, 0.01, 0.02)
rng = np.random.default_rng(0)


def main():
    v = pd.read_csv(R / "v2" / "summary_by_subject.csv")
    rows = []
    for c, j, t in TGT:
        w = v[(v.condition == c) & (v.joint == j) & (v.target == t)].pivot_table(
            index="subject", columns="model", values="r2")
        d = (w["emgtau_tcn"] - w["tau_tcn"]).dropna().values
        p_lo = wilcoxon(d + DELTA, alternative="greater").pvalue
        p_hi = wilcoxon(d - DELTA, alternative="less").pvalue
        b = np.median(rng.choice(d, (10000, len(d))), axis=1)
        row = dict(condition=c, joint=j, target=t, n=len(d), median_diff=np.median(d),
                   ci90_lo=np.percentile(b, 5), ci90_hi=np.percentile(b, 95),
                   p_lower=p_lo, p_upper=p_hi, p_tost=max(p_lo, p_hi), delta=DELTA,
                   n_outside=int((np.abs(d) > DELTA).sum()))
        # margin sensitivity: Wilcoxon TOST and a sign-test TOST that does not assume symmetry
        for m in MARGINS:
            row[f"p_tost_{m}"] = max(wilcoxon(d + m, alternative="greater").pvalue,
                                     wilcoxon(d - m, alternative="less").pvalue)
            row[f"p_sign_{m}"] = max(binomtest(int((d > -m).sum()), len(d), alternative="greater").pvalue,
                                     binomtest(int((d < m).sum()), len(d), alternative="greater").pvalue)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(R / "extra" / "tost.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(main().to_string())

"""Is the short-window advantage explained by sensor dropouts? (Section 2.8)

The EMG-only TCN test predictions saved for late fusion (q, qd (t + 100 ms); W = 25, 50, 200 ms) are evaluated on all
samples and on dropout-free samples only (no dropout in any channel during the 200 ms before the input time, which
also covers the longest window), and on samples without dropouts in the whole network context (receptive field plus
window, 2.73 s; the first 2.73 s of each trial excluded; at least 10 s of data required).
Output: results/latefusion/window_dropout.csv and window_dropout_tests.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from data import JOINTS, MUSCLES, ROOT, SJ_MUSCLES, SUBJECTS  # noqa: E402

OUT = ROOT / "results" / "latefusion"
K = 10
ARMS = ["w25", "w50", "w200"]
RF_RAW = (253 + 20) * 20   # TCN receptive field (253 steps) + 200-ms window, in 2-kHz samples
MIN_SAMPLES = 1000         # participant-condition pairs with < 10 s of clean data are omitted


def run_subject(subject):
    from data import load_subject
    from emgfeat import DECIM, _rsum, dropout_mask
    from run_v2 import FOLDS, features, targets
    P = np.load(OUT / f"preds_{subject}.npz")
    F = features(subject)
    d = load_subject(subject)
    rows = []
    w = int(0.2 * 2000)
    for cond in ("SJ", "MJ"):
        J = JOINTS[cond]
        cols = [MUSCLES.index(m) for m in (SJ_MUSCLES if cond == "SJ" else MUSCLES)]
        clean, clean_rf = {}, {}
        for t in d["trials"][cond]:
            m = dropout_mask(t.emg_raw[:, cols])
            b = (_rsum(m.astype(float), w) > 0).any(axis=1)[::DECIM][:len(t.q)]
            b = np.r_[b, np.repeat(b[-1], max(0, len(t.q) - len(b)))]
            clean[t.index] = ~b[:len(t.q) - K]
            # whole receptive field (253 steps) + 200-ms window, dropouts marked over their full length;
            # the first 2.73 s of each trial (incomplete context) are excluded
            mf = dropout_mask(t.emg_raw[:, cols], causal=False)
            br = (_rsum(mf.astype(float), RF_RAW) > 0).any(axis=1)[::DECIM][:len(t.q)]
            br = np.r_[br, np.repeat(br[-1], max(0, len(t.q) - len(br)))]
            br[:RF_RAW // DECIM] = True
            clean_rf[t.index] = ~br[:len(t.q) - K]
        fold_of = {i: f for f, (test, _) in enumerate(FOLDS) for i in test}
        Y = np.vstack([targets(F, cond, i, ["q", "qd"])[K:] for i in range(10)])
        C = np.concatenate([clean[i] for i in range(10)])
        CR = np.concatenate([clean_rf[i] for i in range(10)])
        for arm in ARMS:
            Yh = np.vstack([P[f"{cond}|{arm}|f{fold_of[i]}|t{i}"] for i in range(10)])
            for ti, tg in enumerate(["q", "qd"]):
                for j, joint in enumerate(J):
                    c = ti * len(J) + j
                    for subset, keep in (("all", np.ones(len(C), bool)), ("dropout_free", C),
                                         ("dropout_free_rf", CR)):
                        if keep.sum() < MIN_SAMPLES:
                            continue
                        y, yh = Y[keep, c], Yh[keep, c]
                        rows.append(dict(subject=subject, condition=cond, joint=joint, target=tg, arm=arm,
                                         subset=subset, frac=keep.mean(),
                                         r2=1 - np.sum((yh - y) ** 2) / np.sum((y - y.mean()) ** 2)))
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:   # summary only: from the saved per-participant results (no data needed)
        df = pd.read_csv(OUT / "window_dropout.csv")
    else:
        with ProcessPoolExecutor(max_workers=3) as ex:
            df = pd.DataFrame([r for rs in ex.map(run_subject, SUBJECTS) for r in rs])
        df.to_csv(OUT / "window_dropout.csv", index=False)
    piv = df.pivot_table(index=["subset", "condition", "joint", "target", "subject"], columns="arm", values="r2")
    rows = []
    rng = np.random.default_rng(0)
    fr = df.drop_duplicates(["subject", "condition", "subset"]).set_index(["subset", "condition", "subject"]).frac
    for (sub, c, j, t), s in piv.groupby(level=[0, 1, 2, 3]):
        g = (s["w25"] - s["w200"]).dropna()
        bs = np.median(rng.choice(g.values, (10000, len(g))), axis=1)
        f = fr.loc[(sub, c)]
        rows.append(dict(subset=sub, condition=c, joint=j, target=t, n=len(g), kept_median=f.median(),
                         kept_min=f.min(), r2_w25=s["w25"].median(), r2_w200=s["w200"].median(), gain=g.median(),
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), wins=int((g > 0).sum()),
                         p=wilcoxon(g).pvalue))
    tests = pd.DataFrame(rows)
    tests.to_csv(OUT / "window_dropout_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(tests.round(4).to_string(index=False))
    print("dropout-free fraction (median):", df[df.subset == "dropout_free"].groupby("condition").frac.median().round(3).to_dict())

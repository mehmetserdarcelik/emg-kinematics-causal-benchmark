"""Causal features for other EMG window lengths (W ms), cached.

W = 200 uses the main cache (run_v2.features). Other W are stored in
cache/feats_w/SXX_wW.npz. Keys: "{cond}_{i}_emg".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import MUSCLES, ROOT, SJ_MUSCLES, load_subject  # noqa: E402
from emgfeat import extract  # noqa: E402

DIR = ROOT / "cache" / "feats_w"


def emg_feats(subject: str, W: int) -> dict:
    if W == 200:
        from run_v2 import features
        F = features(subject)
        return {k: v for k, v in F.items() if k.endswith("_emg")}
    path = DIR / f"{subject}_w{W}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d = load_subject(subject)
    out = {}
    for cond in ("SJ", "MJ"):
        cols = [MUSCLES.index(m) for m in (SJ_MUSCLES if cond == "SJ" else MUSCLES)]
        for t in d["trials"][cond]:
            out[f"{cond}_{t.index}_emg"] = extract(t.emg_raw[:, cols], len(t.q), win_ms=W)
    DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from data import SUBJECTS
    jobs = [(s, W) for s in SUBJECTS for W in (50, 100)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(emg_feats, *zip(*jobs)))
    print("ok")

"""Computational cost (Section 3.6): TCN inference and feature extraction on one CPU core.

TCN: one forward pass over the full context (253 samples, 12 channels x 6 features); 50 warm-up + 1000 repetitions.
Features: extract() time for a complete MJ trial (12 channels, ~29.4 s), amortised per 100-Hz output sample
(offline vectorised computation, not a bound for a streaming implementation); 20 repetitions.
Output: results/extra/timing.csv
"""
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

_ = np.ones((256, 256)) @ np.ones((256, 256))  # load MKL's OpenMP before torch (see deep.py)

sys.path.insert(0, str(Path(__file__).parent))
from data import ROOT, load_subject  # noqa: E402
from emgfeat import extract  # noqa: E402


def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or platform.processor()
    except Exception:
        return platform.processor()


def main():
    import torch
    from deep import TCN
    torch.set_num_threads(1)
    net = TCN(72, 4).eval()
    x = torch.randn(1, 253, 72)
    with torch.no_grad():
        for _ in range(50):
            net(x)
        ts = []
        for _ in range(1000):
            t0 = time.perf_counter()
            net(x)
            ts.append((time.perf_counter() - t0) * 1000)
    raw = load_subject("S01")["trials"]["MJ"][0].emg_raw
    n = len(load_subject("S01")["trials"]["MJ"][0].q)
    fs = []
    for _ in range(20):
        t0 = time.perf_counter()
        extract(raw, n)
        fs.append((time.perf_counter() - t0) * 1000 / n)
    rows = [dict(measure="tcn_forward_ms", median=np.median(ts), q25=np.percentile(ts, 25), q75=np.percentile(ts, 75),
                 reps=1000),
            dict(measure="features_ms_per_10ms_block_amortised", median=np.median(fs), q25=np.percentile(fs, 25),
                 q75=np.percentile(fs, 75), reps=20)]
    import pandas as pd
    out = pd.DataFrame(rows).assign(cpu=cpu_name(), torch=torch.__version__, threads=1)
    out.to_csv(ROOT / "results" / "extra" / "timing.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

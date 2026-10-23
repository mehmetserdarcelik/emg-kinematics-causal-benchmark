"""Reproduces the tables, figures and supplementary tables of the paper.

  python reproduce.py          # from the included per-participant results (CPU, minutes; needs data/ for
                               # Fig. 1 and Supplementary S1, see README)
  python reproduce.py --all    # full pipeline from the raw data (GPU, several days)
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA = [["src/download.py"], ["src/data.py"]]
TRAIN = [  # GPU; per-participant outputs are skipped if they already exist
    ["src/run_v2.py", "1"],
    ["src/run_loso_v2.py", "1"],
    ["src/run_extra.py", "loso_seeds"],
    ["src/run_horizon.py"],
    ["src/run_extra.py", "hbase2"],
    ["src/run_extra.py", "hbase3"],
    ["src/run_extra.py", "hbase4"],
    ["src/run_extra.py", "k3b"],
    ["src/run_extra.py", "wsweep"],
    ["src/check_w200_path.py"],
    ["src/run_latefusion.py"],
    ["src/run_latefusion.py", "round2"],
    ["src/run_kintcn.py"],
]
CPU_HEAVY = [  # need data/ and the full prediction files of run_v2
    ["src/k_checks.py"],
    ["src/k_checks.py", {"K_SWEEP": "1"}],
    ["src/analysis_cpu.py"],
    ["src/check_lag.py"],
    ["src/dropouts.py"],
    ["src/latefusion.py"],
    ["src/causal_kin_check.py"],
    ["src/window_dropout_check.py"],
    ["src/filter_evidence.py"],
    ["src/timing.py"],
]
SUMMARY = [  # CPU, from the per-participant results in results/
    ["src/aggregate.py"],
    ["src/stats.py"],
    ["src/k_analysis.py"],
    ["src/k2_breakeven.py"],
    ["src/tost.py"],
    ["src/paper_tables.py"],
    ["src/paper_figures.py"],
    ["src/supplement.py"],
]


def run(step):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    args = [a for a in step if isinstance(a, str)]
    for a in step:
        if isinstance(a, dict):
            env.update(a)
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    steps = DATA + TRAIN + CPU_HEAVY + SUMMARY if "--all" in sys.argv else SUMMARY
    for s in steps:
        run(s)
    print("done: paper/tables.md, paper/figures/, paper/supplementary.md")

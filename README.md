# When does surface EMG add value for decoding joint kinematics?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23590654.svg)](https://doi.org/10.5281/zenodo.23590654)

Code and derived results for the paper

> M. S. Çelik, *When does surface EMG add value for decoding joint kinematics? An offline benchmark with causal EMG decoding on an open upper-limb exoskeleton dataset*, submitted to Computers in Biology and Medicine (2026).

The study decodes elbow and shoulder joint angle and angular velocity from raw surface EMG with a strictly causal pipeline, using the open dataset of Quesada et al. (17 participants, random single- and multi-joint movements against a viscous exoskeleton field; https://doi.org/10.5281/zenodo.11209324). It asks how well EMG decodes the kinematics, whether EMG carries information beyond the joint torque, and when EMG adds value over measured kinematics with a delay.

## Contents

| Path | Content |
|---|---|
| `src/` | Data loading, causal EMG features, models (ridge, LSTM, causal TCN), all analyses, statistics, tables and figures |
| `tests/test_causality.py` | Future-replacement tests of the causal feature extraction (all window lengths) and of the TCN |
| `results/` | Per-participant results of every analysis and the derived summary tables |
| `paper/` | Generated tables (`tables.md`), figures (`figures/`) and supplementary tables (`supplementary.md`) |
| `reproduce.py` | Runs the pipeline (summary stage by default, `--all` for everything) |

The raw dataset is **not** included. `src/download.py` downloads it from Zenodo (≈ 1.4 GB), checks the MD5 checksum and extracts it to `data/`.

## Quick start: reproduce the tables and figures

```bash
pip install -r requirements.txt
python src/download.py      # needed for Fig. 1 and Supplementary S1 only
python reproduce.py
```

This recomputes the summary tables and statistics from the per-participant result files in `results/` (saved outputs of steps that need the raw data or GPU training, such as the window sweep, prediction horizon, filter, timing and dropout analyses, are read as they are) and writes `paper/tables.md`, `paper/figures/Fig1–Fig6` and `paper/supplementary.md` in a few minutes on a CPU. The outputs are identical to those in the paper.

## Full reproduction from the raw data

```bash
python reproduce.py --all
```

This runs the complete pipeline in the order listed in `reproduce.py`. The model training takes several days on a single consumer GPU (the paper used an NVIDIA RTX 2070 SUPER). Per-participant outputs that already exist are skipped, so the pipeline can be interrupted and resumed. Delete the corresponding files in `results/` to recompute them. For long unattended runs, GPU load and temperature can be limited with the environment variables `GPU_DUTY` (e.g. `0.5`), `GPU_TEMP_HIGH` and `GPU_TEMP_LOW` (see `src/deep.py`). Only `results/v2/preds_S14.npz` (used for Fig. 1) is included among the within-subject prediction files; `src/analysis_cpu.py` needs those of all participants, which `src/run_v2.py` regenerates (it reruns a participant if the prediction file is missing). The EMG-only and kinematics-only TCN predictions used for late fusion are included in `results/latefusion/`.

| Script | Paper section | Output |
|---|---|---|
| `data.py`, `emgfeat.py`, `featw.py` | 2.1–2.3 | data cache, causal features |
| `run_v2.py` | 2.4–2.6, 3.1–3.2 | `results/v2/` |
| `run_loso_v2.py`, `run_extra.py loso_seeds` | 2.5, 3.3 | `results/v2_loso/`, `results/extra/loso_seeds_*` |
| `run_horizon.py`, `analysis_cpu.py` | 2.7, 3.4, 3.7 | `results/horizon/`, `results/extra/cpu_*` |
| `k_checks.py` (`K_SWEEP=1` for all windows) | 2.7–2.8, 3.5 | `results/k_checks/`, `results/k_checks_sweep/` |
| `run_extra.py hbase2/hbase3/hbase4/k3b` | 2.7, 3.5 (early fusion) | `results/extra/` |
| `run_latefusion.py` and `run_latefusion.py round2` (GPU), `latefusion.py` | 2.7, 3.5 (late fusion) | `results/latefusion/` |
| `run_kintcn.py` and `run_kintcn.py nonoise` (GPU) | 2.7, 3.5 (ensemble control) | `results/latefusion/kintcn_*.npz`, `kintcn0_*.npz` |
| `causal_kin_check.py` | 2.7, 3.5 (additional causal filtering) | `results/latefusion/causal_kin*.csv` |
| `run_extra.py wsweep`, `check_lag.py`, `check_w200_path.py`, `dropouts.py`, `window_dropout_check.py` | 2.8, 3.6 | `results/extra/`, `results/latefusion/` |
| `filter_evidence.py`, `timing.py`, `time_audit.py` | 2.2, 3.7, Supplementary S1–S2 | `results/extra/` |
| `aggregate.py`, `stats.py`, `k_analysis.py`, `k2_breakeven.py`, `tost.py` | 2.9 | summary tables in `results/` |
| `paper_tables.py`, `paper_figures.py`, `supplement.py` | Tables 1–3, Figs. 1–6, Supplementary | `paper/` |

## Notes on the dataset

The data audit (paper Section 2.2 and Supplementary S1) found issues that are handled in the code:

- **MJ angular velocities** are provided in rad·sample⁻¹ (100 Hz). `src/data.py` multiplies them by 100.
- **MVC files** contain processed envelopes, not raw EMG. MVC normalisation is therefore not used.
- **Wireless sensor dropouts** (runs of identical raw values) are detected online and handled causally (`src/emgfeat.py`).
- **The provided kinematics are very likely zero-phase filtered** (Supplementary S2). Delayed kinematic inputs therefore still contain some information from the future, and the nominal delays should be interpreted with this in mind.

## Licence

Code: MIT. Derived results (`results/`, `paper/`): CC BY-SA 4.0, the licence of the underlying dataset. See `LICENSE`.

## Citation

If you use this code, please cite the paper (details will be added on publication), this archive and the dataset:

M. S. Çelik, *Code and derived results: When does surface EMG add value for decoding joint kinematics?*, Zenodo, 2026. https://doi.org/10.5281/zenodo.23590654 (all versions; the version used for the paper is v1.1.0)


L. Quesada, D. Verdel, O. Bruneau, B. Berret, M.-A. Amorim, N. Vignais, *A dataset for the investigation of upper limb torque prediction from EMG signals*, Zenodo, 2024. https://doi.org/10.5281/zenodo.11209324

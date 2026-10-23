"""Loader for the Quesada et al. dataset (Zenodo 10.5281/zenodo.11209324).

Reading the raw CSV files is slow, so on first use each participant is converted
into a single .npz cache (cache/SXX.npz).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"

MUSCLES = ["DELTAnt", "DELTMed", "DELTPost", "PECT", "LATI", "TRILong",
           "TRILat", "TRIMed", "BICLong", "BICShort", "BRA", "BRD"]
# Abbreviations used in the source publication
MUSCLE_ABBR = dict(zip(MUSCLES, ["DA", "DM", "DP", "PM", "LD", "TLO",
                                 "TLA", "TM", "BL", "BS", "BRA", "BRD"]))
# In the SJ condition only the elbow muscles are analysed (as in the source publication)
SJ_MUSCLES = ["BICLong", "BICShort", "BRD", "BRA", "TRILong", "TRILat", "TRIMed"]

JOINTS = {"SJ": ["elbow_flexion"], "MJ": ["shoulder_elv", "elbow_flexion"]}
SUBJECTS = [f"S{i:02d}" for i in range(1, 18)]
FS_EMG, FS_KIN = 2000, 100


@dataclass
class Trial:
    subject: str
    condition: str          # "SJ" | "MJ"
    index: int
    emg_raw: np.ndarray     # (N*20, 12) 2 kHz
    emg_filt: np.ndarray    # (N, 12) 100 Hz
    q: np.ndarray           # (N, J) rad
    qd: np.ndarray          # (N, J) rad/s
    tau: np.ndarray         # (N, J) N.m


def _read(path: Path, cols: list[str] | None = None) -> np.ndarray:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # trailing comma at line end
    return df[cols].to_numpy(np.float64) if cols else df.iloc[:, 1:].to_numpy(np.float64)


def _build_cache(subject: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / f"{subject}.npz"
    sdir = DATA_DIR / subject
    arrays = {
        "mvc_elbow": _read(sdir / "MVC" / "emgMVCElbow.csv", MUSCLES),
        "mvc_shoulder": _read(sdir / "MVC" / "emgMVCShoulder.csv", MUSCLES),
    }
    for cond in ("SJ", "MJ"):
        for tdir in sorted((sdir / cond).iterdir(), key=lambda p: int(p.name[1:])):
            k = f"{cond}_{tdir.name[1:]}"
            arrays[f"{k}_emg_raw"] = _read(tdir / "emgRaw.csv", MUSCLES).astype(np.float32)
            arrays[f"{k}_emg_filt"] = _read(tdir / "emgFilt.csv", MUSCLES)
            for name, fname in (("q", "humanPositions"), ("qd", "humanVelocities"),
                                ("tau", "muscleTorque")):
                arrays[f"{k}_{name}"] = _read(tdir / f"{fname}.csv", JOINTS[cond])
    np.savez(out, **arrays)
    return out


def load_subject(subject: str) -> dict:
    path = CACHE_DIR / f"{subject}.npz"
    if not path.exists():
        _build_cache(subject)
    z = np.load(path)
    meta = json.loads((DATA_DIR / subject / f"{subject}.json").read_text())
    trials = {"SJ": [], "MJ": []}
    for cond in ("SJ", "MJ"):
        idx = sorted(int(k.split("_")[1]) for k in z.files
                     if k.startswith(cond + "_") and k.endswith("_tau"))
        for i in idx:
            p = f"{cond}_{i}_"
            qd = z[p + "qd"]
            if cond == "MJ":
                # MJ velocity files are in rad/sample (100 Hz): exactly 100 times smaller
                # than d(q)/dt (r = 0.998). Convert to rad/s.
                qd = qd * FS_KIN
            trials[cond].append(Trial(subject, cond, i, z[p + "emg_raw"], z[p + "emg_filt"],
                                      z[p + "q"], qd, z[p + "tau"]))
    return {"meta": meta, "mvc_elbow": z["mvc_elbow"],
            "mvc_shoulder": z["mvc_shoulder"], "trials": trials}


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=17) as ex:
        for p in ex.map(_build_cache, SUBJECTS):
            print("ok", p.name)

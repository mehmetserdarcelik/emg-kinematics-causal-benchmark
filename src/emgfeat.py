"""Multi-feature input matrix from raw EMG (100 Hz output).

v2 (default, causal=True): every step is causal - the feature at time t depends
only on raw EMG in [0, t] and can be computed online.
  - notch + band-pass: single forward pass (sosfilt, state initialised from the first sample)
  - ZC/SSC threshold: from a causal exponential average of the signal power (tau = 5 s)
  - dropout detection: from the moment a value has been held for >= 20 samples (10 ms)
    (no retrospective marking)
  - invalid windows: last valid value; NaN if there is no earlier valid value
    (0 = training mean after normalisation)
v1 (causal=False): zero-phase version used only in the initial exploratory analyses.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from data import FS_EMG, FS_KIN

DECIM = FS_EMG // FS_KIN
_NOTCH = signal.iirnotch(50.0, 30.0, FS_EMG)
_NOTCH_SOS = signal.tf2sos(*_NOTCH)
_BAND = signal.butter(4, [20, 450], btype="bandpass", fs=FS_EMG, output="sos")
_PRE_SOS = np.vstack([_NOTCH_SOS, _BAND])

FEATS = ["MAV", "RMS", "WL", "ZC", "SSC", "DASDV"]
FEAT_VERSION = "v2"


def preprocess(raw: np.ndarray, causal: bool = True) -> np.ndarray:
    x = raw.astype(np.float64)
    if not causal:
        x = signal.filtfilt(*_NOTCH, x, axis=0)
        return signal.sosfiltfilt(_BAND, x, axis=0)
    zi = signal.sosfilt_zi(_PRE_SOS)[:, :, None] * x[0][None, None, :]
    return signal.sosfilt(_PRE_SOS, x, axis=0, zi=zi)[0]


def _rsum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(x, axis=0)
    c[w:] = c[w:] - c[:-w].copy()
    return c


def dropout_mask(raw: np.ndarray, min_run: int = 20, causal: bool = True) -> np.ndarray:
    """Wireless sensor dropout: the same value repeated for >= min_run samples (10 ms).

    In genuine EMG (even at rest) runs of identical consecutive samples never exceed 2-3;
    during a dropout the last value is held for hundreds to thousands of samples.
    causal=True: a sample is flagged once the value has been constant for at least min_run samples.
    """
    N = len(raw)
    eq = np.vstack([np.zeros((1, raw.shape[1]), bool), raw[1:] == raw[:-1]])
    idx = np.arange(N)[:, None]
    last_change = np.maximum.accumulate(np.where(~eq, idx, 0), axis=0)
    run = idx - last_change + 1  # length of the constant run ending at t
    mask = run >= min_run
    if causal:
        return mask
    # v1: flag the whole run (including its start) retrospectively
    full = mask.copy()
    for k in range(raw.shape[1]):
        ends = np.flatnonzero(mask[:, k])
        for e in ends:
            full[last_change[e, k]:e + 1, k] = True
    return full


def _ffill(F: np.ndarray, bad: np.ndarray, causal: bool = True) -> np.ndarray:
    """Replace invalid samples by the last valid value of the channel."""
    F = F.copy()
    for k in range(F.shape[1]):
        b = bad[:, k]
        if not b.any():
            continue
        if b.all():
            F[:, k] = np.nan
            continue
        idx = np.where(~b, np.arange(len(b)), 0)
        np.maximum.accumulate(idx, out=idx)
        first = int(np.argmax(~b))
        filled = F[idx, k]
        if causal:
            filled[:first] = np.nan  # no earlier valid value
        else:
            filled[:first] = F[first, k]
        F[:, k] = filled
    return F


def extract(raw: np.ndarray, n_out: int, win_ms: int = 200,
            feats: list[str] = FEATS, thr_frac: float = 0.02,
            fix_dropouts: bool = True, causal: bool = True) -> np.ndarray:
    """raw: (N, K) at 2 kHz -> (n_out, K * len(feats)), muscle-major column order."""
    raw = raw.astype(np.float64)
    s = preprocess(raw, causal)
    w = int(win_ms * FS_EMG / 1000)
    bad = None
    if fix_dropouts:
        m = dropout_mask(raw, causal=causal)
        # the feature is invalid if the window [t-W, t] contains any dropout
        bad = (_rsum(m.astype(np.float64), w) > 0)[::DECIM][:n_out]
        if causal:
            bad[: w // DECIM] = True  # first window not yet complete
        s = np.where(m, 0.0, s)
    if causal:
        a = 1.0 / (5 * FS_EMG)  # exponential average, tau = 5 s
        p = signal.lfilter([a], [1, -(1 - a)], s ** 2, axis=0)
        thr = thr_frac * np.sqrt(np.maximum(p, 0))[1:]
    else:
        thr = thr_frac * s.std(axis=0)
    ds = np.diff(s, axis=0, prepend=s[:1])
    out = {}
    if "MAV" in feats:
        out["MAV"] = _rsum(np.abs(s), w) / w
    if "RMS" in feats:
        out["RMS"] = np.sqrt(np.maximum(_rsum(s ** 2, w), 0) / w)
    if "WL" in feats:
        out["WL"] = _rsum(np.abs(ds), w) / w
    if "DASDV" in feats:
        out["DASDV"] = np.sqrt(np.maximum(_rsum(ds ** 2, w), 0) / w)
    if "ZC" in feats:
        zc = (s[1:] * s[:-1] < 0) & (np.abs(s[1:] - s[:-1]) >= thr)
        out["ZC"] = _rsum(np.vstack([np.zeros((1, s.shape[1])), zc]).astype(float), w) / w
    if "SSC" in feats:
        a_, b_ = ds[1:-1], ds[2:]
        t_ = thr[1:] if causal else thr
        ssc = (a_ * b_ < 0) & ((np.abs(a_) >= t_) | (np.abs(b_) >= t_))
        pad = np.zeros((1, s.shape[1]))
        out["SSC"] = _rsum(np.vstack([pad, pad, ssc]).astype(float), w) / w
    F = np.stack([out[f] for f in feats], axis=2)[::DECIM][:n_out]  # (n, K, F)
    if len(F) < n_out:
        F = np.concatenate([F, np.repeat(F[-1:], n_out - len(F), axis=0)])
    # MAV/RMS/WL/DASDV on a log scale: the amplitude distribution is skewed
    for i, f in enumerate(feats):
        if f in ("MAV", "RMS", "WL", "DASDV"):
            F[:, :, i] = np.log(F[:, :, i] + 1e-6)
    if bad is not None:
        if len(bad) < n_out:
            bad = np.vstack([bad, np.repeat(bad[-1:], n_out - len(bad), axis=0)])
        for i in range(F.shape[2]):
            F[:, :, i] = _ffill(F[:, :, i], bad, causal)
    return F.reshape(n_out, -1)


def feature_names(muscles: list[str], feats: list[str] = FEATS) -> list[str]:
    return [f"{m}_{f}" for m in muscles for f in feats]

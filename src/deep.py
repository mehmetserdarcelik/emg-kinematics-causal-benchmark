"""Causal temporal models (TCN, LSTM) and the training loop.

Input: feature sequence (T, D) at 100 Hz. Output: targets at every time step (T, O).
Training uses random 6-s chunks; prediction runs causally (without access to the
future) over the complete trial.
"""
from __future__ import annotations

import copy
import os
import subprocess
import time

import numpy as np

# Windows/Anaconda: numpy's MKL and torch ship different copies of libiomp5md.dll.
# If torch is loaded first, the next numpy BLAS call can kill the process with
# "OMP: Error #15". Load MKL's OpenMP before torch.
_ = np.ones((256, 256)) @ np.ones((256, 256))

import torch  # noqa: E402
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)  # limit CPU load

# --- Optional GPU load and temperature limiting (for long unattended runs) -------
# GPU_DUTY: fraction of time the GPU is busy (1.0 = unlimited, 0.5 = ~50 % load).
# GPU_TEMP_HIGH / GPU_TEMP_LOW: pause at this temperature, resume below that one.
GPU_DUTY = float(os.environ.get("GPU_DUTY", "1.0"))
GPU_TEMP_HIGH = int(os.environ.get("GPU_TEMP_HIGH", "72"))
GPU_TEMP_LOW = int(os.environ.get("GPU_TEMP_LOW", "65"))


def gpu_temp() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _throttle(t_start: float, step: int) -> None:
    if DEV.type != "cuda":
        return
    if GPU_DUTY < 1.0:
        torch.cuda.synchronize()
        busy = time.perf_counter() - t_start
        time.sleep(busy * (1 - GPU_DUTY) / GPU_DUTY)
    if step % 20 == 0:
        t = gpu_temp()
        if t is not None and t >= GPU_TEMP_HIGH:
            print(f"[GPU {t} C >= {GPU_TEMP_HIGH} C, waiting to cool down]", flush=True)
            while t is not None and t > GPU_TEMP_LOW:
                time.sleep(15)
                t = gpu_temp()


class CausalConv(nn.Conv1d):
    def __init__(self, cin, cout, k, d):
        super().__init__(cin, cout, k, dilation=d)
        self.pad = (k - 1) * d

    def forward(self, x):
        return super().forward(nn.functional.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, c, k, d, p):
        super().__init__()
        self.net = nn.Sequential(CausalConv(c, c, k, d), nn.GELU(), nn.Dropout(p),
                                 CausalConv(c, c, k, d), nn.GELU(), nn.Dropout(p))

    def forward(self, x):
        return x + self.net(x)


class TCN(nn.Module):
    """6 blocks, k = 3, dilations 1..32 -> receptive field 253 steps (~2.5 s)."""

    def __init__(self, din, dout, c=64, k=3, levels=6, p=0.2):
        super().__init__()
        self.inp = nn.Conv1d(din, c, 1)
        self.blocks = nn.Sequential(*[TCNBlock(c, k, 2 ** i, p) for i in range(levels)])
        self.out = nn.Conv1d(c, dout, 1)
        self.rf = 1 + 2 * (k - 1) * (2 ** levels - 1)

    def forward(self, x):  # x: (B, T, D)
        return self.out(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)


class LSTM(nn.Module):
    def __init__(self, din, dout, h=128, layers=2, p=0.2):
        super().__init__()
        self.lstm = nn.LSTM(din, h, layers, batch_first=True, dropout=p)
        self.out = nn.Linear(h, dout)
        self.rf = 100  # warm-up: the first 1 s is ignored in the loss

    def forward(self, x):
        return self.out(self.lstm(x)[0])


MODELS = {"tcn": TCN, "lstm": LSTM}


def _predict(model, Xs):
    model.eval()
    with torch.no_grad():
        return [model(torch.as_tensor(X[None], dtype=torch.float32, device=DEV))[0].cpu().numpy()
                for X in Xs]


def train(kind: str, Xtr: list, Ytr: list, Xva: list, Yva: list, seed: int = 0,
          chunk: int = 600, batch: int = 64, steps_per_epoch: int = 50,
          max_epochs: int = 60, patience: int = 8, lr: float = 2e-3,
          noise: float = 0.1, verbose: bool = False):
    """Xtr/Ytr: lists of trials (T, D)/(T, O), already normalised."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MODELS[kind](Xtr[0].shape[1], Ytr[0].shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    warm = min(model.rf, chunk // 2)
    lens = np.array([len(x) for x in Xtr])
    p_trial = lens / lens.sum()
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        for step in range(steps_per_epoch):
            t_step = time.perf_counter()
            ti = rng.choice(len(Xtr), size=batch, p=p_trial)
            st = [rng.integers(0, lens[t] - chunk) for t in ti]
            xb = np.stack([Xtr[t][s:s + chunk] for t, s in zip(ti, st)])
            yb = np.stack([Ytr[t][s:s + chunk] for t, s in zip(ti, st)])
            xb = torch.as_tensor(xb, dtype=torch.float32, device=DEV)
            yb = torch.as_tensor(yb, dtype=torch.float32, device=DEV)
            xb = xb + noise * torch.randn_like(xb)
            loss = nn.functional.mse_loss(model(xb)[:, warm:], yb[:, warm:])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            _throttle(t_step, step)
        pv = _predict(model, Xva)
        vl = float(np.mean([np.mean((p - y) ** 2) for p, y in zip(pv, Yva)]))
        sched.step(vl)
        if verbose:
            print(f"ep {ep:3d} train {loss.item():.4f} val {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict(model, Xs: list) -> list:
    return _predict(model, Xs)

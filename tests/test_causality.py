"""Future-replacement test of the causal feature extraction (Section 2.3).

The raw EMG after a time t* is replaced by random noise; every feature computed for
samples before t* must be unchanged. Synthetic EMG with simulated dropouts is used, so
the test runs without the dataset. Run with: python -m pytest tests  (or python tests/test_causality.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emgfeat import DECIM, extract  # noqa: E402

WINDOWS = [25, 50, 75, 100, 150, 200, 250]


def _synthetic(n=2000 * 12, k=7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)) * np.exp(np.sin(np.arange(n) / 3000.0))[:, None] * 50
    x = np.round(x / 0.1007) * 0.1007                      # amplitude resolution of the sensors
    x[4000:4600, 2] = x[3999, 2]                            # long dropout (held value)
    for s in range(9000, 20000, 700):                       # brief dropouts
        x[s:s + 30, 5] = x[s - 1, 5]
    return x


def test_future_replacement():
    raw = _synthetic()
    n_out = len(raw) // DECIM
    rng = np.random.default_rng(1)
    for W in WINDOWS:
        F = extract(raw, n_out, win_ms=W)
        for t_star in (5000, 11111, 17000):
            alt = raw.copy()
            alt[t_star:] = rng.standard_normal(alt[t_star:].shape) * 80
            G = extract(alt, n_out, win_ms=W)
            k = t_star // DECIM                             # feature samples entirely before t*
            assert np.allclose(np.nan_to_num(F[:k], nan=1e9), np.nan_to_num(G[:k], nan=1e9),
                               rtol=0, atol=1e-9), f"W={W}, t*={t_star}: features depend on the future"


def test_tcn_is_causal():
    """End-to-end check of the network: perturbing future inputs must not change earlier outputs."""
    import torch
    from deep import TCN
    torch.manual_seed(0)
    net = TCN(42, 2).eval()
    x = torch.randn(1, 600, 42)
    for t_star in (100, 253, 400):
        y = x.clone()
        y[:, t_star:] = torch.randn_like(y[:, t_star:]) * 5
        with torch.no_grad():
            a, b = net(x), net(y)
        assert torch.allclose(a[:, :t_star], b[:, :t_star], atol=1e-6), f"TCN output depends on the future (t*={t_star})"
        assert not torch.allclose(a[:, t_star:], b[:, t_star:]), "perturbation had no effect"


if __name__ == "__main__":
    test_future_replacement()
    print("causality test passed for W =", WINDOWS)
    test_tcn_is_causal()
    print("TCN end-to-end causality test passed")

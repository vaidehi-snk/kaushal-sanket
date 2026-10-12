import numpy as np
from .models import MODELS, M


def mase(actual, fc, train):
    scale = np.mean(np.abs(train[M:] - train[:-M]))
    return np.nan if scale == 0 else float(np.mean(np.abs(actual - fc)) / scale)


def min_history(h=3, n_origins=3, min_train=24):
    return min_train + h * n_origins


def rolling_backtest(y, h=3, n_origins=3, min_train=24):
    """Rolling-origin backtest. Returns None if history is too short (=> caller must abstain)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < min_history(h, n_origins, min_train):
        return None
    res = {name: {"mase": [], "err": []} for name in MODELS}
    for k in range(n_origins, 0, -1):
        split = n - k * h
        train, test = y[:split], y[split:split + h]
        for name, fn in MODELS.items():
            fc = fn(train, h)
            res[name]["mase"].append(mase(test, fc, train))
            res[name]["err"].extend((test - fc).tolist())
    return {
        name: {
            "mase": float(np.nanmean(r["mase"])),
            "rmse": float(np.sqrt(np.mean(np.square(r["err"])))),
        }
        for name, r in res.items()
    }

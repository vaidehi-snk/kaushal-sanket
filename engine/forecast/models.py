import warnings
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

M = 12  # monthly seasonality


def seasonal_naive(y, h):
    y = np.asarray(y, float)
    last = y[-M:]
    return np.array([last[i % M] for i in range(h)])


def ets(y, h):
    y = np.asarray(y, float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal="add",
                seasonal_periods=M, initialization_method="estimated",
            ).fit()
        return np.clip(fit.forecast(h), 0, None)
    except Exception:
        return seasonal_naive(y, h)


MODELS = {"snaive": seasonal_naive, "ets": ets}
# TODO: LightGBM global model (Tweedie) + MinTrace reconciliation (hierarchicalforecast)

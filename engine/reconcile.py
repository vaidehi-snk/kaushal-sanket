"""Hierarchical forecast reconciliation (MinT with shrinkage), numpy only.
Grouped hierarchy: geography (national > state > district) x product (sector > trade). Every series is forecast on its own,
then reconciled so all levels add up exactly. Intervals come from the reconciled error covariance (shrunk residual correlations)."""
import warnings
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from engine.forecast.models import seasonal_naive, M
from engine.forecast.backtest import rolling_backtest

GROUPS = {"nat_all": [], "state_all": ["state"], "district_all": ["state", "district"], "nat_trade": ["trade"],
          "state_trade": ["state", "trade"], "nat_sector": ["sector"], "state_sector": ["state", "sector"],
          "district_sector": ["state", "district", "sector"]}


def build_S(bottom):
    """bottom: dicts with state/district/sector/trade. Returns S (series x bottom), meta, alias[(level,key)] -> row of S.
    Aggregates with the same members as an existing row are aliased to it (keeps W invertible)."""
    nb = len(bottom); rows, meta, seen, alias = [], [], {}, {}
    for i in range(nb):
        r = np.zeros(nb); r[i] = 1; seen[frozenset([i])] = len(rows); rows.append(r); meta.append("district_trade")
    for lv, keys in GROUPS.items():
        g = {}
        for i, b in enumerate(bottom): g.setdefault(tuple(b[k] for k in keys), []).append(i)
        for k, idx in g.items():
            m = frozenset(idx)
            if m not in seen:
                r = np.zeros(nb); r[idx] = 1; seen[m] = len(rows); rows.append(r); meta.append(lv)
            alias[(lv, k)] = seen[m]
    return np.array(rows), meta, alias


def fit_one(y, H):
    """Backtest-selected model (seasonal naive vs ETS), H-step forecast, one-step residuals."""
    y = np.asarray(y, float); bt = rolling_backtest(y)
    if bt is None: return None
    best = min(bt, key=lambda k: bt[k]["mase"])
    if best == "ets":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=M,
                                         initialization_method="estimated").fit()
            fc, res = np.clip(f.forecast(H), 0, None), np.asarray(f.resid)[M:]
        except Exception:
            best = "snaive"
    if best == "snaive":
        fc, res = seasonal_naive(y, H), y[M:] - y[:-M]
    return dict(fc=fc, resid=res, rmse=bt[best]["rmse"])


def shrunk_corr(R):
    """Schafer-Strimmer shrinkage of the residual correlation matrix toward the identity."""
    T, n = R.shape; Rc = R - R.mean(0); sd = Rc.std(0, ddof=1); sd = np.where(sd > 1e-9, sd, 1.0); Z = Rc / sd
    C = Z.T @ Z / (T - 1); wbar = Z.T @ Z / T
    var = T / ((T - 1) ** 3) * (np.einsum("ki,kj->ij", Z ** 2, Z ** 2) - T * wbar ** 2)
    off = ~np.eye(n, dtype=bool)
    lam = float(np.clip(var[off].sum() / max((C[off] ** 2).sum(), 1e-12), 0, 1))
    Rs = (1 - lam) * C + lam * np.eye(n); np.fill_diagonal(Rs, 1.0)
    return sd, Rs, lam


def reconcile(Y, S, H=6, hold=0, method="wls"):
    """Y: (series x T), rows aligned with S. Uses the first T-hold months.
    method: 'wls' (structural weights, robust, default), 'mint' (shrunk residual covariance) or 'ols'.
    Returns reconciled F, base forecasts, 3-month-sum sd, and F for every method (for honest comparison)."""
    n = S.shape[0]; T = Y.shape[1] - hold
    fits = [fit_one(Y[i, :T], H) for i in range(n)]
    if any(f is None for f in fits): raise ValueError("a series is too short to forecast")
    F0 = np.array([f["fc"] for f in fits]); R = np.column_stack([f["resid"] for f in fits])
    sd, Rs, lam = shrunk_corr(R); I = np.eye(n)
    Ws = {"mint": Rs * np.outer(sd, sd) + 1e-8 * I, "wls": np.diag(S.sum(1).astype(float)), "ols": I}

    def proj(W):
        Wi = np.linalg.inv(W)
        return np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)

    Gs = {m: proj(W) for m, W in Ws.items()}
    by = {m: S @ np.clip(G @ F0, 0, None) for m, G in Gs.items()}   # clip at zero, then re-sum so levels still add up
    sd3 = np.array([f["rmse"] for f in fits]) * np.sqrt(3); Wh = Rs * np.outer(sd3, sd3)   # interval covariance keeps shrunk correlations
    SG = S @ Gs[method]
    return dict(F=by[method], base=F0, by_method=by, sd3=np.sqrt(np.einsum("ij,jk,ik->i", SG, Wh, SG)), lam=lam)

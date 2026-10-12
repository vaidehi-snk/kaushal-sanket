"""Gap engine. Flags shortage/saturation ONLY when the whole uncertainty interval agrees,
otherwise says so. No hand-weighted index is called 'AI' here."""
import numpy as np
import pandas as pd
from engine.schema import KEY
from engine.forecast.models import MODELS
from engine.forecast.backtest import rolling_backtest, min_history

Z80 = 1.2816
THRESH = 0.15       # |gap_index| needed to call shortage/saturation
MASE_CUTOFF = 1.5   # forecast worse than this => don't trust it


def _row(key, **kw):
    base = dict(zip(KEY, key))
    base.update(kw)
    return base


def build_gaps(df: pd.DataFrame, h: int = 3):
    gaps, bts = [], []
    for key, g in df.groupby(KEY):
        g = g.sort_values("period")
        y = g["demand_signal"].to_numpy(float)
        n = len(y)
        bt = rolling_backtest(y, h=h)
        if bt is None:
            gaps.append(_row(key, status="insufficient_evidence", months=n,
                             reason=f"only {n} months, need {min_history(h)}"))
            continue
        best = min(bt, key=lambda m: bt[m]["mase"])
        bts.append(_row(key, months=n, best_model=best,
                        **{f"mase_{m}": v["mase"] for m, v in bt.items()},
                        **{f"rmse_{m}": v["rmse"] for m, v in bt.items()}))
        if bt[best]["mase"] > MASE_CUTOFF:
            gaps.append(_row(key, status="insufficient_evidence", months=n,
                             reason=f"backtest MASE {bt[best]['mase']:.2f} > {MASE_CUTOFF}"))
            continue
        fc = MODELS[best](y, h)
        d = fc.sum()
        sig = bt[best]["rmse"] * np.sqrt(h)          # assumes independent monthly errors
        supply = g["supply_certified"].tail(6).mean() * h
        gi = lambda dd: (dd - supply) / (dd + supply) if (dd + supply) > 0 else 0.0
        mid, lo, hi = gi(d), gi(max(d - Z80 * sig, 0)), gi(d + Z80 * sig)
        if lo > THRESH:
            status = "shortage"
        elif hi < -THRESH:
            status = "saturation"
        elif abs(mid) > THRESH:
            status = "uncertain"   # point estimate says gap, interval does not support it
        else:
            status = "balanced"
        recent = y[-h:].sum()
        emerging = status == "shortage" and recent > 0 and d / recent > 1.25
        gaps.append(_row(key, status=status, months=n, model=best,
                         forecast_demand=round(d, 1), demand_lo80=round(max(d - Z80 * sig, 0), 1),
                         demand_hi80=round(d + Z80 * sig, 1), supply_3m=round(supply, 1),
                         gap_index=round(mid, 3), gap_lo80=round(lo, 3), gap_hi80=round(hi, 3),
                         emerging=bool(emerging), severity=round(mid * np.sqrt(d), 2),
                         backtest_mase=round(bt[best]["mase"], 3)))
    gaps = pd.DataFrame(gaps)
    if "severity" in gaps:
        gaps["rank"] = gaps["severity"].abs().rank(ascending=False, method="first")
    return gaps, pd.DataFrame(bts)


def coverage(df: pd.DataFrame, h: int = 3) -> pd.DataFrame:
    need = min_history(h)
    s = df.groupby(KEY + ["source"]).size().reset_index(name="months")
    s["sufficient"] = s["months"] >= need
    return (s.groupby(["state", "district"])
              .agg(series=("months", "size"), sufficient=("sufficient", "sum"),
                   sources=("source", lambda x: ",".join(sorted(set(x)))))
              .reset_index())

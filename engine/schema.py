"""Common schema. EVERY adapter (NCS, PLFS, dummy, evaluation data) must return this."""
import pandas as pd

COLUMNS = [
    "period",            # month start, datetime
    "state", "district",
    "nco_code",          # NCO-2015 code, string (4-digit preferred, 3/2 allowed)
    "sector",
    "demand_signal",     # monthly openings / postings attributed to this series
    "supply_seats",      # training seats sanctioned that month
    "supply_certified",  # candidates certified that month
    "source",            # where the row came from (ncs / plfs / dummy ...)
    "confidence",        # 0-1, coding / source confidence
]
KEY = ["state", "district", "nco_code", "sector"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Adapter output missing columns: {missing}")
    out = df[COLUMNS].copy()
    out["period"] = pd.to_datetime(out["period"]).dt.to_period("M").dt.to_timestamp()
    out["nco_code"] = out["nco_code"].astype(str)
    for c in ["demand_signal", "supply_seats", "supply_certified"]:
        if (out[c] < 0).any():
            raise ValueError(f"Negative values in {c}")
    out = out.sort_values(KEY + ["period"]).reset_index(drop=True)
    if out.duplicated(KEY + ["period"]).any():
        raise ValueError("Duplicate (series, period) rows")
    return out

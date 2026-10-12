"""Synthetic generator for development ONLY. Always labelled source='dummy'.
Planted patterns (so we can check the engine finds them):
  Solar      -> growing demand, thin supply   => shortage
  Retail     -> flat demand, heavy supply     => saturation
  Automotive -> demand ~ supply               => balanced
  Two series have short history               => must abstain
NCO codes are illustrative; verify against the official NCO-2015 list.
"""
import numpy as np
import pandas as pd
from engine.schema import validate

DISTRICTS = ["Pune", "Nashik", "Nagpur", "Thane", "Kolhapur", "Chh. Sambhajinagar"]
TRADES = {  # sector: (nco, base demand, monthly growth, supply/demand ratio)
    "Automotive Repair": ("7231", 120, 0.002, 0.95),
    "Solar":             ("7411", 40, 0.025, 0.40),
    "Retail":            ("5223", 300, 0.000, 1.60),
}
SPARSE = {("Kolhapur", "Solar"): 14, ("Nagpur", "Solar"): 18}  # months of history only


class DummyAdapter:
    def __init__(self, seed: int = 7, months: int = 36):
        self.seed, self.months = seed, months

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        periods = pd.date_range("2023-01-01", periods=self.months, freq="MS")
        t = np.arange(self.months)
        rows = []
        for d in DISTRICTS:
            scale = rng.uniform(0.4, 1.6)
            for sector, (nco, base, growth, ratio) in TRADES.items():
                season = 1 + 0.15 * np.sin(2 * np.pi * (t % 12) / 12 + rng.uniform(0, 6))
                demand = base * scale * (1 + growth * t) * season * rng.normal(1, 0.06, self.months)
                certified = base * scale * ratio * rng.normal(1, 0.05, self.months)
                df = pd.DataFrame({
                    "period": periods, "state": "Maharashtra", "district": d,
                    "nco_code": nco, "sector": sector,
                    "demand_signal": demand.round().clip(0),
                    "supply_certified": certified.round().clip(0),
                    "source": "dummy", "confidence": 0.9,
                })
                df["supply_seats"] = (df["supply_certified"] / 0.8).round()
                keep = SPARSE.get((d, sector))
                rows.append(df.tail(keep) if keep else df)
        return validate(pd.concat(rows, ignore_index=True))

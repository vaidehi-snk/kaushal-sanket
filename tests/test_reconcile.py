import json
from pathlib import Path
import numpy as np
from engine.reconcile import build_S, reconcile


def test_reconciled_forecasts_add_up():
    rng = np.random.default_rng(0)
    b = [dict(state=s, district=d, sector=sec, trade=t) for s, d in [("A", "a1"), ("A", "a2"), ("B", "b1")] for sec, t in [("X", "x1"), ("X", "x2"), ("Y", "y1")]]
    S, meta, alias = build_S(b)
    Y = S @ (50 + rng.random((len(b), 36)) * 10 + np.sin(np.arange(36) / 2) * 5)
    r = reconcile(Y, S, H=3)
    assert np.allclose(r["F"], S @ r["F"][:len(b)]) and (r["F"] >= 0).all()


def test_published_results_are_coherent():
    f = Path(__file__).resolve().parent.parent / "data" / "processed" / "results.json"
    if not f.exists(): return
    rows = json.loads(f.read_text(encoding="utf-8"))["rows"]
    tot = [r for r in rows if r["level"] == "nat_all"][0]["forecast_demand"]
    assert abs(tot - sum(r["forecast_demand"] for r in rows if r["level"] == "state_all")) < 0.5

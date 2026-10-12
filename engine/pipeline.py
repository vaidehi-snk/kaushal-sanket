from pathlib import Path
from engine.adapters.dummy import DummyAdapter
from engine.gap.score import build_gaps, coverage

OUT = Path(__file__).resolve().parent.parent / "data" / "processed"


def run(adapter=None, write=True):
    adapter = adapter or DummyAdapter()
    df = adapter.load()
    gaps, bt = build_gaps(df)
    cov = coverage(df)
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        gaps.to_csv(OUT / "gaps.csv", index=False)
        bt.to_csv(OUT / "backtest.csv", index=False)
        cov.to_csv(OUT / "coverage.csv", index=False)
    return df, gaps, bt, cov


if __name__ == "__main__":
    _, gaps, bt, cov = run()
    cols = [c for c in ["district", "sector", "status", "gap_index", "gap_lo80", "gap_hi80", "reason"] if c in gaps]
    print(gaps[cols].to_string(index=False))
    print("\nBacktest (MASE, lower is better):")
    print(bt[["district", "sector", "mase_snaive", "mase_ets", "best_model"]].round(2).to_string(index=False))

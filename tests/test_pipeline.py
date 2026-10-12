import pandas as pd
import pytest
from engine.adapters.dummy import DummyAdapter
from engine.schema import validate, COLUMNS
from engine.pipeline import run
from engine.coding.nco_coder import NCOCoder


def test_schema_rejects_missing_column():
    df = DummyAdapter().load().drop(columns=["confidence"])
    with pytest.raises(ValueError):
        validate(df)


def test_dummy_matches_schema():
    assert list(DummyAdapter().load().columns) == COLUMNS


def test_engine_abstains_on_short_history_and_finds_planted_patterns():
    _, gaps, _, _ = run(write=False)
    short = gaps[(gaps.district == "Kolhapur") & (gaps.sector == "Solar")].iloc[0]
    assert short.status == "insufficient_evidence"
    assert (gaps[gaps.sector == "Solar"].dropna(subset=["gap_index"]).status == "shortage").all()
    assert (gaps[gaps.sector == "Retail"].status == "saturation").all()


def test_shortage_requires_whole_interval_above_threshold():
    _, gaps, _, _ = run(write=False)
    s = gaps[gaps.status == "shortage"]
    assert (s.gap_lo80 > 0.15).all()


def test_nco_coder_abstains_on_gibberish():
    assert NCOCoder().predict("qqzzxx vvkk")["level"] == 0

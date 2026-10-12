from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Query
from engine.pipeline import run, OUT
from engine.coding.nco_coder import NCOCoder, embedding_coder

app = FastAPI(title="SIH26246 Labour Market Intelligence Engine")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])   # local demo: lets index.html call the API
_coder = NCOCoder()


def _load(name):
    p = OUT / f"{name}.csv"
    if not p.exists():
        run()
    df = pd.read_csv(p, dtype={"nco_code": str})
    return df.astype(object).where(df.notna(), None)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/gaps")
def gaps(district: str | None = None, sector: str | None = None, status: str | None = None):
    df = _load("gaps")
    for col, val in [("district", district), ("sector", sector), ("status", status)]:
        if val:
            df = df[df[col] == val]
    return df.to_dict(orient="records")


@app.get("/backtest")
def backtest():
    return _load("backtest").to_dict(orient="records")


@app.get("/coverage")
def coverage():
    return _load("coverage").to_dict(orient="records")


_emb = None


@app.get("/nco/code")
def nco_code(text: str = Query(..., min_length=2), backend: str = "tfidf"):
    global _emb
    if backend == "embed":
        if _emb is None:
            _emb = embedding_coder()   # downloads the multilingual model on first use
        return _emb.predict(text)
    return _coder.predict(text)


@app.get("/nco/eval")
def nco_eval():
    return _coder.evaluate()


# ---- v2: results from engine.v2 (demand index, hierarchy, targets, rebalance) ----
import json, csv, io
from fastapi.responses import Response
from engine.v2 import build as _build_v2

_RES = OUT / "results.json"


def _v2():
    if not _RES.exists():
        _build_v2()
    return json.loads(_RES.read_text(encoding="utf-8"))


@app.get("/v2/meta")
def v2_meta():
    r = _v2(); return {"meta": r["meta"], "proof": r["proof"]}


@app.get("/v2/index")
def v2_index():
    return _v2()["index"]


@app.get("/v2/rows")
def v2_rows(level: str | None = None, state: str | None = None, district: str | None = None,
            trade: str | None = None, sector: str | None = None, status: str | None = None):
    q = dict(level=level, state=state, district=district, trade=trade, sector=sector, status=status)
    return [r for r in _v2()["rows"] if all(v is None or r.get(k) == v for k, v in q.items())]


@app.get("/v2/rebalance")
def v2_rebalance():
    return _v2()["moves"]


@app.get("/v2/export/targets.csv")
def v2_targets_csv():
    cols = ["state", "district", "sector", "trade", "course", "status", "warn", "months_to", "seats_now", "target", "change", "action"]
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(cols)
    for r in _v2()["rows"]:
        if r["level"] == "district_trade" and r.get("target") is not None:
            w.writerow([r.get(c, "") for c in cols])
    return Response(buf.getvalue(), media_type="text/csv")


@app.get("/v2/export/targets.parquet")
def v2_targets_parquet():
    _v2()
    return Response((OUT / "seat_targets.parquet").read_bytes(), media_type="application/octet-stream")

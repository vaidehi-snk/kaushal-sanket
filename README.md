# SIH26246 - Labour Market Intelligence System (prototype), Team Verisonic
Open `index.html` in a browser. No server, no install: data is embedded. (Hosting: drag the file onto Vercel/Netlify.)

## Regenerate results (engine)
```
pip install -r requirements.txt
python -m engine.v2                  # writes data/processed/results.json + docs/METHODOLOGY.md
python build_plfs_supply.py "path\2025.csv"   # real PLFS supply aggregates (Maharashtra)
python -m uvicorn api.main:app --reload        # API docs at /docs  (/v2/rows, /v2/index, /v2/rebalance, /v2/export/targets.csv, /v2/export/targets.parquet; engine also writes data/processed/seat_targets.parquet)
python -m pytest -q
```
In the portal use "Load results JSON" to load a new results.json. Schedule `python -m engine.v2` (Task Scheduler / cron) for periodic refresh.

## Honest status
All demand/supply/forecast data is SYNTHETIC until NCS and scheme data are connected (swap `build_data()` in engine/v2.py for adapters).
PLFS 2025 aggregates are real. Hindi/Marathi labels are drafts. NCO codes and course names are illustrative. Centres are illustrative.

## Occupation coder
`python -m engine.coding.nco_coder` prints leave-one-language-out accuracy on data/nco_labelled.csv (EN/HI/MR).
`python eval_embeddings.py` (needs `pip install sentence-transformers`) runs the same test with a multilingual embedding model. Not yet run by us.

## Embedding coder (measured: EN 78%, HI 86%, MR 80% exact, leave-one-language-out, 49 titles each)
1. `pip install sentence-transformers`   2. `python calibrate_embeddings.py` (writes data/emb_thresholds.json)
3. `python -m uvicorn api.main:app --reload`  (on Windows plain `uvicorn` is often not on PATH)   4. open index.html, Occupation coder tab (uses the API automatically, falls back to the browser matcher)
- Reconciliation: every level is forecast, then reconciled so districts sum to states and states to the total (engine/reconcile.py; WLS default, MinT and OLS compared on a held-out test).

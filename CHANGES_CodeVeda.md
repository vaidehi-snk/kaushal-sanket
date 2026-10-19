# Changes applied for the SIH finals push — Team CodeVeda

These are drop-in replacements for the files of the same name in your repo.
Copy them over the originals (back up first if you like):

- `engine/v2.py`
- `index.html`
- `README.md`
- `data/TEMPLATE.csv`  ← new file

---

## 1. Evaluation-data ingest (the #1 blocker, now fixed)

The engine no longer needs code edits to run on the evaluators' dummy data.

```
python -m engine.v2 --template              # writes data/TEMPLATE.csv — the exact columns an input file needs
python -m engine.v2 --csv path/to/their_data.csv   # runs the whole engine on a real / evaluation file
python -m engine.v2                          # unchanged: built-in synthetic data
```

`--csv` reads a file with the same columns as `data/TEMPLATE.csv`, validates that
every column is present (clear error if not), and runs the full pipeline into
`data/processed/results.json`. Verified: running on `data/TEMPLATE.csv` reproduces
the full 130-row result (54 called, 6 abstained). On demo day you point `--csv` at
their file, reload the portal, done — no live code editing.

**Columns the input file needs** (one row per series per month):
`period, state, district, sector, trade, nco_code, course, kind, placements,
seats, enrolled, certified, quality, job_portal, industry_hiring, ncs_vacancies,
eshram_registrations`. `ncs_vacancies` may be blank where a district has no NCS
feed. `data/TEMPLATE.csv` is a complete, runnable example.

## 2. Team name → CodeVeda, college name removed

- Portal footer: now "by Team CodeVeda." (was "Team Verisonic, APCOER Pune").
- `README.md` title: "Team CodeVeda".
- No college name anywhere.

## 3. "National" wording (matches the problem statement)

The top level now reads **"National (pilot scope)"** instead of "All pilot states",
in the Explore header, the breadcrumb, the gap-formula note, and the rollup label —
so it lands the PS phrase "drill-down from national to district level", while the
note still states plainly that national = Maharashtra + Karnataka only.

---

## Still worth doing before the finals (not done here — they touch the live
## portal's JS, so do them yourself in the editor where you can test each step)

1. **Run-to-run diff** — keep the previous `results.json` and show a "what changed
   since last run" strip (flipped calls, new early warnings, seat targets that
   moved). This is the single most persuasive thing you could add: it makes the
   "updated periodically rather than annually" claim visible. ~half a day.
2. **Course review queue** — one table of trades where placement < 50%, the QP is
   deactivated, or demand is rising fast. Pure assembly from fields the engine
   already computes (`placement_rate`, `qp_status`, `emerging`, `nsqf`).
3. **Severity rank column** in the Explore table — you already compute `rank` and
   `severity`; just surface it so the wording matches "rank by severity".
4. **One real demand feed**, however small (NCS public counts, a state seat file) —
   even one district on real numbers moves "Data access" from Medium to High.
5. **Decision-log persistence** (`localStorage`) so it survives a reload, and the
   **scale number** (time one run, note the reconciliation is the bottleneck).

Replace the placeholder URLs on slide 6 (`kaushal-sanket.vercel.app`,
`github.com/vaidehi-snk/sih26246`) with your real ones before submitting.

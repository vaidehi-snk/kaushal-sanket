"""Engine v2. Multi-source demand index (learned weights), national->district roll-ups, early warning,
seat targets, seat-neutral rebalancing, lead-time validation. ALL DATA HERE IS SYNTHETIC; swap build_data() for real adapters."""
import json, math, warnings, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from engine.forecast.models import MODELS, ets
from engine.forecast.backtest import rolling_backtest, min_history
from engine.reconcile import GROUPS, build_S, reconcile

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
DOCS = Path(__file__).resolve().parent.parent / "docs"
Z80, THR, MASE_CUT, H, COMP = 1.2816, 0.15, 1.5, 6, 0.8
SRC = ["job_portal", "industry_hiring", "ncs_vacancies", "eshram_registrations"]
FACT = {"white": [1.6, 1.0, 0.7, 0.4], "blue": [0.5, 0.8, 1.0, 1.3]}  # how much each source over/under-counts
SD = [0.10, 0.07, 0.09, 0.15]
STATES = {"Maharashtra": ["Pune", "Nashik", "Nagpur", "Thane", "Kolhapur", "Chhatrapati Sambhajinagar"],
          "Karnataka": ["Bengaluru Urban", "Mysuru", "Belagavi", "Dharwad"]}
TRADES = {  # trade: (sector, NCO-2015, course, kind, base, growth, supply/current demand). Codes from data/qp_register.csv (NSDC QP files)
    "Automotive Body Repair Technician": ("Automotive", "7213.0301", "ASC/Q1405 Automotive Body Repair Technician, NSQF L3", "blue", 120, .002, .95),
    "Two-wheeler Service Technician": ("Automotive", "7231.0501", "ASC/Q1411 Two Wheeler Service Technician, NSQF L4", "blue", 90, .030, .78),
    "Solar PV Installer": ("Renewable & Electrical", "7421.1401", "SGJ/Q0101 Solar PV Installer (Suryamitra), NSQF L4", "blue", 40, .025, .40),
    "AC Field Technician": ("Renewable & Electrical", "7421.0401", "ELE/Q3102 Field Technician - Air Conditioner, NSQF L4", "blue", 110, .002, 1.45),
    "Retail Sales Assistant": ("Retail", "5223.0105", "RAS/Q0103 Retail Sales Assistant, NSQF L3", "white", 300, 0, 1.6),
    "Retail Cashier": ("Retail", "4211.0301", "RAS/Q0102 Retail Cashier, NSQF L3", "white", 150, .001, 1.3)}
QP = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "qp_register.csv", dtype=str).fillna("").set_index("trade").to_dict("index")
# Funnel realism (separate random stream so demand, forecasts and validation are unchanged). Planted cases for the levers:
LOW_FILL = {("Thane", "Solar PV Installer"): .52, ("Bengaluru Urban", "Solar PV Installer"): .58, ("Mysuru", "Retail Cashier"): .50}
LOW_PLACE = {("Pune", "Automotive Body Repair Technician"): .42, ("Belagavi", "Automotive Body Repair Technician"): .45}
SHORT = {("Kolhapur", "Solar PV Installer"): 14, ("Dharwad", "Solar PV Installer"): 18}
NO_NCS = {"Belagavi", "Mysuru", "Nagpur"}   # districts with no NCS feed (coverage gap)
LEVELS = GROUPS   # includes nat_all (all pilot states) for reconciliation
TYPES = ["ITI", "PMKVY training centre", "Sector skill partner centre"]


def build_data(seed=11, months=36):
    rng = np.random.default_rng(seed); rf = np.random.default_rng(seed + 1000); t = np.arange(months)
    per = pd.date_range("2023-01-01", periods=months, freq="MS"); parts = []
    for st, ds in STATES.items():
        for d in ds:
            sc = rng.uniform(.5, 1.6) * (1.5 if d in ("Bengaluru Urban", "Pune", "Thane") else 1)
            for tr, (sec, nco, course, kind, base, g, ratio) in TRADES.items():
                season = 1 + .15 * np.sin(2 * np.pi * (t % 12) / 12 + rng.uniform(0, 6))
                hires = base * sc * (1 + g * t) * season * rng.normal(1, .06, months)
                cert = base * sc * (1 + g * (months - 1)) * ratio * rng.normal(1, .05, months)
                sig = {s: hires * FACT[kind][i] * rng.normal(1, SD[i], months) for i, s in enumerate(SRC)}
                if d in NO_NCS: sig["ncs_vacancies"] = np.full(months, np.nan)
                fill = LOW_FILL.get((d, tr), rf.uniform(.82, .97)); qual = LOW_PLACE.get((d, tr), rf.uniform(.86, 1.0))
                df = pd.DataFrame({"period": per, "state": st, "district": d, "sector": sec, "trade": tr, "nco_code": nco,
                                   "course": course, "kind": kind, "placements": hires * rng.normal(1, .04, months),
                                   "seats": cert / .87 / fill, "enrolled": cert / .87, "certified": cert, "quality": qual, **sig})
                parts.append(df.tail(SHORT.get((d, tr), months)))
    return pd.concat(parts, ignore_index=True)


def mad_z(x):
    m = np.nanmedian(x); s = 1.4826 * np.nanmedian(np.abs(x - m)); s = s if s > 0 else 1.0
    return (x - m) / s, m, s


def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def learn_index(df, seed=3):
    blocks = []
    for _, g in df.groupby(["district", "trade"]):
        if g[SRC].isna().any().any(): continue
        z = np.column_stack([mad_z(g[s].to_numpy())[0] for s in SRC]); yz = mad_z(g.placements.to_numpy())[0]
        blocks.append((z[:-1], yz[1:]))   # signals at t predict placements at t+1
    fit = lambda B, cols: Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(
        np.vstack([b[0][:, cols] for b in B]), np.concatenate([b[1] for b in B]))
    X, y = np.vstack([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks]); allc = list(range(4))
    m = fit(blocks, allc); coef = m.coef_; base_r2 = r2(y, X @ coef)
    rng = np.random.default_rng(seed); boots = []
    for _ in range(40):
        c = fit([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))], allc).coef_; boots.append(c / max(c.sum(), 1e-9))
    loo = {}
    for i, s in enumerate(SRC):
        cols = [j for j in allc if j != i]; cc = fit(blocks, cols).coef_
        loo[s] = base_r2 - r2(y, X[:, cols] @ cc)
    w = coef / coef.sum()
    bias = {s: {sec: float(np.nanmedian(df.loc[df.sector == sec, s] / df.loc[df.sector == sec, "placements"]))
                for sec in df.sector.unique()} for s in SRC}
    cov = {s: float(df.groupby(["district", "trade"])[s].apply(lambda x: x.notna().any()).mean()) for s in SRC}
    return coef, dict(sources=SRC, weights=dict(zip(SRC, w)), weights_sd=dict(zip(SRC, np.std(boots, 0))),
                      loo_r2_drop=loo, r2=base_r2, n_series=len(blocks), bias=bias, coverage=cov)


def demand_series(g, coef):
    z = np.column_stack([mad_z(g[s].to_numpy())[0] for s in SRC]); ok = ~np.isnan(z)
    tot = (ok * coef).sum(1)
    idx = np.where(ok, z, 0).dot(coef) * coef.sum() / np.where(tot > 0, tot, 1)   # renormalise when a source is missing
    _, m, s = mad_z(g.placements.to_numpy())
    return np.clip(m + s * idx, 0, None)


def assess(f, s, sig, recent3):
    d = f[:3].sum(); S = 3 * s; sg = np.sqrt((sig[:3] ** 2).sum())
    gi = lambda x: (x - S) / (x + S) if x + S > 0 else 0.0
    mid, lo, hi = gi(d), gi(max(d - Z80 * sg, 0)), gi(d + Z80 * sg)
    st = "shortage" if lo > THR else "saturation" if hi < -THR else "uncertain" if abs(mid) > THR else "balanced"
    gm = (f - s) / np.maximum(f + s, 1e-9)
    ms = next((i + 1 for i, x in enumerate(gm) if x > THR), None); mt = next((i + 1 for i, x in enumerate(gm) if x < -THR), None)
    warn = mo = None
    if st in ("balanced", "uncertain"):
        if ms: warn, mo = "approaching_shortage", ms
        elif mt: warn, mo = "approaching_saturation", mt
    # Probability that the gap clears the threshold (normal demand error, supply as measured). The call rule (whole 80%
    # range beyond the threshold) is the same as "at least 90% likely", so likelihood words stay consistent with calls.
    ph = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p_sh = 1 - ph((S * (1 + THR) / (1 - THR) - d) / max(sg, 1e-9)); p_sa = ph((S * (1 - THR) / (1 + THR) - d) / max(sg, 1e-9))
    side, p = ("shortage", p_sh) if p_sh >= p_sa else ("saturation", p_sa)
    return dict(status=st, warn=warn, months_to=mo, emerging=bool(st == "shortage" and recent3 > 0 and d / recent3 > 1.25),
                p_shortage=p_sh, p_saturation=p_sa, likely_side=side, likelihood=p, likelihood_word=ipcc(p),
                gap_index=mid, gap_lo80=lo, gap_hi80=hi, forecast_demand=d, demand_lo80=max(d - Z80 * sg, 0),
                demand_hi80=d + Z80 * sg, supply_3m=S, severity=mid * np.sqrt(max(d, 0)))


def ipcc(p):
    """IPCC AR5 calibrated likelihood language."""
    return ("virtually certain" if p >= .99 else "very likely" if p >= .9 else "likely" if p >= .66
            else "about as likely as not" if p >= .33 else "unlikely")


def lever(r):
    """What to do, not just where: reads the supply funnel (seat fill, placement of certified trainees)."""
    F = r["funnel"]; fill = F["enrolled"] / max(F["seats"], 1); place = F["placed"] / max(F["certified"], 1)
    st, chg = r["status"], r.get("change", 0)
    if place < .5 and st != "saturation":
        return "review_course", fill, place, (f"Only {place:.0%} of certified trainees are placed. Review the course (QP/NSQF content, employer tie-ups) "
                                              f"with the Sector Skill Council before adding seats")
    if st == "shortage" and fill < .7:
        return "fix_uptake", fill, place, (f"Do not add seats yet: only {fill:.0%} of seats are filled. Fill them first "
                                           f"(mobilisation, stipend, counselling)")
    if st == "shortage":
        return "add_capacity", fill, place, f"Add capacity: seats are {fill:.0%} filled and {place:.0%} of certified trainees are placed"
    if st == "saturation" and fill < .7:
        return "shrink", fill, place, f"Shrink or merge batches: only {fill:.0%} of seats are filled"
    if st == "saturation":
        return "redirect", fill, place, f"Redirect seats to a trade in shortage in this district (seats are {fill:.0%} filled)"
    return None, fill, place, ""


def seat_plan(a, now):
    c = 12 / COMP / 3; tgt, lo, hi = a["forecast_demand"] * c, a["demand_lo80"] * c, a["demand_hi80"] * c
    chg = tgt - now if a["status"] in ("shortage", "saturation") else 0.0
    if a["status"] == "shortage": txt = f"Add about {chg:.0f} seats a year (80% range {lo - now:.0f} to {hi - now:.0f})"
    elif a["status"] == "saturation": txt = f"Cut about {-chg:.0f} seats a year (80% range {now - hi:.0f} to {now - lo:.0f})"
    elif a["warn"]: txt = f"No change yet. {a['warn'].replace('_', ' ')} in about {a['months_to']} month(s)"
    elif a["status"] == "uncertain": txt = "Hold. The evidence does not support a change"
    else: txt = "No change"
    return dict(seats_now=now, target=tgt, target_lo=lo, target_hi=hi, change=chg, action=txt)


def grp(rows, keys):
    g = {}
    for r in rows: g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g


def sig_from_sd3(sd3):
    rm = sd3 / np.sqrt(3)
    return np.array([rm * np.sqrt(max(i + 1, 3) / 3) for i in range(H)])


def finalize(r):
    a = assess(r["f"], r["s"], r["sig"], r["recent3"]); p = seat_plan(a, r["now"]); r.update(a); r.update(p)
    lv, fill, place, why = lever(r); r.update(lever=lv, lever_text=why, seat_fill=fill, placement_rate=place)
    if lv == "fix_uptake": r["change"] = 0.0
    if lv in ("fix_uptake", "review_course", "shrink"): r["action"] = why + (". " + r["action"] if lv == "shrink" else "")
    q = QP.get(r["trade"], {}); r.update(qp_code=q.get("qp_code"), qp_status=q.get("status"), qp_deactivated_on=q.get("deactivated_on") or None,
                                        qp_source=q.get("source"))
    if q.get("status") == "deactivated":
        r["action"] = (f"QP {q['qp_code']} was deactivated on {q['deactivated_on']}: run no new batches on it and move these seats to an "
                       f"active QP ({q['successor_hint']}). " + r["action"])
    r["centres"] = [{"name": f"{r['district']} {chr(65 + i)}", "type": TYPES[i], "seats": round(r["now"] * r["sh"][i]),
                     "change": round(r["change"] * r["sh"][i])} for i in range(3)]   # after the lever, so "fill seats first" means no change

    r["nsqf"] = ("Review the NSQF curriculum: demand is rising fast" if a["emerging"]
                 else "Review intake or retarget the course" if a["status"] == "saturation" else "")


def reconcile_called(base):
    called = [r for r in base if "f" in r]; S, meta, alias = build_S(called)
    L = min(len(r["y"]) for r in called); Y = S @ np.array([r["y"][-L:] for r in called])
    res = reconcile(Y, S, H); res["alias"] = alias
    for i, r in enumerate(called):
        r["f"] = res["F"][i]; r["sig"] = sig_from_sd3(res["sd3"][i]); finalize(r)
    return res


def reconciliation_proof(base, hold=3):
    """Hold out the last months, forecast every level with and without reconciliation, compare methods honestly."""
    called = [r for r in base if "f" in r]; S, meta, _ = build_S(called)
    L = min(len(r["y"]) for r in called); Y = S @ np.array([r["y"][-L:] for r in called])
    res = reconcile(Y, S, H=hold, hold=hold); act = Y[:, -hold:]
    eb = np.sqrt(((res["base"] - act) ** 2).mean(1))
    cls = np.array(["bottom" if m == "district_trade" else "top" if m in ("nat_all", "state_all") else "middle" for m in meta])
    out = dict(hold_months=hold, series=int(len(meta)), method="wls", compare={})
    for m, F in res["by_method"].items():
        er = np.sqrt(((F - act) ** 2).mean(1))
        out["compare"][m] = {c: dict(n=int((cls == c).sum()), improved_share=float(np.mean(er[cls == c] < eb[cls == c])),
                                     rmse_change=float(1 - np.mean(er[cls == c] / np.maximum(eb[cls == c], 1e-9)))) for c in ("bottom", "middle", "top")}
    out.update(out["compare"]["wls"])
    return out


def base_rows(df, coef):
    rng = np.random.default_rng(5); out = []; types = ["ITI", "PMKVY training centre", "Sector skill partner centre"]
    for (st, d, tr), g in df.groupby(["state", "district", "trade"], sort=False):
        g = g.sort_values("period"); y = demand_series(g, coef); n = len(y)
        m = dict(level="district_trade", state=st, district=d, sector=g.sector.iat[0], trade=tr, course=g.course.iat[0],
                 nco=g.nco_code.iat[0], kind=g.kind.iat[0], months=n)
        bt = rolling_backtest(y)
        if bt is None: out.append({**m, "status": "insufficient_evidence", "reason": f"only {n} months, need {min_history()}"}); continue
        best = min(bt, key=lambda k: bt[k]["mase"])
        if bt[best]["mase"] > MASE_CUT: out.append({**m, "status": "insufficient_evidence", "reason": f"backtest MASE {bt[best]['mase']:.2f} above {MASE_CUT}"}); continue
        f = MODELS[best](y, H); rm = bt[best]["rmse"]; sig = np.array([rm * np.sqrt(max(i + 1, 3) / 3) for i in range(H)])
        s = g.certified.tail(6).mean(); now = g.seats.tail(12).sum(); r3 = y[-3:].sum()
        fun = dict(seats=now, enrolled=g.enrolled.tail(12).sum(), certified=g.certified.tail(12).sum(), placed=(np.minimum(g.certified.tail(12), g.placements.tail(12)) * g.quality.iat[0]).sum())
        row = {**m, "model": best, "mase": bt[best]["mase"], "mase_snaive": bt["snaive"]["mase"], "f": f, "sig": sig, "s": s, "recent3": r3,
               "now": now, "hist": y[-18:], "funnel": fun, "y": y, "sh": rng.dirichlet([4, 3, 2])}
        finalize(row); out.append(row)
    return out


def aggregate(base, rec):
    called = [r for r in base if "f" in r]; out = []
    for lv, keys in LEVELS.items():
        tot = {k: len(v) for k, v in grp(base, keys).items()}
        for k, rs in grp(called, keys).items():
            i = rec["alias"][(lv, k)]; f = rec["F"][i]; sig = sig_from_sd3(rec["sd3"][i])   # reconciled, not summed
            s = sum(r["s"] for r in rs); now = sum(r["now"] for r in rs); a = assess(f, s, sig, sum(r["recent3"] for r in rs))
            row = {**dict(level=lv, state="All", district="All", sector="All", trade="All"), **dict(zip(keys, k)),
                   **dict(covered=len(rs), total=tot[k]), **a, **seat_plan(a, now), **dict(f=f, sig=sig, hist=sum(r["hist"] for r in rs), funnel={k: sum(r["funnel"][k] for r in rs) for k in ("seats", "enrolled", "certified", "placed")})}
            out.append(row)
    return out


def rebalance(base):
    out = []
    for (st, d), rs in grp([r for r in base if "f" in r and r["kind"] == "blue"], ["state", "district"]).items():
        sur = sorted([[r, r["seats_now"] - r["target"]] for r in rs if r["status"] == "saturation" and r["seats_now"] > r["target"]], key=lambda x: -x[1])
        need = sorted([[r, r["target"] - r["seats_now"]] for r in rs if r["status"] == "shortage"], key=lambda x: -x[1])
        for nr in need:
            for sr in sur:
                q = min(sr[1], nr[1])
                if q >= 5:
                    out.append(dict(state=st, district=d, source_trade=sr[0]["trade"], target_trade=nr[0]["trade"], seats=round(q)))
                    sr[1] -= q; nr[1] -= q
    return out


def lead_time(df, coef, lag=6):
    hit = miss = fa = n = 0
    lab = lambda g: "shortage" if g > THR else "saturation" if g < -THR else "balanced"
    for _, g in df.groupby(["district", "trade"]):
        g = g.sort_values("period"); y = demand_series(g, coef); N = len(y)
        if N < 33: continue
        o = N - lag; s0 = g.certified.iloc[o - 6:o].mean(); s1 = g.certified.iloc[N - 6:].mean()
        fc = ets(y[:o], lag)[-3:]; pred = lab(np.mean((fc - s0) / (fc + s0))); act = lab(np.mean((y[-3:] - s1) / (y[-3:] + s1)))
        n += 1
        if act != "balanced": hit += pred == act; miss += pred != act
        if pred != "balanced": fa += pred != act
    return dict(series=n, lag_months=lag, recall=hit / max(hit + miss, 1), false_alarm_rate=fa / max(hit + fa, 1))


def pub(r):
    o = {k: v for k, v in r.items() if k not in ("f", "sig", "s", "recent3", "now", "y", "sh")}
    if "f" in r: o["fc"] = r["f"]; o["band"] = Z80 * r["sig"]
    return o


def rnd(o):
    if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [rnd(v) for v in o]
    if isinstance(o, np.ndarray): return [rnd(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else round(float(o), 3)
    if isinstance(o, (np.integer, np.bool_)): return o.item()
    return o


def build(seed=11, write=True):
    df = build_data(seed); coef, idx = learn_index(df); base = base_rows(df, coef); rec = reconcile_called(base); agg = aggregate(base, rec)
    called = [r for r in base if "f" in r]; mv = rebalance(base); lt = lead_time(df, coef)
    proof = dict(series_total=len(base), called=len(called), abstained=len(base) - len(called),
                 median_mase=float(np.median([r["mase"] for r in called])),
                 share_beating_seasonal_naive=float(np.mean([r["mase"] < r["mase_snaive"] for r in called])), lead_time=lt, reconciliation=reconciliation_proof(base))
    meta = dict(run_id=f"run-{dt.date.today():%Y%m%d}-s{seed}", generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), data="SYNTHETIC",
                params=dict(threshold=THR, interval=0.8, completion_rate=COMP, horizon_months=H, min_history=min_history()))
    res = rnd(dict(meta=meta, index=idx, proof=proof, moves=mv, rows=[pub(r) for r in base + agg]))
    if write:
        OUT.mkdir(parents=True, exist_ok=True); DOCS.mkdir(exist_ok=True)
        (OUT / "results.json").write_text(json.dumps(res), encoding="utf-8")
        cols = ["state", "district", "sector", "trade", "course", "status", "warn", "months_to", "seats_now", "target", "change", "action"]
        pd.DataFrame([{c: r.get(c) for c in cols} for r in res["rows"] if r["level"] == "district_trade" and r.get("target") is not None]
                     ).to_parquet(OUT / "seat_targets.parquet", index=False)
        w, sd, lo = idx["weights"], idx["weights_sd"], idx["loo_r2_drop"]
        (DOCS / "METHODOLOGY.md").write_text(f"""# Demand index and forecasting method (run {meta['run_id']}, SYNTHETIC data)
1. **Sources**: {', '.join(SRC)}. Each is normalised per trade and district with median and MAD (robust to outliers).
2. **Learned weights**: non-negative ridge regression predicts next month's placements from this month's four normalised signals, pooled over {idx['n_series']} complete series (R2 {idx['r2']:.2f}).
   Weights: {', '.join(f'{s} {w[s]*100:.0f}% (+/-{sd[s]*100:.0f})' for s in SRC)}. Leaving a source out costs R2: {', '.join(f'{s} {lo[s]:.3f}' for s in SRC)}.
3. **Missing source**: weights of the remaining sources are renormalised for that series.
4. **Scale**: the index is expressed in hires per month using the series' own placements median and MAD.
5. **Forecast**: seasonal-naive vs ETS, chosen per series by rolling-origin backtest (MASE). 6-month horizon, 80% range from backtest RMSE (independence assumed, an approximation).
6. **Call rule**: shortage or oversupply only if the whole 80% range clears +/-{THR}. History under {min_history()} months or MASE above {MASE_CUT}: no call.
7. **Early warning**: central forecast crosses +/-{THR} within 6 months while today's call is balanced or uncertain.
8. **Seat target**: 12 x forecast monthly demand / completion rate ({COMP}). Change = target - seats now, only when a call is made.
9. **Reconciliation**: every level (national, state, district, sector, trade and their combinations) is forecast on its own, then reconciled (structural weighted least squares; MinT with shrinkage and OLS are compared on a held-out test) so all levels add up exactly. Intervals use the reconciled error covariance with shrunk residual correlations. Series that abstained are left out of roll-ups, never imputed.
10. **Validation**: lead-time test, {lt['lag_months']} months ahead, recall {lt['recall']:.2f}, false-alarm rate {lt['false_alarm_rate']:.2f} on {lt['series']} series (synthetic, so a plumbing check, not an accuracy claim).
11. **Likelihood words**: P(gap beyond +/-{THR}) from the forecast error, reported with IPCC AR5 wording (likely >= 66%, very likely >= 90%, virtually certain >= 99%). The call rule (whole 80% range beyond the threshold) equals "at least 90% likely", so words and calls agree.
12. **Levers (what to do, not only where)**: from the supply funnel. Placement of certified trainees under 50% -> review the course with the SSC before adding seats. Shortage with seats under 70% filled -> fill empty seats first (no seat increase). Shortage otherwise -> add capacity. Oversupply with seats under 70% filled -> shrink or merge; otherwise -> redirect seats to a trade in shortage.
13. **QP register**: every trade maps to an NSQF Qualification Pack read from the NSDC QP file (data/qp_register.csv: code, NCO-2015, level, status). A deactivated QP is flagged in every recommendation: no new batches on it, move seats to an active QP.
""", encoding="utf-8")
    return res


if __name__ == "__main__":
    r = build(); rows = r["rows"]
    print(pd.Series([x["status"] for x in rows if x["level"] == "district_trade"]).value_counts().to_dict())
    print("warns:", [(x["district"], x["trade"], x["warn"], x["months_to"]) for x in rows if x.get("warn") and x["level"] == "district_trade"])
    print("moves:", r["moves"][:4], len(r["moves"])); print("weights:", r["index"]["weights"], "r2", r["index"]["r2"])
    print("proof:", r["proof"]); print("rows:", len(rows))

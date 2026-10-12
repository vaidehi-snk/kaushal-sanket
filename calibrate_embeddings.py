"""Run on YOUR machine after eval_embeddings.py works:  python calibrate_embeddings.py
Chooses the confidence threshold honestly: for each language, pick it from the OTHER two languages (precision >= TARGET),
then test on the held-out one. Writes data/emb_thresholds.json (threshold from all three pooled) and checks junk text."""
import json
import numpy as np
from engine.coding.nco_coder import NCOCoder, load_labelled, embedding_encoder, DATA

TARGET = 0.90


def preds(enc):
    lab = load_labelled(); out = {}
    for lang in lab.lang.unique():
        tr, te = lab[lab.lang != lang], lab[lab.lang == lang]
        m = NCOCoder(labelled=tr, extra=False, encoder=enc, th=(9, 9, 9))
        P = [m.predict(t) for t in te.text]
        out[lang] = (np.array([p["confidence"] for p in P]), np.array([p["top_k"][0]["code"] == c for p, c in zip(P, te.code)]))
    return out


def pick(conf, ok):
    for t in np.arange(0.30, 0.99, 0.01):
        s = conf >= t
        if s.sum() >= 10 and ok[s].mean() >= TARGET: return round(float(t), 2)
    return 0.99


def run(enc):
    P = preds(enc); res = {}
    for L in P:
        c = np.concatenate([P[k][0] for k in P if k != L]); o = np.concatenate([P[k][1] for k in P if k != L])
        t = pick(c, o); s = P[L][0] >= t
        res[L] = dict(threshold=t, coverage=round(float(s.mean()), 2), precision=round(float(P[L][1][s].mean()), 2) if s.any() else None)
    c = np.concatenate([v[0] for v in P.values()]); o = np.concatenate([v[1] for v in P.values()])
    return res, pick(c, o)


if __name__ == "__main__":
    enc = embedding_encoder(); res, t = run(enc)
    print("Honest test (threshold chosen on the other two languages):")
    for k, v in res.items(): print(" ", k, v)
    json.dump({"call": t}, open(DATA / "emb_thresholds.json", "w")); print("Saved call threshold:", t)
    m = NCOCoder(encoder=enc, th=(t, t - 0.05, t - 0.10))
    for q in ["qqzz xxvv", "asdf lkjh", "the quick brown fox", "lorem ipsum dolor", "plumber", "नळकामगार", "solar panel installer"]:
        r = m.predict(q); print(f"  {q!r:28} conf {r['confidence']:.2f} -> {r['code'] or 'NO CALL'}")

"""NCO occupation coder (baseline): char n-gram TF-IDF nearest title, with hierarchy back-off and abstention.
Labelled data: data/nco_labelled.csv (team-labelled, EN/HI/MR). Pilot-trade entries from data/nco_lexicon.csv are
marked verified=no (codes not yet checked against the official NCO-2015 list).
Thresholds are fixed a priori, NOT tuned on the test set. Swap the encoder for multilingual embeddings for real cross-language transfer."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).resolve().parent.parent.parent / "data"
HI, MID, LO = 0.55, 0.35, 0.30   # LO raised from 0.20 after gibberish scored 0.24 (checked on a sanity string, not the test set)


def load_labelled(path=DATA / "nco_labelled.csv"):
    return pd.read_csv(path, dtype=str)


def load_pilot(path=DATA / "nco_pilot.csv"):
    """Pilot-trade titles (EN/HI/MR) with NCO-2015 codes as reported from the DGE list. Marked 'pending' until spot-checked."""
    if not Path(path).exists(): return pd.DataFrame(columns=["text", "code", "lang", "verified"])
    vf = DATA / "nco_verified_codes.txt"
    ok = {l.strip() for l in open(vf, encoding="utf-8") if l.strip() and not l.startswith("#")} if vf.exists() else set()
    df = pd.read_csv(path, dtype=str)
    return df.assign(verified=np.where(df.code.isin(ok), "yes", "pending"))


def load_lexicon(path=DATA / "nco_lexicon.csv"):
    rows = []
    for _, r in pd.read_csv(path, dtype=str).iterrows():
        for p in [r["title"]] + str(r["aliases"]).split("|"):
            rows.append((p.strip(), r["code"], "en"))
    return pd.DataFrame(rows, columns=["text", "code", "lang"])


def embedding_encoder(name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name)
    return lambda xs: m.encode(list(xs), normalize_embeddings=True)


def embedding_coder():
    """Embedding-backed coder. Call threshold comes from data/emb_thresholds.json (written by calibrate_embeddings.py)."""
    f = DATA / "emb_thresholds.json"
    t = json.load(open(f))["call"] if f.exists() else 0.80
    return NCOCoder(encoder=embedding_encoder(), th=(t, t - 0.05, t - 0.10))   # narrow back-off: junk text scored 0.58-0.69 vs a call threshold of 0.81


class NCOCoder:
    def __init__(self, labelled=None, extra=True, encoder=None, th=(HI, MID, LO)):
        lab = load_labelled() if labelled is None else labelled
        parts = [lab.assign(verified="yes")]
        if extra:
            pil, lex = load_pilot(), load_lexicon()
            drop = set(pil.code.str[:4]) | {"5230"}   # pilot titles supersede our earlier guessed codes for these groups
            parts += [pil, lex[~lex.code.isin(drop)].assign(verified="no")]
        self.df = pd.concat(parts, ignore_index=True)
        self.enc, self.th = encoder, th   # encoder: list[str] -> L2-normalised array (e.g. a multilingual sentence model)
        if encoder:
            self.X = np.asarray(encoder(self.df.text.str.lower().tolist()))
        else:
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self.X = self.vec.fit_transform(self.df.text.str.lower())

    def predict(self, text, k=3):
        if self.enc: sims = (self.X @ np.asarray(self.enc([text.lower()]))[0]).ravel()
        else: sims = (self.X @ self.vec.transform([text.lower()]).T).toarray().ravel()
        best = {}
        for i in np.argsort(-sims)[:80]:
            best.setdefault(self.df.code.iat[i], (float(sims[i]), self.df.text.iat[i], self.df.verified.iat[i]))
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        code, (conf, near, ver) = ranked[0]
        hi, mid, lo = self.th
        out, level = (code, 6) if conf >= hi else (code[:4], 4) if conf >= mid else (code[:3], 3) if conf >= lo else ("", 0)
        return {"code": out, "level": level, "confidence": round(conf, 3), "nearest": near, "verified": ver == "yes", "status": ver,
                "top_k": [{"code": c, "score": round(v[0], 3)} for c, v in ranked]}

    def evaluate(self, path=DATA / "nco_labelled.csv"):
        return evaluate_lolo(path)


def evaluate_lolo(path=DATA / "nco_labelled.csv", encoder=None, th=(HI, MID, LO)):
    """Leave-one-language-out: train on two languages, test on the third. The strictest honest test with this data."""
    lab = load_labelled(path); res = {}
    for lang in lab.lang.unique():
        tr, te = lab[lab.lang != lang], lab[lab.lang == lang]
        m = NCOCoder(labelled=tr, extra=False, encoder=encoder, th=th)
        P = [m.predict(t) for t in te.text]; truth = te.code.tolist(); top = [p["top_k"][0]["code"] for p in P]
        called = [(p, c) for p, c in zip(P, truth) if p["level"] > 0]
        res[lang] = dict(n=len(te), exact_title_in_training=float(te.text.isin(tr.text).mean()),
                         top1_full=float(np.mean([a == b for a, b in zip(top, truth)])),
                         top1_full_excluding_exact=float(np.mean([a == b for a, b, x in zip(top, truth, ~te.text.isin(tr.text).values) if x])),
                         top1_unit_group=float(np.mean([a[:4] == b[:4] for a, b in zip(top, truth)])),
                         top1_3digit=float(np.mean([a[:3] == b[:3] for a, b in zip(top, truth)])),
                         coverage=len(called) / len(te),
                         accuracy_when_called=float(np.mean([c.startswith(p["code"]) for p, c in called])) if called else None)
    return res


if __name__ == "__main__":
    for k, v in evaluate_lolo().items(): print(k, {a: round(b, 2) if isinstance(b, float) else b for a, b in v.items()})
    c = NCOCoder()
    for q in ["plumber", "प्लंबर", "नळकामगार", "solar panel installer", "qqzz xxvv"]: print(q, "->", c.predict(q)["code"], c.predict(q)["confidence"])

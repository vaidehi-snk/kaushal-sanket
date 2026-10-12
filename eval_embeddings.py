"""Run on YOUR machine:  python eval_embeddings.py   (leave-one-language-out, general titles and pilot-trade titles)"""
from sentence_transformers import SentenceTransformer
from engine.coding.nco_coder import evaluate_lolo, DATA
m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
enc = lambda xs: m.encode(list(xs), normalize_embeddings=True)
for name in ["nco_labelled.csv", "nco_pilot.csv"]:
    print("==", name)
    for k, v in evaluate_lolo(DATA / name, encoder=enc, th=(0.81, 0.76, 0.71)).items():
        print(k, {a: round(b, 2) if isinstance(b, float) else b for a, b in v.items()})

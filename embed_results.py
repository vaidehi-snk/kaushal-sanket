"""Copy data/processed/results.json into index.html (the portal stays one offline file).
Run after `python -m engine.v2`:  python embed_results.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
html = (ROOT / "index.html").read_text(encoding="utf-8")
res = (ROOT / "data" / "processed" / "results.json").read_text(encoding="utf-8")
start = html.index("let RES=") + len("let RES=")
_, end = json.JSONDecoder().raw_decode(html, start)   # find where the old JSON ends, whatever follows it
(ROOT / "index.html").write_text(html[:start] + json.dumps(json.loads(res), ensure_ascii=False) + html[end:], encoding="utf-8")
print(f"embedded {len(res) // 1024} KB of results into index.html")

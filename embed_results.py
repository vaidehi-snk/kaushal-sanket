"""Copy data/processed/results.json (and the district map) into index.html, so the portal stays one offline file.
Run after `python -m engine.v2`:  python embed_results.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
html = (ROOT / "index.html").read_text(encoding="utf-8")


def embed(html, marker, path):
    start = html.index(marker) + len(marker)
    _, end = json.JSONDecoder().raw_decode(html, start)   # find where the old JSON ends, whatever follows it
    return html[:start] + json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":")) + html[end:]


html = embed(html, "let RES=", ROOT / "data" / "processed" / "results.json")
html = embed(html, "const MAP=", ROOT / "data" / "map_mh_ka.json")
(ROOT / "index.html").write_text(html, encoding="utf-8")
print(f"embedded results + map; index.html is {len(html.encode()) // 1024} KB")

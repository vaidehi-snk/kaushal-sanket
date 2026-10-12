"""One-off: Maharashtra + Karnataka district outlines as SVG paths for index.html.
Source: LGD_Districts.parquet from ramSeraph/indian_admin_boundaries (LGD/Bharatmaps boundaries, CC0).
Usage: python tools/build_map.py <LGD_Districts.parquet>  -> data/map_mh_ka.json"""
import json, math, sys
from pathlib import Path
import pyarrow.parquet as pq
from shapely import wkb

RENAME = {"Aurangabad": "Chhatrapati Sambhajinagar", "Ahmednagar": "Ahilyanagar", "Osmanabad": "Dharashiv"}
t = pq.read_table(sys.argv[1], columns=["dtname", "state_lgd", "geometry"]).to_pandas()
t = t[t.state_lgd.isin([27, 29])]
geoms = [(RENAME.get(n.title(), n.title()), "Maharashtra" if s == 27 else "Karnataka", wkb.loads(g).simplify(0.02, preserve_topology=True))
         for n, s, g in zip(t.dtname, t.state_lgd, t.geometry)]
x0, y0, x1, y1 = (min(g.bounds[0] for *_, g in geoms), min(g.bounds[1] for *_, g in geoms),
                  max(g.bounds[2] for *_, g in geoms), max(g.bounds[3] for *_, g in geoms))
k = math.cos(math.radians((y0 + y1) / 2))          # equirectangular, good enough for two states
W = 600; S = W / ((x1 - x0) * k); H = round((y1 - y0) * S)
P = lambda x, y: f"{(x - x0) * k * S:.1f},{(y1 - y) * S:.1f}"
out = []
for name, state, g in geoms:
    polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
    d = "".join("M" + "L".join(P(*c) for c in p.exterior.coords) + "Z" for p in polys if p.area > 1e-4)
    c = g.representative_point()
    out.append({"n": name, "s": state, "d": d, "c": [round((c.x - x0) * k * S), round((y1 - c.y) * S)]})
Path(__file__).resolve().parents[1].joinpath("data", "map_mh_ka.json").write_text(
    json.dumps({"w": W, "h": H, "districts": out}, separators=(",", ":")), encoding="utf8")
print(len(out), "districts", W, "x", H)

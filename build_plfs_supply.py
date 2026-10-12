"""Run: python build_plfs_supply.py "E:\\path\\2025.csv"
Reads only needed columns, keeps Maharashtra, writes AGGREGATE tables to data/processed (no person rows)."""
import sys
from pathlib import Path
import pandas as pd

want = ["State/UT", "District", "Survey Month", "Sector", "Person Final Weight", "Age",
        "Technical Education Level", "Principal Occupation", "Vocational Training",
        "Training/Course Completion", "Training Completed (Last 365 Days)", "Field of Training",
        "Training Type", "Training Certifying Body"]
path = sys.argv[1]
have = pd.read_csv(path, nrows=0, encoding_errors="replace").columns
df = pd.read_csv(path, usecols=[c for c in want if c in have], low_memory=False, encoding_errors="replace")
df = df[df["State/UT"].astype(str).str.contains("maharashtra", case=False)]
out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
W = "Person Final Weight"

cov = df.groupby("District").size().rename("persons").reset_index()
cov["low_sample"] = cov.persons < 500
cov.to_csv(out / "plfs_coverage.csv", index=False)

occ = (df.dropna(subset=["Principal Occupation"]).groupby(["District", "Principal Occupation"])
         .agg(n=(W, "size"), wt=(W, "sum")).reset_index())
occ["share"] = occ.wt / occ.groupby("District").wt.transform("sum")   # shares, not totals
occ[occ.n >= 30].drop(columns="wt").to_csv(out / "plfs_district_occupation.csv", index=False)  # n<30 suppressed

for c in ["Vocational Training", "Training/Course Completion", "Training Completed (Last 365 Days)",
          "Training Type", "Training Certifying Body", "Field of Training"]:
    if c in df:
        print(f"\n== {c} ==\n", df[c].value_counts(dropna=False).head(12).to_string())
if "Field of Training" in df:
    tr = df.dropna(subset=["Field of Training"]).groupby(["District", "Field of Training"]).size().rename("n").reset_index()
    tr[tr.n >= 30].to_csv(out / "plfs_district_training.csv", index=False)
print("\nDone. Files in", out)

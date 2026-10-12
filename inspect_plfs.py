"""Run:  python inspect_plfs.py "C:\\path\\to\\2025.csv"
Prints a summary of the PLFS file and saves columns.txt. Does not modify your data."""
import sys
import pandas as pd

if len(sys.argv) < 2:
    sys.exit('Usage: python inspect_plfs.py "path\\to\\2025.csv"')
path = sys.argv[1]

header = pd.read_csv(path, nrows=0, encoding_errors="replace").columns.tolist()
with open("columns.txt", "w", encoding="utf-8") as f:
    for i, c in enumerate(header):
        f.write(f"{i}\t{c}\n")
print(f"Columns: {len(header)}  (full list saved to columns.txt)")


def find(*words):
    return [c for c in header if all(w in c.lower() for w in words)]


print("\n--- Columns that look like codes or training ---")
for w in ["code", "nco", "training", "vocational", "skill", "certif", "technical"]:
    print(f"{w:12s}", find(w))


def pick(*words):
    m = find(*words)
    return m[0] if m else None


cols = {
    "state": pick("state"), "district": pick("district"), "month": pick("survey", "month"),
    "quarter": pick("quarter"), "sector": pick("sector"), "weight": pick("person", "final", "weight"),
    "sex": pick("sex"), "age": pick("age"), "tech": pick("technical", "education", "lev"),
    "gen_edu": pick("general", "education"), "status": pick("principal", "activity", "status"),
    "occ": pick("principal", "occupation"), "ind": pick("principal", "industry"),
    "person": pick("person", "id"),
}
print("\n--- Key columns found ---")
for k, v in cols.items():
    print(f"{k:9s} -> {v}")

use = sorted({v for v in cols.values() if v} | set(find("training") + find("vocational") + find("code")))
df = pd.read_csv(path, usecols=use, low_memory=False, encoding_errors="replace")
print(f"\nRows (persons): {len(df):,}")

S, D, M = cols["state"], cols["district"], cols["month"]
if S:
    print("\n--- Persons per state (top 10) ---")
    print(df[S].value_counts().head(10).to_string())
if M:
    print("\n--- Persons per survey month ---")
    print(df[M].value_counts().to_string())
if D:
    print(f"\nDistricts in file: {df[D].nunique()}")
    mh = df[df[S].astype(str).str.contains("maharashtra", case=False, na=False)] if S else df
    print(f"\n--- Maharashtra: {len(mh):,} persons, {mh[D].nunique()} districts ---")
    per = mh[D].value_counts()
    print("Persons per district: min", per.min(), "median", int(per.median()), "max", per.max())
    print(per.to_string())
    if M:
        pm = mh.groupby([D, M]).size()
        print("\nPersons per district-month: min", pm.min(), "median", int(pm.median()), "max", pm.max())
if cols["tech"]:
    print("\n--- Technical education level (all India) ---")
    print(df[cols["tech"]].value_counts(dropna=False).to_string())
if cols["occ"]:
    print(f"\nDistinct principal occupation labels: {df[cols['occ']].nunique()}")
    print("\n--- Top 15 occupations (Maharashtra) ---" if D else "\n--- Top 15 occupations ---")
    src = mh if D else df
    print(src[cols["occ"]].value_counts().head(15).to_string())

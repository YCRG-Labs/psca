import pandas as pd
from pathlib import Path
import sys

csv_path = Path("data/anes_timeseries_2024_csv_20250808.csv")

if not csv_path.exists():
    print(f"Error: Could not find {csv_path}")
    sys.exit(1)

print(f"Scanning headers in {csv_path.name}...\n")

df = pd.read_csv(csv_path, nrows=0, low_memory=False)
cols = df.columns.tolist()

print("--- Possible Party ID Candidates ---")
party_matches = [c for c in cols if 'V24122' in c or 'PARTY' in c.upper() or 'PID' in c.upper()]
if party_matches:
    for c in party_matches:
        print(f"  - {c}")
else:
    print("  No obvious matches found. You may need to check the codebook.")

print("\n--- Possible Immigration Candidates ---")
immig_matches = [c for c in cols if 'V24174' in c or 'IMMIG' in c.upper()]
if immig_matches:
    for c in immig_matches:
        print(f"  - {c}")
else:
    print("  No obvious matches found. You may need to check the codebook.")
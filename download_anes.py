import sys
import zipfile
from pathlib import Path

DATA_DIR = Path("data")
EXPECTED_PATTERN = "anes_timeseries_2024*"
OUTPUT_NAME = "anes_timeseries_2024.csv"

PARTY_VAR = "V241227"
ITEM_VARS = {
    "gov_spending": "V241239",
    "immigration": "V241747SPS",
    "gun_control": "V242325",
}


def find_csv(data_dir):
    for f in sorted(data_dir.glob("*.csv")):
        if "anes_timeseries_2024" in f.name:
            return f
    return None


def extract_if_zip(data_dir):
    for f in data_dir.glob("*.zip"):
        if "anes" in f.name.lower():
            print(f"Extracting {f.name}...")
            with zipfile.ZipFile(f) as zf:
                zf.extractall(data_dir)
            return True
    return False


def validate(csv_path):
    import pandas as pd
    print(f"Validating {csv_path.name}...")
    df = pd.read_csv(csv_path, nrows=5, low_memory=False)

    missing = []
    for item, var in ITEM_VARS.items():
        if var not in df.columns:
            missing.append(f"  {item}: {var}")
    if PARTY_VAR not in df.columns:
        missing.append(f"  party: {PARTY_VAR}")

    if missing:
        print("MISSING VARIABLES:")
        for m in missing:
            print(m)
        return False

    df_full = pd.read_csv(csv_path, low_memory=False)
    print(f"Rows: {len(df_full):,}")
    print(f"Columns: {len(df_full.columns):,}")

    for item, var in ITEM_VARS.items():
        valid = df_full[df_full[var] > 0][var]
        print(f"{item} ({var}): {len(valid):,} valid responses, range {valid.min()}-{valid.max()}")

    party = df_full[PARTY_VAR]
    party_valid = party[party.between(1, 7)]
    dem = party_valid.between(1, 3).sum()
    rep = party_valid.between(5, 7).sum()
    ind = (party_valid == 4).sum()
    print(f"Party ID: {dem:,} D, {rep:,} R, {ind:,} I")

    symlink = csv_path.parent / OUTPUT_NAME
    if symlink != csv_path and not symlink.exists():
        symlink.symlink_to(csv_path.name)
        print(f"Symlinked to {OUTPUT_NAME}")

    print("\nReady for: python3 pilot.py anes --output full_lhs.json")
    return True


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    csv = find_csv(DATA_DIR)
    if csv is None:
        extract_if_zip(DATA_DIR)
        csv = find_csv(DATA_DIR)

    if csv is None:
        print("No ANES CSV found in data/")
        print()
        print("Download from: https://electionstudies.org/data-center/2024-time-series-study/")
        print("Put the CSV (or ZIP) in the data/ folder, then run this script again.")
        sys.exit(1)

    if not validate(csv):
        sys.exit(1)

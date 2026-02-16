# scripts/sanity_check.py

from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED_PATH = Path("data/processed")

def check_dataset(csv_path):
    print("=" * 80)
    print(f"Checking: {csv_path.name}")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    # 1️⃣ Shape
    print(f"Shape: {df.shape}")

    # 2️⃣ Column count
    print(f"Number of columns: {len(df.columns)}")

    # 3️⃣ Check target existence
    if "activity" not in df.columns:
        print("ERROR: 'activity' column missing")
        return

    # 4️⃣ NaN check
    total_nans = df.isna().sum().sum()
    print(f"Total NaNs: {total_nans}")

    # 5️⃣ Zero variance features
    variances = df.drop(columns=["activity"]).var()
    zero_var_cols = variances[variances == 0].index.tolist()
    print(f"Zero variance features: {len(zero_var_cols)}")

    # 6️⃣ Target statistics
    print("\nTarget statistics:")
    print(df["activity"].describe())

    # 7️⃣ Feature summary
    print("\nFeature variance summary:")
    print(variances.describe())

    print("\n")


def main():
    csv_files = list(PROCESSED_PATH.glob("*_rdkit.csv"))

    if not csv_files:
        print("No processed CSV files found.")
        return

    for csv_file in csv_files:
        check_dataset(csv_file)


if __name__ == "__main__":
    main()
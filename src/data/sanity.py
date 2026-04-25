# src/data/sanity.py

from pathlib import Path
import pandas as pd
import numpy as np


PROCESSED_PATH = Path("data/processed")


def sanity_check(csv_path: Path):

    df = pd.read_csv(csv_path)

    print(f"\nChecking: {csv_path.name}")
    print("=" * 70)

    print(f"Shape: {df.shape}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Total NaNs: {df.isna().sum().sum()}")

    # Zero variance features
    variances = df.drop(columns=["activity"]).var()
    zero_var = (variances == 0).sum()
    print(f"Zero variance features: {zero_var}")

    # Target stats
    print("\nTarget statistics:")
    print(df["activity"].describe())

    print("\nFeature variance summary:")
    print(variances.describe())


def run_all_sanity_checks():
    """
    Run sanity checks for all processed CSV datasets.
    """

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError("data/processed does not exist")

    csv_files = sorted(PROCESSED_PATH.glob("*_rdkit.csv"))

    if len(csv_files) == 0:
        raise ValueError("No processed CSV files found.")

    for csv_file in csv_files:
        sanity_check(csv_file)

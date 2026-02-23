import os
import sys
import argparse

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ==============================
# Imports
# ==============================

from src.data.build_dataset import build_all_datasets
from src.data.sanity import run_all_sanity_checks
from src.analysis.reproduce_table1 import reproduce_table1
from src.analysis.shap_postprocessing import run_full_analysis
from src.modeling.train_lightgbm import train_and_evaluate
from src.modeling.shap_analysis import run_all_shap


# ==============================
# Endpoint CSV files
# ==============================

ENDPOINTS = [
    "ADME_HLM_rdkit.csv",
    "ADME_hPPB_rdkit.csv",
    "ADME_MDR1_ER_rdkit.csv",
    "ADME_RLM_rdkit.csv",
    "ADME_rPPB_rdkit.csv",
    "ADME_Sol_rdkit.csv",
]


# =====================================
# TRAINING PIPELINE
# =====================================

def run_training_pipeline():

    print("\n==============================")
    print("Training all endpoints")
    print("==============================")

    results = []

    for csv_file in ENDPOINTS:
        print(f"\nTraining {csv_file}...")
        res = train_and_evaluate(csv_file)
        results.append(res)

    import pandas as pd

    os.makedirs("reports/results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("reports/results/final_results.csv", index=False)

    print("\nFinal Results:")
    print(df)
    print("\nSaved to reports/results/final_results.csv")


# =====================================
# FULL PIPELINE
# =====================================

def run_full_pipeline():

    print("\n========== FULL PIPELINE START ==========\n")

    print("Step 1️⃣  Building datasets...")
    build_all_datasets()

    print("\nStep 2️⃣  Running sanity checks...")
    run_all_sanity_checks()

    print("\nStep 3️⃣  Reproducing Table 1...")
    reproduce_table1()

    print("\nStep 4️⃣  Training LightGBM models...")
    run_training_pipeline()

    print("\nStep 5️⃣  Running SHAP analysis...")
    run_all_shap()

    print("\nStep 6️⃣  Running SHAP post-analysis...")
    run_full_analysis()

    print("\n========== PIPELINE COMPLETE ==========\n")


# =====================================
# CLI ENTRY POINT
# =====================================

def main():

    parser = argparse.ArgumentParser(description="ADME ML Pipeline")

    parser.add_argument("--build", action="store_true", help="Build datasets")
    parser.add_argument("--sanity", action="store_true", help="Run sanity checks")
    parser.add_argument("--table1", action="store_true", help="Reproduce Table 1")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--shap", action="store_true", help="Run SHAP analysis")
    parser.add_argument("--analysis", action="store_true", help="Run SHAP post-analysis (Table3 + correlations)")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    if args.all:
        run_full_pipeline()
        return

    if args.build:
        build_all_datasets()

    if args.sanity:
        run_all_sanity_checks()

    if args.table1:
        reproduce_table1()

    if args.train:
        run_training_pipeline()

    if args.shap:
        run_all_shap()

    if args.analysis:
        run_full_analysis()


if __name__ == "__main__":
    main()

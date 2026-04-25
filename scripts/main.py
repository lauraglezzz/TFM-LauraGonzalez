import os
import sys
import argparse

# =========================================================
# PATH
# =========================================================
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(ROOT_PATH, "src")

sys.path.append(ROOT_PATH)
sys.path.append(SRC_PATH)

# ==============================
# DATA & MODELING
# ==============================

from src.data.build_dataset import build_all_datasets
from src.data.sanity import run_all_sanity_checks
from src.analysis.reproduce_table1 import reproduce_table1
from src.analysis.shap_postprocessing import run_full_analysis

from src.modeling.train_lightgbm import train_and_evaluate          # baseline (log)
from src.modeling.train_lightgbm_log import train_and_evaluate_log  # real scale

from src.modeling.shap_analysis import run_all_shap
from src.analysis.shap_local_explanations import run_all_local_shap

# ==============================
# LLM
# ==============================

from llm.core.explanation_generator import run_llm_explanations        # baseline
from src.analysis.generate_local_llm_explanations import run_local_llm # local (real)

# ==============================
# LLM EXPERIMENTS (BASELINE)
# ==============================

from llm.experiments.run_llm_model_comparison import run_model_comparison
from llm.experiments.run_llm_prompt_experiments import run_prompt_experiments

# ==============================
# LLM EVALUATION (BASELINE)
# ==============================

from llm.evaluation.run_llm_judge import run_llm_judge
from src.analysis.evaluate_llm_explanations import run_evaluation
from src.analysis.llm_as_a_judge_results import run_judge_analysis

# ==============================
# ENDPOINTS
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
# TRAINING (BASELINE - LOG SCALE)
# =====================================

def run_training_pipeline():

    print("\n==============================")
    print("Training BASELINE (log scale)")
    print("==============================")

    results = []

    for csv_file in ENDPOINTS:
        print(f"\nTraining {csv_file}...")
        res = train_and_evaluate(csv_file)
        results.append(res)

    import pandas as pd
    os.makedirs("reports/results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("reports/results/final_results_log.csv", index=False)

    print("\nSaved baseline results to reports/results/final_results_log.csv")


# =====================================
# TRAINING (REAL SCALE)
# =====================================

def run_training_pipeline_real():

    print("\n==============================")
    print("Training REAL SCALE models")
    print("==============================")

    results = []

    for csv_file in ENDPOINTS:
        print(f"\nTraining {csv_file}...")
        res = train_and_evaluate_log(csv_file)   # 👈 FIX AQUÍ
        results.append(res)

    import pandas as pd
    os.makedirs("reports/results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("reports/results/final_results_real.csv", index=False)

    print("\nSaved real-scale results to reports/results/final_results_real.csv")


# =====================================
# FULL PIPELINE BASELINE (PAPER)
# =====================================

def run_full_pipeline_baseline():

    print("\n========== BASELINE PIPELINE ==========\n")

    build_all_datasets()
    run_all_sanity_checks()
    reproduce_table1()

    run_training_pipeline()
    run_all_shap()
    run_full_analysis()

    # LLM experiments (baseline only)
    run_llm_explanations()
    run_prompt_experiments()
    run_model_comparison()
    run_evaluation()
    run_llm_judge()
    run_judge_analysis()

    print("\n========== BASELINE COMPLETE ==========\n")


# =====================================
# FULL PIPELINE REAL (INTERPRETABILITY)
# =====================================

def run_full_pipeline_real():

    print("\n========== REAL SCALE PIPELINE ==========\n")

    build_all_datasets()
    run_all_sanity_checks()

    run_training_pipeline_real()
    run_all_shap()
    run_all_local_shap()
    run_full_analysis()

    run_local_llm()   # 👈 LLM LOCAL

    print("\n========== REAL PIPELINE COMPLETE ==========\n")


# =====================================
# CLI
# =====================================

def main():

    parser = argparse.ArgumentParser(description="ADME ML + LLM Pipeline")

    # Core
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_real", action="store_true")
    parser.add_argument("--shap", action="store_true")
    parser.add_argument("--shap_local", action="store_true")

    # LLM
    parser.add_argument("--llm_gen", action="store_true")   # baseline
    parser.add_argument("--llm_local", action="store_true") # real

    # Full pipelines
    parser.add_argument("--all_baseline", action="store_true")
    parser.add_argument("--all_real", action="store_true")

    args = parser.parse_args()

    if args.all_baseline:
        run_full_pipeline_baseline()
        return

    if args.all_real:
        run_full_pipeline_real()
        return

    # -------------------------
    # CORE
    # -------------------------

    if args.train:
        run_training_pipeline()

    if args.train_real:
        run_training_pipeline_real()

    if args.shap:
        run_all_shap()

    if args.shap_local:
        run_all_local_shap()

    # -------------------------
    # LLM
    # -------------------------

    if args.llm_gen:
        run_llm_explanations()

    if args.llm_local:
        run_local_llm()


if __name__ == "__main__":
    main()
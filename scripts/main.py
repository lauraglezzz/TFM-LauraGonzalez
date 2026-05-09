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

from src.modeling.train_lightgbm import train_and_evaluate
from src.modeling.train_lightgbm_log import train_and_evaluate_log

from src.modeling.shap_analysis import run_all_shap
from src.analysis.shap_local_explanations import run_all_local_shap

from src.modeling.train_lightgbm_classifier import run_all_classifiers
from src.analysis.shap_local_classifier import run_all_local_shap_classifier  # 👈 NUEVO

# ==============================
# LLM
# ==============================

from llm.core.explanation_generator import run_llm_explanations
from src.analysis.generate_local_llm_explanations import run_local_llm

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

    print("\nSaved baseline results")


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
        res = train_and_evaluate_log(csv_file)
        results.append(res)

    import pandas as pd
    os.makedirs("reports/results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("reports/results/final_results_real.csv", index=False)

    print("\nSaved real-scale results")


# =====================================
# TRAINING CLASIFICACIÓN
# =====================================

def run_training_pipeline_classifier():

    print("\n==============================")
    print("Training CLASSIFICATION models")
    print("==============================")

    run_all_classifiers()

    print("\nSaved classification results")


# =====================================
# FULL PIPELINE BASELINE
# =====================================

def run_full_pipeline_baseline():

    print("\n========== BASELINE PIPELINE ==========\n")

    build_all_datasets()
    run_all_sanity_checks()
    reproduce_table1()

    run_training_pipeline()
    run_all_shap()
    run_full_analysis()

    run_llm_explanations()
    run_prompt_experiments()
    run_model_comparison()
    run_evaluation()
    run_llm_judge()
    run_judge_analysis()

    print("\n========== BASELINE COMPLETE ==========\n")


# =====================================
# FULL PIPELINE REAL
# =====================================

def run_full_pipeline_real():

    print("\n========== REAL SCALE PIPELINE ==========\n")

    build_all_datasets()
    run_all_sanity_checks()

    run_training_pipeline_real()
    run_training_pipeline_classifier()

    run_all_shap()                   # global regresión
    run_all_local_shap()             # local regresión
    run_all_local_shap_classifier()  # 👈 NUEVO (clasificación)

    run_full_analysis()

    run_local_llm()

    print("\n========== REAL PIPELINE COMPLETE ==========\n")


# =====================================
# CLI
# =====================================

def main():

    parser = argparse.ArgumentParser(description="ADME ML + LLM Pipeline")

    # Core
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_real", action="store_true")
    parser.add_argument("--train_clf", action="store_true")
    parser.add_argument("--shap", action="store_true")
    parser.add_argument("--shap_local", action="store_true")
    parser.add_argument("--shap_local_clf", action="store_true")  # 👈 NUEVO

    # LLM
    parser.add_argument("--llm_gen", action="store_true")
    parser.add_argument("--llm_local", action="store_true")

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

    if args.train_clf:
        run_training_pipeline_classifier()

    if args.shap:
        run_all_shap()

    if args.shap_local:
        run_all_local_shap()

    if args.shap_local_clf:
        run_all_local_shap_classifier()

    # -------------------------
    # LLM
    # -------------------------

    if args.llm_gen:
        run_llm_explanations()

    if args.llm_local:
        run_local_llm()


if __name__ == "__main__":
    main()
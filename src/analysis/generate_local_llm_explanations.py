import os
import sys
from pathlib import Path
import json
import pandas as pd

# =========================================================
# PATH 
# =========================================================
SRC_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_PATH))

from llm.core.llm_client import query_llm
from llm.core.prompt_builder_local import build_prompt_local


# =========================================================
# CONFIG
# =========================================================
MODEL = "gpt-4.1-mini"
STRATEGY = "biomedical"
TOP_K = 3

BASE_PATH = Path("reports/shap_local")
OUTPUT_PATH = Path("reports/llm_local")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# =========================================================
# CLEAN OUTPUT
# =========================================================
def clean_explanation(text: str) -> str:
    if not isinstance(text, str):
        return text

    prefixes = [
        "Certainly!",
        "Sure!",
        "Here is the explanation:",
        "Here’s the explanation:",
        "Here's an explanation:",
    ]

    for p in prefixes:
        if text.strip().startswith(p):
            text = text.strip()[len(p):].strip()

    return text


# =========================================================
# MAIN
# =========================================================
def run_local_llm():

    print("\n==============================")
    print("Running LOCAL LLM explanations")
    print("==============================")

    if not BASE_PATH.exists():
        print(f"ERROR: {BASE_PATH} does not exist")
        return

    # =====================================================
    # 1. GENERATE (ONLY MISSING)
    # =====================================================
    for endpoint_dir in BASE_PATH.iterdir():

        if not endpoint_dir.is_dir():
            continue

        endpoint = endpoint_dir.name
        print(f"\nProcessing endpoint: {endpoint}")

        for sample_dir in endpoint_dir.iterdir():

            if not sample_dir.is_dir():
                continue

            sample_name = sample_dir.name

            top_features_file = sample_dir / "top_features.json"
            pred_file = sample_dir / "prediction.json"
            explanation_file = sample_dir / "explanation.txt"

            # -------------------------
            # SKIP if already done
            # -------------------------
            if explanation_file.exists():
                try:
                    with open(explanation_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()

                    if content != "" and content != "ERROR":
                        print(f"Skipping {sample_name} (already done)")
                        continue
                except:
                    pass

            # -------------------------
            # Checks
            # -------------------------
            if not top_features_file.exists():
                continue

            if not pred_file.exists():
                continue

            # -------------------------
            # Load data
            # -------------------------
            try:
                with open(top_features_file, "r") as f:
                    top_features = json.load(f)

                with open(pred_file, "r") as f:
                    pred_value = json.load(f)["prediction"]

            except:
                continue

            if len(top_features) == 0:
                continue

            features = [f["feature"] for f in top_features][:TOP_K]
            shap_vals = [float(f["impact"]) for f in top_features][:TOP_K]

            # -------------------------
            # Prompt + LLM
            # -------------------------
            try:
                prompt = build_prompt_local(
                    features=features,
                    shap_values=shap_vals,
                    endpoint=endpoint,
                    prediction=pred_value,
                    strategy=STRATEGY
                )

                explanation = query_llm(prompt, model=MODEL)
                explanation = clean_explanation(explanation)

            except Exception as e:
                print(f"LLM error in {sample_name}: {e}")
                explanation = "ERROR"

            # -------------------------
            # Save
            # -------------------------
            try:
                with open(explanation_file, "w", encoding="utf-8") as f:
                    f.write(explanation)
            except:
                pass

            print(f"Done: {sample_name}")

    # =====================================================
    # 2. REBUILD FULL CSV (SOURCE OF TRUTH)
    # =====================================================
    print("\nRebuilding CSV from disk...")

    all_results = []

    for endpoint_dir in BASE_PATH.iterdir():

        if not endpoint_dir.is_dir():
            continue

        endpoint = endpoint_dir.name

        for sample_dir in endpoint_dir.iterdir():

            if not sample_dir.is_dir():
                continue

            sample_name = sample_dir.name

            explanation_file = sample_dir / "explanation.txt"
            top_features_file = sample_dir / "top_features.json"
            pred_file = sample_dir / "prediction.json"

            molecule_img = sample_dir / "molecule.png"
            shap_img = sample_dir / "waterfall.png"

            if not explanation_file.exists():
                continue

            try:
                with open(explanation_file, "r", encoding="utf-8") as f:
                    explanation = f.read().strip()

                if explanation == "" or explanation == "ERROR":
                    continue
            except:
                continue

            try:
                with open(top_features_file, "r") as f:
                    top_features = json.load(f)

                features = [f["feature"] for f in top_features][:TOP_K]
                shap_vals = [float(f["impact"]) for f in top_features][:TOP_K]
            except:
                features, shap_vals = [], []

            try:
                with open(pred_file, "r") as f:
                    pred_value = json.load(f)["prediction"]
            except:
                pred_value = None

            all_results.append({
                "endpoint": endpoint,
                "sample": sample_name,
                "model": MODEL,
                "strategy": STRATEGY,
                "top_k": TOP_K,
                "prediction": pred_value,
                "features": str(features),
                "shap_values": str(shap_vals),
                "molecule_img": str(molecule_img) if molecule_img.exists() else "",
                "shap_img": str(shap_img) if shap_img.exists() else "",
                "explanation": explanation
            })

    if len(all_results) == 0:
        print("No valid explanations found.")
        return

    df = pd.DataFrame(all_results)

    output_csv = OUTPUT_PATH / "local_llm_explanations.csv"
    df.to_csv(output_csv, index=False)

    print("\nSaved FULL dataset:")
    print(output_csv)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_local_llm()
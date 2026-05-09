# src/analysis/generate_local_llm_explanations.py

import json
import pandas as pd
from pathlib import Path

from llm.core.llm_client import query_llm
from llm.core.prompt_builder_local import build_prompt_local


# ==============================
# PATHS
# ==============================
BASE_PATH = Path("reports/shap_local_v2")
OUTPUT_PATH = Path("reports/llm_local_v2")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4.1-mini"


# ==============================
# CLEAN TEXT
# ==============================
def clean_explanation(text):
    if not isinstance(text, str):
        return text

    prefixes = ["Certainly!", "Sure!", "Here is", "Here's"]

    for p in prefixes:
        if text.strip().startswith(p):
            text = text[len(p):].strip()

    return text


# ==============================
# UNITS
# ==============================
ENDPOINT_UNITS = {
    "ADME_HLM": "mL/min/kg",
    "ADME_RLM": "mL/min/kg",
    "ADME_hPPB": "%",
    "ADME_rPPB": "%",
    "ADME_MDR1_ER": "",
    "ADME_Sol": "µg/mL",
}


# ==============================
# CATEGORIZATION
# ==============================
def categorize_endpoint(endpoint, value):

    if endpoint in ["ADME_HLM", "ADME_RLM"]:
        if value <= 10:
            return "low"
        elif value <= 40:
            return "moderate"
        else:
            return "high"

    elif endpoint in ["ADME_hPPB", "ADME_rPPB"]:
        if value <= 0.5:
            return "very high binding"
        elif value <= 5:
            return "intermediate binding"
        else:
            return "low binding"

    elif endpoint == "ADME_MDR1_ER":
        if value <= 2:
            return "non-substrate"
        elif value <= 5:
            return "moderate efflux"
        else:
            return "strong efflux"

    elif endpoint == "ADME_Sol":
        if value < 10:
            return "poor solubility"
        elif value < 100:
            return "moderate solubility"
        else:
            return "high solubility"

    return "unknown"


# ==============================
# MAIN
# ==============================
def run_local_llm(force=True):

    print("\n==============================")
    print("Running LOCAL LLM (v2)")
    print("==============================")

    # =========================
    # GENERATE TXT FILES
    # =========================
    for endpoint_dir in BASE_PATH.iterdir():

        if not endpoint_dir.is_dir():
            continue

        endpoint = endpoint_dir.name
        unit = ENDPOINT_UNITS.get(endpoint, "")

        print(f"\nProcessing {endpoint}")

        for sample_dir in endpoint_dir.iterdir():

            if not sample_dir.is_dir():
                continue

            explanation_file = sample_dir / "explanation.txt"

            # Skip si ya existe (opcional)
            if explanation_file.exists() and not force:
                continue

            pred_file = sample_dir / "prediction.json"
            feat_file = sample_dir / "top_features.json"

            if not pred_file.exists() or not feat_file.exists():
                continue

            # LOAD DATA
            with open(pred_file) as f:
                pred_data = json.load(f)

            with open(feat_file) as f:
                features_data = json.load(f)

            prediction = pred_data["prediction"]
            base_value = pred_data["base_value"]

            features = [f["feature"] for f in features_data]
            shap_vals = [float(f["impact"]) for f in features_data]

            category = categorize_endpoint(endpoint, prediction)

            # BUILD PROMPT
            prompt = build_prompt_local(
                features,
                shap_vals,
                endpoint,
                pred_data["prediction"],
                pred_data["base_value"]
            )

            # CALL LLM
            try:
                explanation = query_llm(prompt, model=MODEL)
                explanation = clean_explanation(explanation)
            except Exception as e:
                print(f"LLM error: {e}")
                explanation = "ERROR"

            # SAVE TXT
            with open(explanation_file, "w", encoding="utf-8") as f:
                f.write(explanation)

            print("Done", sample_dir.name)

    # =========================
    # REBUILD CSV
    # =========================
    print("\nRebuilding CSV...")

    all_results = []

    for endpoint_dir in BASE_PATH.iterdir():

        if not endpoint_dir.is_dir():
            continue

        endpoint = endpoint_dir.name

        for sample_dir in endpoint_dir.iterdir():

            explanation_file = sample_dir / "explanation.txt"
            pred_file = sample_dir / "prediction.json"
            feat_file = sample_dir / "top_features.json"

            if not explanation_file.exists():
                continue

            try:
                with open(explanation_file) as f:
                    explanation = f.read().strip()

                if explanation == "" or explanation == "ERROR":
                    continue

                with open(pred_file) as f:
                    pred_data = json.load(f)

                with open(feat_file) as f:
                    features_data = json.load(f)

            except:
                continue

            prediction = pred_data["prediction"]
            base_value = pred_data["base_value"]

            category = categorize_endpoint(endpoint, prediction)

            features = [f["feature"] for f in features_data]
            shap_vals = [float(f["impact"]) for f in features_data]

            all_results.append({
                "endpoint": endpoint,
                "sample": sample_dir.name,
                "prediction": prediction,
                "base_value": base_value,
                "category": category,
                "features": str(features[:3]),
                "shap_values": str(shap_vals[:3]),
                "explanation": explanation
            })

    df = pd.DataFrame(all_results)

    output_csv = OUTPUT_PATH / "local_llm_explanations_v2.csv"
    df.to_csv(output_csv, index=False)

    print("\nSaved CSV:")
    print(output_csv)


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    run_local_llm()
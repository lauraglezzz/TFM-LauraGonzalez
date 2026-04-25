# =========================================================
# LOCAL PROMPT BUILDER (FINAL VERSION)
# =========================================================

# =========================================================
# REAL SCALE UNITS
# =========================================================
ENDPOINT_UNITS = {
    "ADME_HLM": "mL/min/kg",
    "ADME_RLM": "mL/min/kg",
    "ADME_hPPB": "% unbound",
    "ADME_rPPB": "% unbound",
    "ADME_MDR1_ER": "efflux ratio",
    "ADME_Sol": "µg/mL"
}


# =========================================================
# DISCRETIZATION (CLINICAL CONTEXT)
# =========================================================
def discretize_endpoint(endpoint, value):

    if endpoint in ["ADME_HLM", "ADME_RLM"]:
        if value <= 10:
            return "Low"
        elif value <= 40:
            return "Moderate"
        else:
            return "High"

    elif endpoint in ["ADME_hPPB", "ADME_rPPB"]:
        if value <= 0.5:
            return "Very high binding"
        elif value <= 5:
            return "Intermediate binding"
        else:
            return "Low binding"

    elif endpoint == "ADME_MDR1_ER":
        if value <= 2:
            return "Non-substrate"
        elif value <= 5:
            return "Moderate efflux"
        else:
            return "Strong efflux"

    elif endpoint == "ADME_Sol":
        if value < 10:
            return "Poor solubility"
        elif value < 100:
            return "Moderate solubility"
        else:
            return "High solubility"

    return ""


# =========================================================
# PROMPT BUILDER
# =========================================================
def build_prompt_local(
    features,
    shap_values,
    endpoint,
    prediction,
    strategy="biomedical"
):

    unit = ENDPOINT_UNITS.get(endpoint, "")
    category = discretize_endpoint(endpoint, prediction)

    # =========================
    # Feature formatting
    # =========================
    feature_lines = []
    for i, (feat, val) in enumerate(zip(features, shap_values), start=1):
        sign = "+" if val > 0 else ""
        feature_lines.append(f"{i}. {feat} ({sign}{round(val, 3)})")

    feature_text = "\n".join(feature_lines)

    # =========================
    # BIOMEDICAL PROMPT
    # =========================
    if strategy == "biomedical":

        prompt = f"""
You are a biomedical expert in pharmacokinetics and drug metabolism.

A drug property has been predicted for {endpoint}.

Predicted value:
{round(prediction, 3)} {unit} → {category}

The most influential molecular descriptors are:

{feature_text}

INSTRUCTIONS:

- For EACH descriptor, explicitly interpret the SIGN:
    - (+) increases the predicted property
    - (-) decreases the predicted property

- For each descriptor you MUST:
    1. Explain its chemical meaning
    2. Explain whether it increases or decreases the property
    3. Connect it to pharmacokinetics (metabolism, binding, permeability, etc.)

- The final prediction results from the combination of many molecular features.
- If the listed descriptors do not fully explain the final value, acknowledge that additional molecular features not shown also influence the outcome.

- Keep explanations mechanistic and biomedical:
  (enzyme interactions, polarity, lipophilicity, steric effects, etc.)

- Do NOT mention SHAP, models, or machine learning.

- Structure:
    - Start with 1–2 sentences interpreting the predicted value and category
    - Then explain each descriptor in order (1, 2, 3)

- Be precise and avoid generic statements.

Start directly with the explanation.
"""

    else:
        prompt = f"""
Predicted: {round(prediction, 3)} {unit} ({category})

Top features:
{feature_text}

Explain how each feature increases or decreases the prediction.
"""

    return prompt
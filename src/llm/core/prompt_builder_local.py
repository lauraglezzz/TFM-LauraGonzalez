# src/llm/core/prompt_builder_local.py


# ==============================
# ENDPOINT CONTEXT
# ==============================
def get_endpoint_context(endpoint):

    if endpoint in ["ADME_HLM", "ADME_RLM"]:
        return "hepatic metabolic clearance (intrinsic clearance in liver microsomes)", "mL/min/kg"

    if endpoint in ["ADME_hPPB", "ADME_rPPB"]:
        return "plasma protein binding (fraction unbound)", "%"

    if endpoint == "ADME_MDR1_ER":
        return "efflux ratio mediated by transporters such as P-glycoprotein", ""

    if endpoint == "ADME_Sol":
        return "aqueous solubility at physiological pH", "µg/mL"

    return "biological property", ""


# ==============================
# DISCRETIZATION (THRESHOLDS)
# ==============================
def discretize(endpoint, value):

    if endpoint in ["ADME_HLM", "ADME_RLM"]:
        if value <= 10:
            return "low"
        elif value <= 40:
            return "moderate"
        else:
            return "high"

    if endpoint in ["ADME_hPPB", "ADME_rPPB"]:
        if value <= 0.5:
            return "very high binding"
        elif value <= 5:
            return "intermediate binding"
        else:
            return "low binding"

    if endpoint == "ADME_MDR1_ER":
        if value <= 2:
            return "non-substrate"
        elif value <= 5:
            return "moderate efflux"
        else:
            return "strong efflux"

    if endpoint == "ADME_Sol":
        if value < 10:
            return "poor solubility"
        elif value < 100:
            return "moderate solubility"
        else:
            return "high solubility"

    return "unknown"


# ==============================
# PROMPT BUILDER
# ==============================
def build_prompt_local(
    features,
    shap_values,
    endpoint,
    prediction,
    base_value,
):

    property_desc, unit = get_endpoint_context(endpoint)
    category = discretize(endpoint, prediction)

    # =========================
    # GLOBAL DIRECTION (CLAVE)
    # =========================
    if prediction > base_value:
        direction = "higher"
    else:
        direction = "lower"

    # =========================
    # TOP 3 FEATURES
    # =========================
    features = features[:3]
    shap_values = shap_values[:3]

    feature_lines = []
    for i, (f, v) in enumerate(zip(features, shap_values), 1):
        sign = "+" if v > 0 else ""
        feature_lines.append(f"{i}. {f} ({sign}{round(v, 2)})")

    feature_text = "\n".join(feature_lines)

    # =========================
    # PROMPT
    # =========================
    prompt = f"""
You are a biomedical and pharmacokinetics expert.

The model predicts {property_desc} ({endpoint}).

Baseline: {round(base_value, 2)} {unit}
Prediction: {round(prediction, 2)} {unit}
Category: {category}

The final prediction is {direction} than the baseline.

Top contributing features:
{feature_text}

Write a clear scientific explanation with this structure:

1. First paragraph:
- One sentence stating prediction, category and pharmacological meaning

2. Second paragraph:
- Explain the baseline as an average value
- Clearly state that the prediction is {direction} than the baseline
- Explain the biological implication

3. Feature section:
- Write EXACTLY 3 numbered bullet points
- Each bullet must follow this format:

1. FeatureName (+/-value): explanation in ONE paragraph

- Explain:
  • what the feature represents  
  • why it increases/decreases the property  
  • pharmacokinetic interpretation  

IMPORTANT (CRITICAL LOGIC RULES):
- The overall prediction is {direction} than the baseline
- A positive value increases the prediction
- A negative value decreases the prediction
- If all listed features are positive but the prediction is lower:
  → they partially counteract the decrease
- If all listed features are negative but the prediction is higher:
  → they partially counteract the increase
- DO NOT claim that positive features explain a decrease
- DO NOT contradict the global direction

4. Final sentence:
- Explain whether these features reinforce or counteract the overall prediction
- Mention that other features (not shown) may dominate if needed

STRICT RULES:
- ALWAYS include numeric SHAP values (e.g., +16.73)
- ALWAYS use numbered bullet points (1, 2, 3)
- NO sub-bullets
- Each feature = ONE paragraph
- DO NOT repeat the baseline explanation
- DO NOT mention SHAP or machine learning
- Use confident scientific language

Start directly.
"""

    return prompt
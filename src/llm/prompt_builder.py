def build_prompt(features, shap_values, strategy="basic"):

    feature_text = ""

    for f, v in zip(features, shap_values):
        feature_text += f"- {f}: {v:.3f}\n"

    if strategy == "basic":

        prompt = f"""
Explain the prediction using the following feature contributions.

{feature_text}
"""

    elif strategy == "biomedical":

        prompt = f"""
You are a biomedical expert.

A machine learning model predicted a drug property.

The most influential molecular descriptors were:

{feature_text}

Explain in biomedical and chemical terms why these descriptors influence the prediction.
Avoid mentioning SHAP or machine learning terminology.
"""

    elif strategy == "reasoning":

        prompt = f"""
You are a biomedical expert.

Explain step by step why the model predicted this drug property.

Step 1: Identify the most important descriptors  
Step 2: Explain their biochemical meaning  
Step 3: Explain how they influence the prediction  

Descriptors:

{feature_text}
"""

    else:
        raise ValueError("Unknown prompting strategy")

    return prompt
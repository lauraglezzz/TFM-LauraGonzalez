import os
import glob
import pandas as pd


# ==============================
# PATH
# ==============================

RESULTS_PATH = "reports/results/llm"


# ==============================
# CHEMICAL TERMS
# ==============================

CHEM_TERMS = [
    "hydrogen",
    "bond",
    "lipophilicity",
    "polarizability",
    "permeability",
    "binding",
    "solubility",
    "electronic",
    "charge",
    "membrane",
    "interaction",
    "affinity",
]


# ==============================
# METRICS
# ==============================

def explanation_length(text):
    return len(str(text).split())


def count_chem_terms(text):

    text = str(text).lower()

    count = 0

    for term in CHEM_TERMS:
        if term in text:
            count += 1

    return count


# ==============================
# LOAD ALL CSV
# ==============================

def load_all_results():

    files = glob.glob(f"{RESULTS_PATH}/*.csv")

    dfs = []

    for f in files:

        df = pd.read_csv(f)

        df["file"] = os.path.basename(f)

        dfs.append(df)

    all_results = pd.concat(dfs, ignore_index=True)

    return all_results


# ==============================
# ADD METRICS
# ==============================

def compute_metrics(df):

    df["explanation_length"] = df["explanation"].apply(explanation_length)

    df["chem_terms"] = df["explanation"].apply(count_chem_terms)

    return df


# ==============================
# AGGREGATE RESULTS
# ==============================

def aggregate_results(df):

    summary = df.groupby(["strategy", "top_k"]).agg(
        avg_length=("explanation_length", "mean"),
        avg_chem_terms=("chem_terms", "mean"),
        samples=("explanation", "count"),
    )

    return summary.reset_index()


# ==============================
# SAVE RESULTS
# ==============================

def save_results(summary):

    os.makedirs("reports/results/analysis", exist_ok=True)

    output = "reports/results/analysis/llm_experiment_summary.csv"

    summary.to_csv(output, index=False)

    print("\nSaved analysis →", output)


# ==============================
# MAIN
# ==============================

def run_evaluation():

    print("\n==============================")
    print("Evaluating LLM explanations")
    print("==============================")

    df = load_all_results()

    df = compute_metrics(df)

    summary = aggregate_results(df)

    print("\nExperiment Results\n")
    print(summary)

    save_results(summary)


if __name__ == "__main__":
    run_evaluation()
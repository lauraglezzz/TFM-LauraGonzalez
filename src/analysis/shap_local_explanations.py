import os
import sys
from pathlib import Path
import json
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

# =========================================================
# FIND ROOT (TFM)
# =========================================================
ROOT = Path(__file__).resolve()
while ROOT.name != "TFM":
    ROOT = ROOT.parent

sys.path.append(str(ROOT))

# =========================================================
# PATHS
# =========================================================
DATA_PATH = ROOT / "data/processed"
SDF_PATH = ROOT / "data/raw"
MODELS_PATH = ROOT / "models"
OUTPUT_PATH = ROOT / "reports/shap_local_v2"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 20

# =========================================================
# LOAD SDF
# =========================================================
def load_sdf_as_df(sdf_file):

    supplier = Chem.SDMolSupplier(str(sdf_file))

    data = []
    mols = []

    for mol in supplier:
        if mol is None:
            continue

        props = mol.GetPropsAsDict()

        smiles = props.get("SMILES", Chem.MolToSmiles(mol))
        mol_id = props.get("Vendor ID", None)

        data.append({
            "smiles": smiles,
            "id": mol_id
        })

        mols.append(mol)

    df = pd.DataFrame(data)
    return df, mols


# =========================================================
# MAIN
# =========================================================
def run_all_local_shap():

    print("\n==============================")
    print("Running LOCAL SHAP (FIXED)")
    print("==============================")

    for csv_file in DATA_PATH.glob("*_rdkit.csv"):

        endpoint = csv_file.stem.replace("_rdkit", "")
        print(f"\nProcessing {endpoint}")

        # =========================
        # LOAD DATA
        # =========================
        df = pd.read_csv(csv_file)

        # 👉 quitar target si existe
        if "target" in df.columns:
            X = df.drop(columns=["target"])
        else:
            X = df.copy()

        # 👉 QUITAR columnas problemáticas (ESTO ES EL FIX)
        X = X.select_dtypes(include=["number"])

        # =========================
        # LOAD MODEL
        # =========================
        model_path = MODELS_PATH / f"{endpoint}_lightgbm.pkl"

        if not model_path.exists():
            print(f"Model not found for {endpoint}")
            continue

        model = joblib.load(model_path)

        # 👉 ALINEAR FEATURES (ESTO ARREGLA EL ERROR)
        model_features = model.feature_name_

        try:
            X = X[model_features]
        except Exception as e:
            print("Feature mismatch:")
            print("Missing:", set(model_features) - set(X.columns))
            print("Extra:", set(X.columns) - set(model_features))
            continue

        # =========================
        # FIND SDF
        # =========================
        possible_sdf = list(SDF_PATH.glob(f"{endpoint}*.sdf"))

        if not possible_sdf:
            print(f"No SDF found for {endpoint}")
            continue

        sdf_file = possible_sdf[0]
        print(f"Using SDF: {sdf_file.name}")

        sdf_df, mols = load_sdf_as_df(sdf_file)

        if len(sdf_df) != len(X):
            print(f"Size mismatch: CSV={len(X)} vs SDF={len(sdf_df)}")
            continue

        explainer = shap.TreeExplainer(model)

        # =========================
        # LOOP
        # =========================
        for i in range(min(len(X), MAX_SAMPLES)):

            mol_id = sdf_df.iloc[i]["id"]
            sample_name = f"mol_{i}_{mol_id}"

            sample_path = OUTPUT_PATH / endpoint / sample_name
            sample_path.mkdir(parents=True, exist_ok=True)

            row = X.iloc[[i]]

            # =========================
            # PRED
            # =========================
            pred = float(model.predict(row)[0])

            # =========================
            # SHAP
            # =========================
            shap_values = explainer.shap_values(row)
            shap_vals = shap_values[0]
            base_value = float(explainer.expected_value)

            # =========================
            # SAVE JSON
            # =========================
            with open(sample_path / "prediction.json", "w") as f:
                json.dump({
                    "prediction": pred,
                    "base_value": base_value
                }, f, indent=2)

            # =========================
            # TOP FEATURES
            # =========================
            features = X.columns.tolist()

            top_features = sorted(
                [
                    {
                        "feature": feat,
                        "impact": float(val)
                    }
                    for feat, val in zip(features, shap_vals)
                ],
                key=lambda x: abs(x["impact"]),
                reverse=True
            )[:10]

            with open(sample_path / "top_features.json", "w") as f:
                json.dump(top_features, f, indent=2)

            # =========================
            # WATERFALL
            # =========================
            shap_exp = shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=row.iloc[0],
                feature_names=features
            )

            plt.figure()
            shap.plots.waterfall(shap_exp, show=False)
            plt.savefig(sample_path / "waterfall.png", bbox_inches="tight")
            plt.close()

            # =========================
            # MOLECULE IMAGE
            # =========================
            mol = mols[i]

            try:
                img = Draw.MolToImage(mol, size=(300, 300))
                img.save(sample_path / "molecule.png")
            except:
                pass

            print(f"Done: {sample_name}")

    print("\nSHAP generation complete.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_all_local_shap()
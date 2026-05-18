import os
import sys
from pathlib import Path
import json
import numpy as np
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

# REAL SCALE MODELS
MODELS_PATH = ROOT / "models"

# OFFICIAL V2 OUTPUT
OUTPUT_PATH = ROOT / "reports/shap_local_v2"

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# CONFIG
# =========================================================
MAX_SAMPLES = 50


# =========================================================
# LOAD SDF
# =========================================================
def load_sdf_as_df(sdf_file):

    supplier = Chem.SDMolSupplier(
        str(sdf_file)
    )

    data = []
    mols = []

    for mol in supplier:

        if mol is None:
            continue

        props = mol.GetPropsAsDict()

        smiles = props.get(
            "SMILES",
            Chem.MolToSmiles(mol)
        )

        mol_id = props.get(
            "Vendor ID",
            None
        )

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
    print("Running LOCAL SHAP (v2)")
    print("==============================")

    for csv_file in DATA_PATH.glob("*_rdkit.csv"):

        endpoint = csv_file.stem.replace(
            "_rdkit",
            ""
        )

        print(f"\nProcessing {endpoint}")

        # =================================================
        # LOAD DATA
        # =================================================
        df = pd.read_csv(csv_file)

        if "activity" in df.columns:

            X = df.drop(columns=["activity"])

        else:

            X = df.copy()

        # Keep only numeric features
        X = X.select_dtypes(
            include=["number"]
        )

        # =================================================
        # LOAD MODEL
        # =================================================
        model_path = (
            MODELS_PATH /
            f"{endpoint}_lightgbm.pkl"
        )

        if not model_path.exists():

            print(
                f"Model not found for {endpoint}"
            )

            continue

        model = joblib.load(model_path)

        # =================================================
        # ALIGN FEATURES
        # =================================================
        model_features = model.feature_name_

        try:

            X = X[model_features]

        except Exception:

            print("\nFeature mismatch")

            print(
                "Missing:",
                set(model_features)
                - set(X.columns)
            )

            print(
                "Extra:",
                set(X.columns)
                - set(model_features)
            )

            continue

        # =================================================
        # FIND SDF
        # =================================================
        possible_sdf = list(
            SDF_PATH.glob(f"{endpoint}*.sdf")
        )

        if not possible_sdf:

            print(
                f"No SDF found for {endpoint}"
            )

            continue

        sdf_file = possible_sdf[0]

        print(f"Using SDF: {sdf_file.name}")

        sdf_df, mols = load_sdf_as_df(
            sdf_file
        )

        if len(sdf_df) != len(X):

            print(
                f"Size mismatch: CSV={len(X)} vs SDF={len(sdf_df)}"
            )

            continue

        # =================================================
        # PREDICTIONS
        # =================================================
        preds = model.predict(X)

        pred_df = pd.DataFrame({
            "idx": np.arange(len(preds)),
            "prediction": preds
        })

        # Sort predictions
        pred_df = pred_df.sort_values(
            "prediction"
        )

        # Uniform sampling across prediction range
        selected_idx = np.linspace(
            0,
            len(pred_df) - 1,
            min(MAX_SAMPLES, len(pred_df)),
            dtype=int
        )

        selected_rows = pred_df.iloc[
            selected_idx
        ]

        selected_indices = selected_rows[
            "idx"
        ].tolist()

        print(
            f"Selected {len(selected_indices)} molecules"
        )

        # =================================================
        # SHAP EXPLAINER
        # =================================================
        explainer = shap.TreeExplainer(
            model
        )

        # =================================================
        # LOOP
        # =================================================
        for i in selected_indices:

            mol_id = sdf_df.iloc[i]["id"]

            sample_name = (
                f"mol_{i}_{mol_id}"
            )

            sample_path = (
                OUTPUT_PATH /
                endpoint /
                sample_name
            )

            sample_path.mkdir(
                parents=True,
                exist_ok=True
            )

            row = X.iloc[[i]]

            # =============================================
            # PREDICTION
            # =============================================
            pred = float(
                model.predict(row)[0]
            )

            # =============================================
            # SHAP VALUES
            # =============================================
            shap_values = explainer.shap_values(
                row
            )

            shap_vals = shap_values[0]

            base_value = float(
                explainer.expected_value
            )

            # =============================================
            # SAVE PREDICTION
            # =============================================
            with open(
                sample_path / "prediction.json",
                "w"
            ) as f:

                json.dump({
                    "prediction": pred,
                    "base_value": base_value
                }, f, indent=2)

            # =============================================
            # TOP FEATURES
            # =============================================
            features = X.columns.tolist()

            top_features = sorted(
                [
                    {
                        "feature": feat,
                        "impact": float(val)
                    }

                    for feat, val in zip(
                        features,
                        shap_vals
                    )
                ],
                key=lambda x: abs(x["impact"]),
                reverse=True
            )[:10]

            with open(
                sample_path / "top_features.json",
                "w"
            ) as f:

                json.dump(
                    top_features,
                    f,
                    indent=2
                )

            # =============================================
            # WATERFALL PLOT
            # =============================================
            shap_exp = shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=row.iloc[0],
                feature_names=features
            )

            plt.figure()

            shap.plots.waterfall(
                shap_exp,
                show=False
            )

            plt.savefig(
                sample_path / "waterfall.png",
                bbox_inches="tight"
            )

            plt.close()

            # =============================================
            # MOLECULE IMAGE
            # =============================================
            mol = mols[i]

            try:

                img = Draw.MolToImage(
                    mol,
                    size=(300, 300)
                )

                img.save(
                    sample_path / "molecule.png"
                )

            except:
                pass

            print(f"Done: {sample_name}")

    print("\nSHAP generation complete.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    run_all_local_shap()
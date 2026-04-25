from pathlib import Path
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import json

# =========================================================
# Paths
# =========================================================
PROCESSED_PATH = Path("data/processed")
RAW_PATH = Path("data/raw")
MODELS_PATH = Path("models")
OUTPUT_PATH = Path("reports/shap_local")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =========================================================
# Endpoints
# =========================================================
ENDPOINTS = [
    "ADME_HLM",
    "ADME_hPPB",
    "ADME_MDR1_ER",
    "ADME_RLM",
    "ADME_rPPB",
    "ADME_Sol",
]

# =========================================================
# Load SMILES + ID
# =========================================================
def load_metadata_from_sdf(sdf_path):
    supplier = Chem.SDMolSupplier(str(sdf_path))

    smiles_list = []
    id_list = []

    for i, mol in enumerate(supplier):

        if mol is None:
            smiles_list.append(None)
            id_list.append(f"mol_{i}")
            continue

        # SMILES
        if mol.HasProp("SMILES"):
            smiles = mol.GetProp("SMILES")
        else:
            smiles = Chem.MolToSmiles(mol)

        smiles_list.append(smiles)

        # ID robusto
        if mol.HasProp("Vendor ID"):
            mol_id = mol.GetProp("Vendor ID")
        elif mol.HasProp("Internal ID"):
            mol_id = mol.GetProp("Internal ID")
        elif mol.HasProp("ID"):
            mol_id = mol.GetProp("ID")
        elif mol.HasProp("_Name"):
            mol_id = mol.GetProp("_Name")
        else:
            mol_id = f"mol_{i}"

        id_list.append(mol_id)

    return smiles_list, id_list


# =========================================================
# Main
# =========================================================
def run_local_shap(endpoint_name):

    print(f"\nRunning LOCAL SHAP for {endpoint_name}")

    data_path = PROCESSED_PATH / f"{endpoint_name}_rdkit.csv"
    model_path = MODELS_PATH / f"{endpoint_name}_lightgbm.pkl"
    sdf_path = RAW_PATH / f"{endpoint_name}.sdf"

    if not data_path.exists() or not model_path.exists():
        print(f"Skipping {endpoint_name}")
        return

    df = pd.read_csv(data_path)

    # =========================
    # SMILES + ID
    # =========================
    if sdf_path.exists():
        smiles_list, id_list = load_metadata_from_sdf(sdf_path)
        df["smiles"] = smiles_list[:len(df)]
        df["mol_id"] = id_list[:len(df)]
    else:
        df["smiles"] = None
        df["mol_id"] = [f"idx_{i}" for i in range(len(df))]

    # =========================
    # Features
    # =========================
    X = df.drop(columns=["activity", "smiles", "mol_id"], errors="ignore")

    model = joblib.load(model_path)

    # =========================
    # SHAP 
    # =========================
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    #  predictions
    preds = model.predict(X)

    # =========================
    # Output folder
    # =========================
    endpoint_out = OUTPUT_PATH / endpoint_name
    endpoint_out.mkdir(parents=True, exist_ok=True)

    # =========================
    # LOOP ALL SAMPLES
    # =========================
    for idx in range(len(df)):

        mol_id = str(df.iloc[idx].get("mol_id", f"idx_{idx}"))

        print(f"  Processing idx={idx} (ID={mol_id})")

        sample_out = endpoint_out / f"mol_{idx}_{mol_id}"
        sample_out.mkdir(parents=True, exist_ok=True)

        # prediction for this molecule
        pred_value = float(preds[idx])

        # =========================
        # Waterfall
        # =========================
        try:
            plt.figure()
            shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
            plt.tight_layout()
            plt.savefig(sample_out / "waterfall.png", dpi=120)
            plt.close()
        except Exception as e:
            print(f"Waterfall error: {e}")

        # =========================
        # Molecule image
        # =========================
        smiles = df.iloc[idx].get("smiles")

        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    img.save(sample_out / "molecule.png")
            except:
                pass

        # =========================
        # TOP-3 SHAP
        # =========================
        values = shap_values.values[idx]
        feature_names = X.columns

        top_idx = np.argsort(np.abs(values))[-3:]

        top_features = [
            {
                "feature": feature_names[i],
                "impact": float(values[i])
            }
            for i in top_idx[::-1]
        ]

        # =========================
        # Save features
        # =========================
        with open(sample_out / "top_features.json", "w") as f:
            json.dump(top_features, f, indent=2)

        pd.DataFrame(top_features).to_csv(
            sample_out / "top_features.csv",
            index=False
        )

        # save prediction
        with open(sample_out / "prediction.json", "w") as f:
            json.dump({"prediction": pred_value}, f, indent=2)


# =========================================================
# RUN ALL
# =========================================================
def run_all_local_shap():
    for endpoint in ENDPOINTS:
        run_local_shap(endpoint)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_all_local_shap()
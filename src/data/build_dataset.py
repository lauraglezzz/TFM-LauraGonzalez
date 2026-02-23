# src/data/build_dataset.py

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd


# ==============================
# Paths
# ==============================

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


# ==============================
# Endpoint → Property mapping
# ==============================

ENDPOINTS = {
    "ADME_HLM": "LOG HLM_CLint (mL/min/kg)",
    "ADME_RLM": "LOG RLM_CLint (mL/min/kg)",
    "ADME_MDR1_ER": "LOG MDR1-MDCK ER (B-A/A-B)",
    "ADME_Sol": "LOG SOLUBILITY PH 6.8 (ug/mL)",
    "ADME_hPPB": "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)",
    "ADME_rPPB": "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)",
}


# ==============================
# Molecule standardization
# ==============================

def standardize_molecule(mol):
    try:
        clean = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(clean)
        uncharger = rdMolStandardize.Uncharger()
        uncharged = uncharger.uncharge(parent)
        te = rdMolStandardize.TautomerEnumerator()
        return te.Canonicalize(uncharged)
    except Exception:
        return mol


# ==============================
# Descriptor computation (ALL RDKit descriptors)
# ==============================

def compute_rdkit_descriptors(mol):
    descriptor_names = []
    descriptor_values = []

    for name, func in Descriptors.descList:
        try:
            value = func(mol)
        except Exception:
            value = None

        descriptor_names.append(name)
        descriptor_values.append(value)

    return descriptor_names, descriptor_values


# ==============================
# Build single dataset
# ==============================

def build_dataset(sdf_filename: str, target_property: str):

    sdf_path = RAW_DATA_PATH / sdf_filename

    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    supplier = Chem.SDMolSupplier(str(sdf_path))

    rows = []
    descriptor_names = None

    for mol in supplier:
        if mol is None:
            continue

        mol = standardize_molecule(mol)

        try:
            activity = float(mol.GetProp(target_property))
        except Exception:
            continue

        names, values = compute_rdkit_descriptors(mol)

        if descriptor_names is None:
            descriptor_names = names

        rows.append(values + [activity])

    if len(rows) == 0:
        raise ValueError(f"No valid molecules found in {sdf_filename}")

    columns = descriptor_names + ["activity"]

    df = pd.DataFrame(rows, columns=columns)

    # Remove descriptors that are entirely NaN
    df = df.dropna(axis=1, how="all")

    output_file = PROCESSED_DATA_PATH / f"{sdf_filename.replace('.sdf','')}_rdkit.csv"
    df.to_csv(output_file, index=False)

    print(f"\nSaved dataset to {output_file}")
    print(f"Shape: {df.shape}")


# ==============================
# Build ALL datasets
# ==============================

def build_all_datasets():
    """
    Build all ADME endpoint datasets automatically.
    """

    for endpoint_name, property_name in ENDPOINTS.items():

        print(f"\n==============================")
        print(f"Building {endpoint_name}")
        print(f"==============================")

        sdf_filename = f"{endpoint_name}.sdf"
        build_dataset(sdf_filename, property_name)

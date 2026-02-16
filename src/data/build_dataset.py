# src/data/build_dataset.py

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd


RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


def standardize_molecule(mol):
    try:
        clean = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(clean)
        uncharger = rdMolStandardize.Uncharger()
        uncharged = uncharger.uncharge(parent)
        te = rdMolStandardize.TautomerEnumerator()
        return te.Canonicalize(uncharged)
    except:
        return mol


def compute_rdkit_descriptors(mol):
    desc = []

    desc.append(rdMolDescriptors.CalcTPSA(mol))
    desc.append(rdMolDescriptors.CalcFractionCSP3(mol))
    desc.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
    desc.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
    desc.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
    desc.append(rdMolDescriptors.CalcNumAmideBonds(mol))
    desc.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
    desc.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
    desc.append(rdMolDescriptors.CalcNumAromaticRings(mol))
    desc.append(rdMolDescriptors.CalcNumLipinskiHBA(mol))
    desc.append(rdMolDescriptors.CalcNumLipinskiHBD(mol))
    desc.append(rdMolDescriptors.CalcNumHeteroatoms(mol))
    desc.append(rdMolDescriptors.CalcNumRings(mol))
    desc.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
    desc.append(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
    desc.append(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
    desc.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
    desc.append(rdMolDescriptors.CalcHallKierAlpha(mol))
    desc.append(rdMolDescriptors.CalcKappa1(mol))
    desc.append(rdMolDescriptors.CalcKappa2(mol))
    desc.append(rdMolDescriptors.CalcKappa3(mol))
    desc.append(rdMolDescriptors.CalcChi0n(mol))
    desc.append(rdMolDescriptors.CalcChi0v(mol))
    desc.append(rdMolDescriptors.CalcChi1n(mol))
    desc.append(rdMolDescriptors.CalcChi1v(mol))
    desc.append(rdMolDescriptors.CalcChi2n(mol))
    desc.append(rdMolDescriptors.CalcChi2v(mol))
    desc.append(rdMolDescriptors.CalcChi3n(mol))
    desc.append(rdMolDescriptors.CalcChi3v(mol))
    desc.append(rdMolDescriptors.CalcChi4n(mol))
    desc.append(rdMolDescriptors.CalcChi4v(mol))
    desc.append(rdMolDescriptors.CalcAsphericity(mol))
    desc.append(rdMolDescriptors.CalcEccentricity(mol))
    desc.append(rdMolDescriptors.CalcInertialShapeFactor(mol))
    desc.append(rdMolDescriptors.CalcExactMolWt(mol))
    desc.append(rdMolDescriptors.CalcPBF(mol))
    desc.append(rdMolDescriptors.CalcPMI1(mol))
    desc.append(rdMolDescriptors.CalcPMI2(mol))
    desc.append(rdMolDescriptors.CalcPMI3(mol))
    desc.append(rdMolDescriptors.CalcRadiusOfGyration(mol))
    desc.append(rdMolDescriptors.CalcSpherocityIndex(mol))
    desc.append(rdMolDescriptors.CalcLabuteASA(mol))
    desc.append(rdMolDescriptors.CalcNPR1(mol))
    desc.append(rdMolDescriptors.CalcNPR2(mol))

    desc.extend(rdMolDescriptors.PEOE_VSA_(mol))
    desc.extend(rdMolDescriptors.SMR_VSA_(mol))
    desc.extend(rdMolDescriptors.SlogP_VSA_(mol))
    desc.extend(rdMolDescriptors.MQNs_(mol))
    desc.extend(rdMolDescriptors.CalcCrippenDescriptors(mol))
    desc.extend(rdMolDescriptors.CalcAUTOCORR2D(mol))

    return desc


def build_dataset(sdf_filename: str, target_property: str):
    sdf_path = RAW_DATA_PATH / sdf_filename
    supplier = Chem.SDMolSupplier(str(sdf_path))

    rows = []

    for mol in supplier:
        if mol is None:
            continue

        mol = standardize_molecule(mol)

        try:
            activity = float(mol.GetProp(target_property))
        except:
            continue

        descriptors = compute_rdkit_descriptors(mol)
        rows.append(descriptors + [activity])

    n_features = len(rows[0]) - 1
    columns = [f"rdMD_{i+1}" for i in range(n_features)] + ["activity"]

    df = pd.DataFrame(rows, columns=columns)

    output_file = PROCESSED_DATA_PATH / f"{sdf_filename.replace('.sdf','')}_rdkit.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved dataset to {output_file}")
    print(f"Shape: {df.shape}")
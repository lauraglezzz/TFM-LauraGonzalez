# scripts/build_all.py

from src.data.build_dataset import build_dataset

endpoints = {
    "ADME_HLM.sdf": "LOG HLM_CLint (mL/min/kg)",
    "ADME_RLM.sdf": "LOG RLM_CLint (mL/min/kg)",
    "ADME_hPPB.sdf": "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)",
    "ADME_rPPB.sdf": "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)",
    "ADME_MDR1_ER.sdf": "LOG MDR1-MDCK ER (B-A/A-B)",
    "ADME_Sol.sdf": "LOG SOLUBILITY PH 6.8 (ug/mL)",
}

for sdf, target in endpoints.items():
    build_dataset(sdf, target)
# ADME Prediction with Explainable AI

**TFM — Master en Data Science, Laura González**

Predicting ADME (Absorption, Distribution, Metabolism, Excretion) drug properties using LightGBM models with SHAP-based explanations and LLM-generated natural language summaries.

---

## Overview

This project builds interpretable ML models for 6 ADME endpoints and evaluates the quality of LLM-generated explanations using an LLM-as-a-Judge framework with Claude (Anthropic) as the independent judge.

**Endpoints:** HLM, RLM, hPPB, rPPB, MDR1-ER, Solubility  
**Compounds:** 3,521 drug-like molecules  
**Features:** 217 RDKit molecular descriptors

---

## Results

### Model Performance

| Endpoint | Task | Pearson r / Accuracy |
|----------|------|----------------------|
| HLM | Regression | — |
| RLM | Regression | — |
| hPPB | Regression | — |
| rPPB | Regression | — |
| MDR1-ER | Classification | — |
| Solubility | Classification | — |

### LLM-as-a-Judge Evaluation

1,500 pairwise comparisons across 6 models (100 per pair × 15 pairs).  
Judge: **Claude Haiku 4.5** (Anthropic) — independent model family, no self-preference bias.

| Model | Wins (total) | Win Rate (head-to-head) |
|-------|-------------|------------------------|
| gpt-4.1-mini | 483 | **96.6%** |
| gpt-4o-mini | 365 | 73.0% |
| mistral | 266 | 53.2% |
| gemma | 150 | 30.0% |
| llama3 | 140 | 28.0% |
| phi3 | 96 | 19.2% |

Full results: [`reports/results/llm/judge/judge_summary.csv`](reports/results/llm/judge/judge_summary.csv)

---

## Project Structure

```
├── scripts/
│   ├── main.py                        # Full pipeline (train → SHAP → LLM)
│   └── check_judge.py                 # Monitor LLM judge progress
│
├── src/
│   ├── modeling/
│   │   ├── train_lightgbm.py          # LightGBM regressor training
│   │   ├── train_lightgbm_classifier.py
│   │   └── shap_analysis.py           # SHAP values via TreeExplainer
│   ├── analysis/
│   │   ├── shap_local_explanations.py # Per-molecule SHAP summaries
│   │   └── generate_local_llm_explanations.py
│   └── llm/
│       ├── core/llm_client.py         # Unified client (OpenAI, Gemini, Anthropic, Ollama)
│       └── evaluation/run_llm_judge.py # LLM-as-a-Judge pairwise evaluation
│
├── reports/
│   ├── results/llm/judge/             # Judge results and summary
│   ├── shap_local/                    # Per-molecule SHAP reports
│   └── llm_local/                     # Per-molecule LLM explanations
│
└── data/processed/                    # Processed molecular datasets
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

Required keys in `.env`:
```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
CLAUDE_API_KEY=...
```

## Run

```bash
# Full pipeline
python scripts/main.py

# LLM judge evaluation
python -m src.llm.evaluation.run_llm_judge

# Check judge progress
python scripts/check_judge.py
```

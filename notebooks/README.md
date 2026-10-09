# Notebook sequence

Create and run these in order. Each corresponds to a phase in `PROJECT_GUIDE.md`.

| Notebook | Phase | Purpose |
|---|---|---|
| `00_variable_shortlist.md` | 1 | Candidate BRFSS variables + codebook meanings |
| `01_data_cleaning.ipynb` | 2 | Load raw XPT, apply `src/data_loader.py`, save cleaned parquet |
| `02_eda.ipynb` | 3 | Target distribution, missingness, correlations, exported figures |
| `03_feature_engineering.ipynb` | 4 | Encoding, derived features, `src/preprocessing.py` build |
| `04_feature_selection.ipynb` | 6 | Compare correlation/chi2/MI/RFE/L1/importance-based selection |
| `05_model_training.ipynb` | 7–9 | Baseline models, tuning, imbalance handling, ensembles |
| `06_shap.ipynb` | 12 | Global + local SHAP explanations on the final model |
| `07_robustness_fairness.ipynb` | 10 | Missingness/noise sweep, calibration, subgroup fairness |
| `08_final_evaluation.ipynb` | 11 | One-shot scoring on the held-out test set |

Keep heavy reusable logic in `src/` and call it from these notebooks — notebooks should read like a narrative, not contain your whole pipeline inline.

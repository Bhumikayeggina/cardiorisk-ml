# Notebooks

Run in order. Each notebook uses code from `src/` and writes its results to `reports/` (tables, JSON decisions, figures).

| Notebook | Step | What it does | Main outputs |
|---|---|---|---|
| `01_data_cleaning` | Data | Downloads BRFSS 2024, keeps 29 candidate questions, applies per-column recoding (`src/data_loader.py` reproduces it in one command) | `data/processed/cleaned.parquet` |
| `02_eda` | Data | Class balance, missingness, associations with the outcome | figures 01 to 05 |
| `03_feature_engineering` | Features | Leak-safe preprocessing pipeline, stratified 80/20 split | `data/processed/X_train, X_test, y_train, y_test` |
| `04_feature_selection` | Features | Seven selection methods rolled up per question | `reports/feature_selection_summary.csv`, `selected_features.json` |
| `05_baseline_models` | Models | Six model families, two feature sets, 5-fold CV | `reports/baseline_results_*.csv` |
| `06_ablation` | Models | Paired ablations choose the final 14 questions | `reports/final_features.json`, `ablation_results.csv` |
| `07_tuning` | Models | Grid and randomized search for all six models | `reports/best_params.json`, `reports/tuning/` |
| `08_imbalance` | Models | None vs class weighting vs undersampling | `reports/imbalance_results.csv` |
| `09_threshold` | Evaluation | Out-of-fold predictions, calibration, thresholds, decision curve | `data/processed/oof_probabilities.parquet` |
| `10_ensembles` | Evaluation | Voting and stacking; final model and threshold fixed | `reports/ensemble_comparison.csv`, `final_decision.json` |
| `11_final_evaluation` | Evaluation | One-shot test evaluation with bootstrap intervals | `models/final_xgb_pipeline.joblib`, `reports/final_test_results.csv` |
| `12_robustness` | Analysis | Missing, misreported and noisy inputs | `reports/robustness_results.csv` |
| `13_fairness` | Analysis | Subgroup audit by sex, age, income, race/ethnicity | `reports/fairness_audit.csv` |
| `14_shap` | Analysis | Global and individual SHAP explanations | figures 16 to 19 |
| `15_app_assets` | App | Risk-band summary read by the app | `reports/risk_bands.json` |

Keep heavy logic in `src/` and call it from here, so notebooks read like a narrative and the app can reuse the same code.

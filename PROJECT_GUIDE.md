# PROJECT_GUIDE.md: methodology and decision log

How CardioRisk was built, step by step, with the evidence behind each decision. Every number below comes from a notebook in `notebooks/` and a file in `reports/`.

**Setup.** Python 3.10+, a virtual environment, `pip install -r requirements.txt`. Work is split between reusable, tested code in `src/` and narrative notebooks in `notebooks/`.

---

## 1. Data and outcome (notebooks 01 to 02, `src/data_loader.py`)

- **Source:** 2024 CDC BRFSS, downloaded straight from the CDC into memory (nothing but two parquet files is saved). The public file has 457,670 rows and 301 columns. The original synopsis cites 345 variables; reconcile that figure against the codebook before submission.
- **Outcome:** `_MICHD`, ever told coronary heart disease or heart attack. 5,206 respondents with an unknown outcome were dropped, leaving 452,464, of whom 42,338 (9.4%) are positive.
- **Features:** 29 candidates. Blood pressure, cholesterol and sleep variables used by older BRFSS studies are not in the 2024 file; the heavy-drinking flag was renamed `_RFDRHV9`.
- **Cleaning:** every column recoded with its own "don't know / refused" codes (for example `PHYSHLTH` 88 means "none" and becomes 0; BMI is stored x100; `_DRNKWK3` has a 99900 sentinel and floating-point zeros). Duplicates were not removed because identical survey answers are legitimate respondents.
- **EDA:** heart disease prevalence rises from under 1% at ages 18 to 24 to about 17% at 65+, from about 2.5% (excellent health) to about 31% (poor health), and is higher for diabetes (22% vs 7%) and difficulty walking (23% vs 7%).

## 2. Features and split (notebook 03, `src/preprocessing.py`)

- Stratified 80/20 split (361,971 train / 90,493 test, seed 42). The test set stays sealed until the final evaluation.
- One leak-safe pipeline shared by the notebooks and the app: fixed outlier caps (BMI 70, drinks per week 100), median/most-frequent imputation with missing-value flags, scaling, and one-hot encoding with "unknown" as its own category. Every cross-validation fold fits its own preprocessing.

## 3. Feature selection (notebook 04, `src/feature_selection.py`)

Seven methods (correlation, chi-square, mutual information, RFE, L1 logistic regression, Random Forest and XGBoost importance) scored the encoded columns, rolled up to the original questions and averaged by rank. Age, general health, employment status, diabetes, COPD and difficulty walking were top-15 for every method. No pair of selected features was near-duplicate (highest Cramer's V 0.45).

## 4. Baselines and ablations (notebooks 05 to 06)

- Six untuned models, two feature sets, 5-fold CV: logistic regression, Random Forest and XGBoost tied at ROC-AUC of about 0.83. More data barely helped (logistic regression 0.830 at both 30k and 362k rows).
- Paired ablations on identical folds decided the final set:

| Question | Result | Decision |
|---|---|---|
| Does `chronic_count` help? | change of at most 0.0001 ROC-AUC | dropped |
| Does employment status help beyond age? | costs 0.0004 to 0.0008 when removed | dropped |
| 29 to 15 questions | costs about 0.002 ROC-AUC | accepted |
| 15 to 10 questions | costs a further 0.002 to 0.003 | stopped at 14 |

**Final feature set:** 14 questions (age group, sex, general health, diabetes, arthritis, smoking, COPD, difficulty walking, last checkup, income, days of poor physical health, kidney disease, marital status, cancer other than skin cancer).

## 5. Tuning (notebook 07, `src/tuning.py`)

Grid and randomized search optimising PR-AUC, with the untuned baseline scored on identical rows and folds. Only KNN gained clearly (PR-AUC 0.279 to 0.316, k = 200); logistic regression, Random Forest and the SVM gained nothing, and XGBoost's gain (+0.005) was within one fold standard deviation.

## 6. Class imbalance (notebook 08, `src/imbalance.py`)

Class weighting and undersampling (SMOTE was skipped because inputs are mostly categorical) left ROC-AUC unchanged but moved recall at a 0.5 cutoff from 1 to 8% to 75 to 79%, and raised Brier score from about 0.073 to 0.11 to 0.18. **Decision:** train unweighted to keep probabilities calibrated, and set sensitivity with a threshold. Class weights also explained the weak SVM (ROC-AUC 0.654 to 0.810).

## 7. Calibration and threshold (notebook 09, `src/thresholds.py`)

Out-of-fold predictions on the full training set. All three leading models were already well calibrated (mean predicted risk equals prevalence; Brier about 0.072 vs 0.085 for a constant forecast). The threshold giving 80% out-of-fold recall (0.093) was fixed before the test set was opened; the max-F2 threshold (0.100) agrees.

## 8. Ensembles and final model (notebook 10, `src/ensembles.py`)

Soft voting and stacking built from out-of-fold predictions gained at most 0.0002 ROC-AUC and 0.0005 PR-AUC over XGBoost, and the stacker gave logistic regression a weight of about zero. **Decision:** tuned, unweighted XGBoost on 14 questions.

## 9. Final evaluation (notebook 11, `src/final_eval.py`)

One-shot test evaluation with 500-resample bootstrap intervals: ROC-AUC 0.832, PR-AUC 0.341, Brier 0.0717, recall 0.798 at the fixed threshold, matching the out-of-fold estimates. The fitted pipeline is saved as `models/final_xgb_pipeline.joblib`.

## 10. Robustness, fairness and explainability (notebooks 12 to 14)

- **Robustness (`src/robustness.py`):** ROC-AUC falls about 0.001 per percentage point of answers missing; the share flagged stays near 34%. Misreported and noisy answers barely matter. Leaving a whole question blank can shift scores, so the app requires every answer.
- **Fairness (`src/fairness.py`):** calibrated within sex, age and income; a single cutoff flags very different shares by age (0.2% at 18 to 24 vs 67.5% at 65+) and sex (27% of women vs 41% of men); risk is slightly over-estimated for some racial and ethnic groups although race is not an input.
- **SHAP (`src/explain.py`):** values rolled up to the 14 questions and verified to reproduce the model output. Age, general health and sex account for 57% of attribution.

## 11. The app (notebook 15, `app/streamlit_app.py`)

A form for the 14 questions (no blank answers allowed), the estimated risk, a Low / Moderate / High band with its observed rate in training data, a SHAP bar chart of what drove the estimate, and the limitations. Answer labels were checked against the 2024 codebook. Deployed on Streamlit Community Cloud from `app/streamlit_app.py` with the pinned `app/requirements.txt`.

---

## Decision log

| Decision | Evidence | Where |
|---|---|---|
| Drop duplicate-row removal | identical survey answers are legitimate respondents | 01 |
| Impute inside the pipeline, not before the split | prevents leakage | 03 |
| 14 questions, no derived feature | paired ablations | 06 |
| Tune on subsamples | more data changed ROC-AUC by at most 0.004 | 05, 07 |
| Train unweighted, set a threshold | weights raise Brier 0.073 to 0.17 with no ranking gain | 08, 09 |
| Single XGBoost, not an ensemble | ensembles +0.0005 PR-AUC at best | 10 |
| Threshold at 80% recall (0.093) | fixed on out-of-fold training predictions | 09, 10 |
| Require all 14 answers in the app | a blank question shifts scores | 12, 15 |

## Limitations

Self-reported outcome from one survey year; no blood pressure or cholesterol; high sensitivity means many false alarms at 9.4% prevalence; subgroup differences exist; robustness was tested with random missingness only. See the README for the full list.

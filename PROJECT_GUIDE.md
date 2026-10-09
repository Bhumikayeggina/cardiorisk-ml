# PROJECT_GUIDE.md — Build CardioRisk from Zero

This is a full walkthrough, phase by phase. Each phase says **what to do**, **how to do it**, and **what "done" looks like**. Work through it in order — each phase builds on the last. Budget roughly 3–4 weeks part-time across a team of 3.

---

## Phase 0 — Environment & Repo Setup (Day 1)

**What to do:** Get your tools installed and your repo skeleton pushed to GitHub before touching data.

**How to do it:**
1. Install Python 3.10+ and Git if you don't have them.
2. Create a GitHub repo, e.g. `cardiorisk-ml`. Clone it locally.
3. Copy this whole scaffold (`README.md`, `PROJECT_GUIDE.md`, `requirements.txt`, `.gitignore`, and the folders) into the repo.
4. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
5. Commit: `git add . && git commit -m "Initial project scaffold" && git push`
6. Set up a shared task board (GitHub Projects, Trello, or Notion) with the phases below as columns, so all three of you know who owns what.

**Done when:** repo exists on GitHub, environment installs cleanly, `.gitignore` is in place so you never accidentally commit the raw 457K-row dataset (it's too large for GitHub anyway).

---

## Phase 1 — Get the Data (Day 1–2)

**What to do:** Download the 2024 BRFSS dataset and the variable documentation.

**How to do it:**
1. Use `src/data_loader.py`'s `download_and_load_brfss()` to pull the data directly from the CDC — no manual browser download needed:
   ```python
   from src.data_loader import download_and_load_brfss, download_codebook

   df = download_and_load_brfss()          # fetches + unzips + parses in one call
   df.to_parquet("data/raw/full_2024.parquet")
   print(df.shape)                          # (457670, 345)

   download_codebook()                      # saves data/raw/codebook24.zip — unzip and keep the PDF open for reference
   ```
   If the request times out or 403s, try the `www.` prefix on the URL (network-dependent).
2. From now on, load `data/raw/full_2024.parquet` (seconds) instead of re-downloading.
3. Read the codebook alongside the data. You specifically need to look up:
   - `_MICHD` — your target variable (ever reported coronary heart disease or MI)
   - Candidate features: `_AGE_G`, `SEXVAR`, `_RACE`, `EDUCA`, `INCOME3`, `GENHLTH`, `_BMI5`, `DIABETE4`, `_RFHYPE6`, `TOLDHI3`, `_SMOKER3`, `_TOTINDA`, `SLEPTIM1`, `_RFDRHV8`, etc. (exact variable names shift slightly year to year — always confirm against the 2024 codebook, not an older one.)
5. Never commit the raw file to GitHub. Add a `data/raw/README.md` that says "download from [CDC link], place here" so anyone cloning the repo can reproduce it.

**Done when:** you have a local Parquet/CSV of the full dataset, and a shortlist of ~25–40 candidate feature variable names with their codebook meanings written down (put this list in `notebooks/00_variable_shortlist.md`).

---

## Phase 2 — Data Cleaning (Week 1)

**What to do:** Turn raw survey codes into a clean, modelable table.

**How to do it (in `notebooks/01_data_cleaning.ipynb`):**
1. Load only your shortlisted columns (no need to load all 345).
2. Drop rows where `_MICHD` is missing/unknown/refused (these are documented as specific codes like 7/9 — check the codebook).
3. For each feature, look up BRFSS's "don't know / refused / missing" sentinel codes (often 7, 9, 77, 99, 777, 999) and convert them to `NaN` rather than treating them as real values — this is the single most common BRFSS mistake.
4. Recode Yes/No fields (usually 1=Yes, 2=No) into 1/0.
5. Deduplicate if needed; check `df.shape` before/after each step and log it.
6. Save the cleaned table to `data/processed/cleaned.parquet`.

**Done when:** a clean dataframe with your ~25–40 features + binary target, no sentinel codes left disguised as numbers, saved to disk.

---

## Phase 3 — Exploratory Data Analysis (Week 1)

**What to do:** Understand the data before modeling it.

**How to do it (in `notebooks/02_eda.ipynb`):**
1. Target distribution — what % have `_MICHD` = 1? (Expect meaningful imbalance, likely 8-12%.) This number directly justifies your Phase 8 class-imbalance work later.
2. Missingness heatmap per feature (`seaborn.heatmap(df.isna())` or a bar of `%missing` per column).
3. Univariate distributions of key numeric fields (BMI, sleep hours, age group) split by target class.
4. A correlation matrix / Cramér's V for categorical associations with the target.
5. 3–5 sentences per chart on what you observe — this becomes your report's EDA section almost verbatim.
6. Export your best 4–6 plots to `reports/figures/` at 300dpi — you'll reuse these in the report and README.

**Done when:** you can state, in one paragraph, which 5-6 features look most associated with CVD risk, with a chart backing each claim.

---

## Phase 4 — Feature Engineering (Week 1–2)

**What to do:** Build the final feature set going into modeling.

**How to do it (`src/preprocessing.py` + `notebooks/03_feature_engineering.ipynb`):**
1. Encode categoricals: one-hot for nominal (race, education tier), ordinal encoding where order is meaningful (general health: excellent→poor).
2. Derive 2–3 justified new features if useful, e.g. a combined "unhealthy lifestyle score" from smoking + inactivity + poor sleep. Only keep ones that measurably help later — say so explicitly in the report either way.
3. Write this as a `sklearn.pipeline.Pipeline`-compatible `ColumnTransformer`, not ad-hoc pandas code — you'll reuse it for train, test, and the Streamlit app.

**Done when:** `src/preprocessing.py` exposes a `build_preprocessor()` function returning a fitted-ready `ColumnTransformer`.

---

## Phase 5 — Train/Test Split (Week 2, quick)

**What to do:** Stratified split, held out until the very end.

**How to do it:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```
Save both splits to `data/processed/`. Never touch `X_test`/`y_test` again until Phase 11's final evaluation — no peeking, no re-splitting.

---

## Phase 6 — Preprocessing Pipeline + Feature Selection (Week 2)

**What to do:** Wrap imputation, scaling, and class handling into a leak-safe pipeline; compare feature-selection methods.

**How to do it (`src/feature_selection.py`, `notebooks/04_feature_selection.ipynb`):**
1. Inside a `Pipeline`, chain: `SimpleImputer` → `ColumnTransformer` (from Phase 4) → model. Fitting the whole pipeline only on `X_train` prevents leakage.
2. Run each feature-selection method (correlation ranking, chi-square, mutual information, RFE with a fast estimator, L1-logistic, Random Forest/XGBoost importances) and record the top-15 features from each.
3. Compare overlap across methods (a simple Venn or overlap table is a nice figure).
4. Pick one final feature set (or keep 2 variants — "full" and "minimal-15" — to directly support the efficiency angle from your synopsis).

**Done when:** you have a justified final feature list and a comparison table/plot of what each selection method chose.

---

## Phase 7 — Model Training + Hyperparameter Tuning (Week 2–3)

**What to do:** Train all 6 base models, then tune the top 3.

**How to do it (`src/models.py`, `notebooks/05_model_training.ipynb`):**
1. Baseline-train all 6 models (LogReg, KNN, SVM, Decision Tree, Random Forest, XGBoost) with default params inside your pipeline, 5-fold `StratifiedKFold` on `X_train` only.
2. Record mean ± std of ROC-AUC, PR-AUC, F1, Recall, Specificity for each.
3. `GridSearchCV`/`RandomizedSearchCV` on SVM (C, kernel, gamma), Random Forest (n_estimators, max_depth, min_samples_split), and XGBoost (n_estimators, learning_rate, max_depth) — always nested inside the training data, never touching `X_test`.
4. Save each fitted pipeline to `models/` with `joblib.dump()`.

**Done when:** a results table (like the one in README.md) filled in for all 6 tuned models.

---

## Phase 8 — Class Imbalance Handling (Week 3, in parallel with Phase 7)

**What to do:** Quantify and address the imbalance you measured in Phase 3.

**How to do it:**
1. Re-run your best 2–3 models with `class_weight='balanced'` and/or SMOTE (`imblearn.over_sampling.SMOTE`) *inside* the CV pipeline (never oversample before splitting — that leaks).
2. Compare recall/specificity/PR-AUC before vs. after.
3. Write one paragraph on which strategy you kept and why — PR-AUC and recall changes are the numbers that matter here, not accuracy.

---

## Phase 9 — Ensemble Learning (Week 3)

**What to do:** Build Voting and Stacking ensembles from your best base models.

**How to do it:**
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
voting = VotingClassifier(estimators=[...], voting='soft')
stacking = StackingClassifier(estimators=[...], final_estimator=LogisticRegression())
```
Evaluate both the same way as Phase 7 and add them to your results table.

---

## Phase 10 — Robustness, Calibration & Fairness (Week 3–4) — your differentiators

**What to do:** This is what separates a resume-worthy project from a standard lab submission.

**How to do it:**
1. **Robustness (`src/robustness.py`):** On the held-out test set, randomly null out 5%/10%/20% of feature values (re-impute using your fitted pipeline) and re-score. Also add small Gaussian noise to numeric features. Plot performance degradation curves.
2. **Calibration:** Plot a reliability diagram (`sklearn.calibration.calibration_curve`) and compute the Brier score for your best model. If poorly calibrated, try `CalibratedClassifierCV`.
3. **Fairness (`src/fairness.py`):** Break down recall and specificity by age band, sex, and income tier on the test set. A simple grouped bar chart showing "recall by subgroup" is a strong, discussion-worthy figure.
4. **Cost-sensitive threshold:** Instead of the default 0.5 cutoff, plot a decision curve (net benefit vs. threshold) and justify a chosen operating threshold given that missed CVD risk (false negative) is costlier than a false alarm.

**Done when:** you have one figure each for robustness degradation, calibration, and subgroup fairness, plus a written threshold justification.

---

## Phase 11 — Final Evaluation (Week 4)

**What to do:** Score your single best model/ensemble on the untouched `X_test` exactly once.

**How to do it:** Load the saved pipeline, predict on `X_test`, compute the full metric suite (accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, confusion matrix), and fill in the README results table for real.

---

## Phase 12 — Explainability with SHAP (Week 4)

**What to do:** Global and local explanations for your final model.

**How to do it (`src/explainability.py`, `notebooks/06_shap.ipynb`):**
1. `shap.TreeExplainer` for tree-based models (fast); `shap.KernelExplainer` or `LinearExplainer` otherwise.
2. Global: `shap.summary_plot()` — save as a figure, this is your single most important chart for the report/resume.
3. Local: pick 2-3 individual test cases (one true positive, one false negative, one clear true negative) and show `shap.force_plot()` or waterfall plots for each, with a sentence explaining what drove that prediction.

---

## Phase 13 — Streamlit App (Week 4)

**What to do:** Build the interactive demo.

**How to do it (`app/streamlit_app.py`):**
1. Load your saved pipeline + model with `joblib.load()`.
2. Build input widgets (`st.slider`, `st.selectbox`) for your final feature set.
3. On submit: run the pipeline, show predicted risk probability, and render a SHAP force/waterfall plot for *that specific input* using `streamlit-shap` or `st.pyplot()`.
4. Add a clear "for academic demonstration only, not medical advice" banner.
5. Test locally: `streamlit run app/streamlit_app.py`

**Deploy it:**
1. Push the repo to GitHub (make sure `models/*.pkl` is committed or regenerable — Git LFS if the model file is large).
2. Go to [share.streamlit.io](https://share.streamlit.io) (or HuggingFace Spaces as an alternative), connect your GitHub repo, point it at `app/streamlit_app.py`, and deploy.
3. Put the live link in your README and resume bullet.

**Done when:** you have a public URL you can click from your phone.

---

## Phase 14 — Report, README, and Resume Polish (Final days)

**What to do:** Package everything for both your grade and your portfolio.

**How to do it:**
1. Fill in the results table in `README.md` for real.
2. Write a short technical blog post (Medium / GitHub Pages / LinkedIn article) summarizing the SHAP findings — "what actually predicts CVD risk in this data." This is a shareable, interview-ready artifact.
3. Write your final resume bullet, e.g.:
   > *Built an end-to-end ML pipeline on 457K-record CDC population health data to predict cardiovascular disease risk; benchmarked 6 classifiers + stacking/voting ensembles achieving [X] ROC-AUC; added calibration, fairness, and robustness analysis; deployed a live SHAP-explained Streamlit demo.*
4. Make sure the GitHub repo README alone (without your lab report) tells a complete story to someone with 60 seconds — problem, data, approach, results table, demo link, key figure.

---

## Team split suggestion (3 people)

- **Person A:** Phases 1–4 (data acquisition, cleaning, EDA, feature engineering)
- **Person B:** Phases 5–9 (pipeline, feature selection, model training, tuning, ensembles)
- **Person C:** Phases 10–13 (robustness/fairness/calibration, SHAP, Streamlit + deployment)
- **All three:** Phase 14 together, and cross-review each other's notebooks before merging to `main`.

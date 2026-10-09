# CardioRisk: An Explainable, Robust ML Framework for Cardiovascular Disease Risk Prediction

An end-to-end machine learning system that predicts cardiovascular disease risk from the **2024 CDC BRFSS** population health survey (457,670 records). Benchmarks 6 classifiers + 2 ensemble strategies, evaluates robustness under noisy/missing inputs, explains predictions with SHAP, and ships as a live interactive risk-screening demo.

> Built as a Fundamentals of ML Lab mini-project by Bhumika Anushka Yeggina, Ginjupalli Tamoghna Sri Ram, and Karimireddy Likith Reddy — MIT Manipal.

## Why this project

Most student heart-disease projects use the 303-row Cleveland dataset. This one uses a real, current, large-scale (457K-record) population health dataset, adds calibration and fairness analysis on top of the standard "compare 6 models" exercise, and deploys as a working app rather than staying in a notebook.

## Results (fill in once trained)

| Model | ROC-AUC | PR-AUC | Recall | Specificity | Brier Score |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| KNN | | | | | |
| SVM | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |
| Voting Ensemble | | | | | |
| Stacking Ensemble | | | | | |

## Project structure

```
cardiorisk-project/
├── data/
│   ├── raw/                # original BRFSS extract (not committed — see data/raw/README.md)
│   └── processed/          # cleaned train/test splits
├── notebooks/               # exploratory + phase-by-phase analysis notebooks
├── src/                     # reusable pipeline code (importable modules)
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── evaluate.py
│   ├── robustness.py
│   ├── explainability.py
│   └── fairness.py
├── models/                   # saved trained model artifacts (.pkl / .json)
├── reports/
│   └── figures/               # exported plots for the report/README
├── app/
│   └── streamlit_app.py       # deployed demo
├── requirements.txt
├── PROJECT_GUIDE.md            # full step-by-step build guide — start here
└── README.md
```

## Quickstart

```bash
git clone <your-repo-url>
cd cardiorisk-project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Place the BRFSS 2024 XPT/CSV extract in data/raw/ (see PROJECT_GUIDE.md Phase 1)
# 2. Run notebooks/ in order (01 → 08), or run the pipeline scripts in src/
# 3. Launch the demo locally
streamlit run app/streamlit_app.py
```

## Full build guide

See **[PROJECT_GUIDE.md](./PROJECT_GUIDE.md)** — a phase-by-phase walkthrough from downloading the data to deploying the demo, written for someone starting from zero.

## Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Matplotlib/Seaborn · Streamlit · Jupyter · Git/GitHub

## Disclaimer

This is an academic risk-screening framework built on self-reported population survey data, not a validated clinical diagnostic tool.

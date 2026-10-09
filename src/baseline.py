"""
Step 6: untuned baseline models + a cross-validation helper.

No imbalance handling here on purpose: Step 8 compares "before vs after".
Preprocessing sits INSIDE the pipeline, so every CV fold fits its own
imputers/scalers on its own training part only (no leakage).
"""

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",                                        # sensitivity
    "specificity": make_scorer(recall_score, pos_label=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
}

# how many CV folds to run in parallel (RF/XGB/KNN already use all cores inside)
FOLD_JOBS = {"logreg": 5, "dtree": 5, "svm": 5, "knn": 1, "rf": 1, "xgb": 1}


def baseline_models(seed=42):
    return {
        "logreg": LogisticRegression(max_iter=1000),
        "knn":    KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        "svm":    SVC(kernel="rbf"),                  # scores via decision_function
        "dtree":  DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=seed),
        "rf":     RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                         n_jobs=-1, random_state=seed),
        "xgb":    XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                tree_method="hist", n_jobs=-1, eval_metric="logloss",
                                random_state=seed),
    }


def run_cv(name, model, features, X, y, derive_chronic_count=True, n_splits=5, seed=42):
    """Stratified k-fold CV of preprocessing + model. Returns one results row."""
    pipe = Pipeline([
        ("prep", build_preprocessor(features, derive_chronic_count)),
        ("model", model),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    t = time.time()
    res = cross_validate(pipe, X[list(features)], y, cv=cv, scoring=SCORING,
                         n_jobs=FOLD_JOBS.get(name, 1), error_score="raise")
    row = {"model": name, "n_input_features": len(features),
           "fit_seconds": round(time.time() - t, 1)}
    for m in SCORING:
        row[f"{m}_mean"] = res[f"test_{m}"].mean()
        row[f"{m}_std"] = res[f"test_{m}"].std()
    return row

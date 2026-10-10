"""
Step 9: does handling class imbalance help, and how?

Strategies compared, all inside CV so nothing leaks:
  none         - the tuned model as is
  class_weight - minority class weighted up (class_weight / scale_pos_weight)
  undersample  - random undersampling of the majority class to a 2:1 ratio
                 (SMOTE is skipped: our inputs are mostly categorical/ordinal,
                  so interpolated synthetic rows would not be realistic)
"""

import json
import time

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.baseline import baseline_models
from src.preprocessing import build_preprocessor

ROWS = {"logreg": 100000, "dtree": 100000, "xgb": 100000, "rf": 100000,
        "knn": 30000, "svm": 30000}
JOBS = {"logreg": 5, "dtree": 5, "svm": 5, "knn": 1, "rf": 1, "xgb": 1}

BASE_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "specificity": make_scorer(recall_score, pos_label=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
}


def tuned_model(name, best_params_path="../reports/best_params.json", seed=42):
    """Baseline model with the Step 8 hyperparameters applied."""
    model = baseline_models(seed)[name]
    params = json.load(open(best_params_path))[name]["best_params"]
    return model.set_params(**params)


def with_class_weight(name, model, pos_rate):
    if name in ("logreg", "dtree", "svm"):
        return model.set_params(class_weight="balanced")
    if name == "rf":
        return model.set_params(class_weight="balanced_subsample")
    if name == "xgb":
        return model.set_params(scale_pos_weight=(1 - pos_rate) / pos_rate)
    raise ValueError(f"{name} has no class-weight option")


def run_strategy(name, strategy, X, y, features, seed=42):
    n = ROWS[name]
    if n < len(y):
        _, X, _, y = train_test_split(X, y, test_size=n, stratify=y, random_state=seed)
    X = X[list(features)]

    model = tuned_model(name, seed=seed)
    # imblearn forbids a nested Pipeline as an intermediate step, so unpack its steps
    steps = list(build_preprocessor(features, derive_chronic_count=False).steps)
    if strategy == "class_weight":
        model = with_class_weight(name, model, float(y.mean()))
    elif strategy == "undersample":
        steps.append(("under", RandomUnderSampler(sampling_strategy=0.5, random_state=seed)))
    elif strategy != "none":
        raise ValueError(strategy)
    pipe = ImbPipeline(steps + [("model", model)])

    scoring = dict(BASE_SCORING)
    if name != "svm":                       # SVC has no predict_proba here
        scoring["brier"] = "neg_brier_score"

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    t = time.time()
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=JOBS[name],
                         error_score="raise")
    row = {"model": name, "strategy": strategy, "rows": len(y),
           "seconds": round(time.time() - t, 1)}
    for m in scoring:
        sign = -1 if m == "brier" else 1
        row[m] = sign * res[f"test_{m}"].mean()
        row[f"{m}_std"] = res[f"test_{m}"].std()
    return row

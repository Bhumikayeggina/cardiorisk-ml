"""
Step 8: hyperparameter search (GridSearchCV / RandomizedSearchCV).

Searches run inside the TRAINING set only. For each model we also score the
untuned baseline on exactly the same rows and folds, so "before vs after" is
a fair comparison. Primary metric: PR-AUC (average precision).
"""

import time

import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline

from src.baseline import baseline_models
from src.preprocessing import build_preprocessor

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}

# name -> (search type, search space, rows used (None = all), parallel jobs for the search)
SEARCHES = {
    "logreg": ("grid", {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3]}, None, -1),
    "dtree":  ("grid", {"model__max_depth": [4, 6, 8, 10, 12],
                        "model__min_samples_leaf": [20, 50, 100, 200]}, None, -1),
    "knn":    ("grid", {"model__n_neighbors": [15, 25, 50, 100, 200]}, 30000, 1),
    "xgb":    ("random", {"model__n_estimators": [200, 400, 600],
                          "model__learning_rate": loguniform(0.02, 0.2),
                          "model__max_depth": [3, 4, 5, 6],
                          "model__min_child_weight": [1, 5, 20],
                          "model__subsample": uniform(0.6, 0.4),
                          "model__colsample_bytree": uniform(0.5, 0.5),
                          "model__reg_lambda": loguniform(0.5, 20)}, 100000, 1),
    "rf":     ("random", {"model__n_estimators": [200, 300, 500],
                          "model__max_depth": [8, 12, 16, None],
                          "model__min_samples_leaf": [5, 10, 20, 50],
                          "model__max_features": ["sqrt", 0.3, 0.5]}, 100000, 1),
    "svm":    ("grid", {"model__C": [0.1, 1, 10],
                        "model__gamma": [0.001, 0.01, 0.1]}, 30000, -1),
}
N_ITER = {"xgb": 20, "rf": 12}          # random-search budgets


def _py(v):
    return v.item() if hasattr(v, "item") else v


def run_search(name, X, y, features, seed=42):
    """Returns (summary dict, full cv_results DataFrame)."""
    kind, space, n_rows, jobs = SEARCHES[name]
    if n_rows and n_rows < len(y):
        _, X, _, y = train_test_split(X, y, test_size=n_rows, stratify=y, random_state=seed)
    X = X[list(features)]

    def make_pipe(model):
        return Pipeline([("prep", build_preprocessor(features, derive_chronic_count=False)),
                         ("model", model)])

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    t = time.time()

    base = cross_validate(make_pipe(baseline_models(seed)[name]), X, y, cv=cv,
                          scoring=SCORING, n_jobs=jobs)

    common = dict(scoring=SCORING, refit=False, cv=cv, n_jobs=jobs)
    pipe = make_pipe(baseline_models(seed)[name])
    if kind == "grid":
        search = GridSearchCV(pipe, space, **common)
    else:
        search = RandomizedSearchCV(pipe, space, n_iter=N_ITER[name],
                                    random_state=seed, **common)
    search.fit(X, y)

    cvr = pd.DataFrame(search.cv_results_)
    i = cvr["rank_test_pr_auc"].idxmin()
    summary = {
        "model": name, "rows": int(len(y)), "n_configs": int(len(cvr)),
        "best_params": {k.replace("model__", ""): _py(v) for k, v in cvr.loc[i, "params"].items()},
        "tuned_pr_auc": float(cvr.loc[i, "mean_test_pr_auc"]),
        "tuned_pr_auc_std": float(cvr.loc[i, "std_test_pr_auc"]),
        "tuned_roc_auc": float(cvr.loc[i, "mean_test_roc_auc"]),
        "baseline_pr_auc": float(base["test_pr_auc"].mean()),
        "baseline_roc_auc": float(base["test_roc_auc"].mean()),
        "seconds": round(time.time() - t, 1),
    }
    return summary, cvr

"""
Step 7: feature ablations with paired cross-validation.

Every variant is scored on exactly the same CV folds, so the per-fold
difference against a reference variant is a fair ("paired") comparison.
"""

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def fold_scores(model, features, X, y, derive_chronic_count=True, n_splits=5, seed=42):
    """Per-fold ROC-AUC and PR-AUC (numpy arrays) for preprocessing + model."""
    pipe = Pipeline([
        ("prep", build_preprocessor(features, derive_chronic_count)),
        ("model", model),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res = cross_validate(pipe, X[list(features)], y, cv=cv,
                         scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                         n_jobs=1, error_score="raise")
    return {m: res[f"test_{m}"] for m in ("roc_auc", "pr_auc")}

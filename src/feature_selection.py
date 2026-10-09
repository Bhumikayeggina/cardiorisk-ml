"""
Step 5: compare feature-selection methods on the TRAINING data only.

Every method scores the encoded columns; scores are then rolled up to the
original features (so a one-hot group like EMPLOY1 counts as one feature,
not nine). Higher score = more important for every method.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from src.preprocessing import (BINARY_COLS, CATEGORICAL_COLS, NUMERIC_COLS,
                               ORDINAL_COLS)

SOURCE_FEATURES = NUMERIC_COLS + ORDINAL_COLS + BINARY_COLS + CATEGORICAL_COLS


def source_feature(encoded_name: str) -> str:
    """Map an encoded column name back to the original feature it came from."""
    for prefix in ("num__", "bin__", "cat__"):
        if encoded_name.startswith(prefix):
            n = encoded_name[len(prefix):]
            break
    else:
        raise ValueError(f"Unexpected column name: {encoded_name}")
    if n.startswith("missingindicator_"):
        return n[len("missingindicator_"):]
    if n in SOURCE_FEATURES:
        return n
    for c in sorted(CATEGORICAL_COLS, key=len, reverse=True):
        if n.startswith(c + "_"):
            return c
    raise ValueError(f"Cannot map column: {encoded_name}")


def roll_up(scores, names, how="max") -> pd.Series:
    s = pd.Series(np.asarray(scores, dtype=float), index=[source_feature(n) for n in names])
    s = s.fillna(0.0)
    s = s.groupby(level=0).max() if how == "max" else s.groupby(level=0).sum()
    return s.reindex([f for f in SOURCE_FEATURES if f in s.index])


def _subsample(A, y, n, seed=42):
    if n >= len(y):
        return A, y
    _, A_s, _, y_s = train_test_split(A, y, test_size=n, stratify=y, random_state=seed)
    return A_s, y_s


# ---------------- individual methods (return one score per encoded column) ----------------

def score_correlation(A, y):
    y = y.astype(float)
    yc = y - y.mean()
    Ac = A - A.mean(axis=0)
    denom = A.std(axis=0) * y.std()
    denom[denom == 0] = np.inf
    return np.abs((yc @ Ac) / len(y) / denom)


def score_chi2(A, y):
    A01 = MinMaxScaler().fit_transform(A)          # chi2 needs non-negative input
    return np.nan_to_num(chi2(A01, y)[0])


def score_mutual_info(A, y, n_sub=60000):
    A_s, y_s = _subsample(A, y, n_sub)
    discrete = np.array([np.unique(A_s[:, j]).size <= 2 for j in range(A_s.shape[1])])
    return mutual_info_classif(A_s, y_s, discrete_features=discrete, random_state=42)


def score_rfe(A, y, n_sub=40000):
    """Recursive elimination down to 1 feature gives a full ranking of columns."""
    A_s, y_s = _subsample(A, y, n_sub)
    rfe = RFE(LogisticRegression(max_iter=500, class_weight="balanced"),
              n_features_to_select=1, step=1).fit(A_s, y_s)
    return 1.0 / rfe.ranking_                      # best column (rank 1) -> score 1


def score_l1(A, y, n_sub=100000, C=0.02):
    A_s, y_s = _subsample(A, y, n_sub)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                class_weight="balanced", max_iter=300).fit(A_s, y_s)
    return np.abs(lr.coef_[0])


def score_random_forest(A, y, n_sub=100000):
    A_s, y_s = _subsample(A, y, n_sub)
    rf = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=30,
                                class_weight="balanced_subsample", n_jobs=-1,
                                random_state=42).fit(A_s, y_s)
    return rf.feature_importances_


def score_xgboost(A, y):
    pos = y.mean()
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        scale_pos_weight=(1 - pos) / pos, importance_type="gain",
                        n_jobs=-1, eval_metric="logloss", random_state=42).fit(A, y)
    return xgb.feature_importances_


METHODS = {                     # name -> (function, how to roll up one-hot groups)
    "correlation":   (score_correlation,   "max"),
    "chi_square":    (score_chi2,          "max"),
    "mutual_info":   (score_mutual_info,   "max"),
    "rfe_logreg":    (score_rfe,           "max"),
    "l1_logreg":     (score_l1,            "sum"),
    "random_forest": (score_random_forest, "sum"),
    "xgboost":       (score_xgboost,       "sum"),
}


def run_all(A, y, names, top_k=15):
    """Run every method. Returns (scores, ranks, summary) at the original-feature level."""
    y = np.asarray(y).astype(int)
    cols = {}
    for name, (fn, how) in METHODS.items():
        t = time.time()
        cols[name] = roll_up(fn(A, y), names, how)
        print(f"{name:14s} done in {time.time() - t:5.0f}s")
    scores = pd.DataFrame(cols)
    ranks = scores.rank(ascending=False, method="min")        # 1 = most important
    summary = pd.DataFrame({
        "mean_rank": ranks.mean(axis=1),
        "best_rank": ranks.min(axis=1),
        "worst_rank": ranks.max(axis=1),
        f"top{top_k}_votes": (ranks <= top_k).sum(axis=1),
    }).sort_values("mean_rank")
    return scores, ranks, summary

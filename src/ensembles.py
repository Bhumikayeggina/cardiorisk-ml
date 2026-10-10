"""
Step 11: ensembles built from out-of-fold (OOF) predictions, plus a paired
comparison across row folds.

Soft voting = average probability. Stacking = logistic-regression
meta-learner on the logit of the base models' OOF probabilities, itself
scored with 5-fold CV. No test data is used anywhere.
"""

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

EPS = 1e-6


def _logits(oof, names):
    return np.column_stack([logit(np.clip(oof[n], EPS, 1 - EPS)) for n in names])


def soft_vote(oof, names):
    return np.mean([oof[n] for n in names], axis=0)


def stack_oof(oof, names, y, seed=42):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(LogisticRegression(max_iter=1000), _logits(oof, names), y,
                             cv=cv, method="predict_proba")[:, 1]


def stacking_weights(oof, names, y):
    """Meta-learner coefficients fitted on all OOF rows (for the report)."""
    meta = LogisticRegression(max_iter=1000).fit(_logits(oof, names), y)
    return pd.Series(meta.coef_[0], index=names), float(meta.intercept_[0])


def fold_table(preds, y, n_splits=5, seed=43):
    """Per-fold scores of each prediction vector on identical row subsets."""
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    rows = []
    for k, (_, idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        for name, p in preds.items():
            rows.append({"fold": k, "model": name,
                         "roc_auc": roc_auc_score(y[idx], p[idx]),
                         "pr_auc": average_precision_score(y[idx], p[idx]),
                         "brier": brier_score_loss(y[idx], p[idx])})
    return pd.DataFrame(rows)


def summarize(folds, reference):
    """Mean score per model and paired difference vs `reference` (mean, spread)."""
    ref = folds[folds.model == reference].set_index("fold")
    out = []
    for name, g in folds.groupby("model"):
        g = g.set_index("fold")
        row = {"model": name}
        for m in ("roc_auc", "pr_auc", "brier"):
            d = g[m] - ref[m]
            row[m] = g[m].mean()
            row[f"{m}_diff"] = d.mean()
            row[f"{m}_spread"] = d.std()
        out.append(row)
    return pd.DataFrame(out).sort_values("pr_auc", ascending=False).reset_index(drop=True)

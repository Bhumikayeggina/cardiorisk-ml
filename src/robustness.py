"""
Step 13: robustness of the FINAL fitted pipeline to imperfect inputs.

All perturbations are applied to the TEST inputs only; the model and the
decision threshold stay fixed. This is a simulated stress test, not clinical
validation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def score(y, p, t):
    y = np.asarray(y)
    flag = p >= t
    tp = int((flag & (y == 1)).sum())
    return {"roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "recall": tp / int((y == 1).sum()),
            "precision": tp / max(int(flag.sum()), 1),
            "flagged_pct": 100 * flag.mean()}


# ---------------- perturbations ----------------

def inject_missing(X, frac, seed):
    """Set a random `frac` of all cells to missing (completely at random)."""
    rng = np.random.default_rng(seed)
    return X.mask(rng.random(X.shape) < frac)


def misreport_ordinal(X, cols, prob, ranges, seed):
    """With probability `prob`, an ordinal answer is off by one category."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    for c in cols:
        v = X[c].to_numpy(dtype=float, copy=True)
        move = (rng.random(len(v)) < prob) & ~np.isnan(v)
        step = rng.choice([-1.0, 1.0], size=len(v))
        v[move] = np.clip(v[move] + step[move], *ranges[c])
        X[c] = v
    return X


def perturb_numeric(X, col, sigma_frac, seed, lo, hi):
    """Add Gaussian noise (sigma = sigma_frac x the column's std) to a numeric column."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    v = X[col].to_numpy(dtype=float, copy=True)
    v = v + rng.normal(0.0, sigma_frac * np.nanstd(v), size=len(v))
    X[col] = np.clip(v, lo, hi)
    return X


def knock_out(X, col):
    """Make one question completely unanswered for everybody."""
    X = X.copy()
    X[col] = np.nan
    return X


# ---------------- experiment runner ----------------

def run_scenarios(pipe, X, y, features, t, ordinal_cols, ranges, numeric_col="PHYSHLTH",
                  numeric_range=(0, 30), n_repeats=5):
    X = X[list(features)]
    y = np.asarray(y)
    proba = lambda Z: pipe.predict_proba(Z)[:, 1]
    rows = [{"scenario": "clean", "level": 0.0, "repeat": 0, **score(y, proba(X), t)}]

    for frac in (0.05, 0.10, 0.20, 0.30, 0.40):
        for r in range(n_repeats):
            rows.append({"scenario": "missing_cells", "level": frac, "repeat": r,
                         **score(y, proba(inject_missing(X, frac, 100 + r)), t)})

    for prob in (0.05, 0.10, 0.20, 0.30):
        for r in range(n_repeats):
            Xp = misreport_ordinal(X, ordinal_cols, prob, ranges, 200 + r)
            rows.append({"scenario": "ordinal_misreport", "level": prob, "repeat": r,
                         **score(y, proba(Xp), t)})

    for sig in (0.10, 0.25, 0.50):
        for r in range(n_repeats):
            Xp = perturb_numeric(X, numeric_col, sig, 300 + r, *numeric_range)
            rows.append({"scenario": f"noise_{numeric_col}", "level": sig, "repeat": r,
                         **score(y, proba(Xp), t)})

    for col in features:
        rows.append({"scenario": "knockout", "level": col, "repeat": 0,
                     **score(y, proba(knock_out(X, col)), t)})
    return pd.DataFrame(rows)

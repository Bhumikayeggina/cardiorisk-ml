"""
Phase 10: Robustness analysis — how much does performance degrade when
inputs are corrupted (missing values, numeric noise)?
"""

import numpy as np
import pandas as pd


def inject_missingness(X: pd.DataFrame, frac: float, random_state: int = 42) -> pd.DataFrame:
    """Randomly null out `frac` of all cell values."""
    rng = np.random.default_rng(random_state)
    X_corrupt = X.copy()
    mask = rng.random(X_corrupt.shape) < frac
    X_corrupt = X_corrupt.mask(mask)
    return X_corrupt


def inject_noise(X: pd.DataFrame, numeric_cols: list[str], sigma: float = 0.1,
                  random_state: int = 42) -> pd.DataFrame:
    """Add Gaussian noise (as a fraction of each column's std) to numeric columns."""
    rng = np.random.default_rng(random_state)
    X_noisy = X.copy()
    for col in numeric_cols:
        std = X_noisy[col].std()
        X_noisy[col] = X_noisy[col] + rng.normal(0, sigma * std, size=len(X_noisy))
    return X_noisy


def robustness_sweep(pipeline, X_test, y_test, evaluate_fn, fracs=(0.0, 0.05, 0.1, 0.2)):
    """Re-score the fitted pipeline at increasing missingness levels.
    `evaluate_fn` should be `src.evaluate.evaluate_model`."""
    results = []
    for frac in fracs:
        X_corrupt = inject_missingness(X_test, frac) if frac > 0 else X_test
        y_proba = pipeline.predict_proba(X_corrupt)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = evaluate_fn(y_test, y_pred, y_proba)
        metrics["missing_frac"] = frac
        results.append(metrics)
    return results

"""
Feature engineering + leak-safe preprocessing for CardioRisk.

Everything here is shared by the notebooks and the Streamlit app, so the
same transformations are applied in training and in the demo.

Usage:
    from src.preprocessing import build_preprocessor, ALL_FEATURES
    pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
    pipe.fit(X_train[ALL_FEATURES], y_train)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "_MICHD"

# ---- column groups (names after Step 2 cleaning) ----
NUMERIC_COLS = ["_BMI5", "PHYSHLTH", "MENTHLTH", "_DRNKWK3", "chronic_count"]
ORDINAL_COLS = ["_AGE_G", "GENHLTH", "INCOME3", "EDUCA"]   # order is meaningful
BINARY_COLS = [
    "CHCKDNY2", "CHCCOPD3", "ASTHMA3", "ADDEPEV3", "CHCSCNC1", "CHCOCNC1",
    "HAVARTH4", "DIFFWALK", "DIFFDRES", "DECIDE", "MEDCOST1", "DRNKANY6",
    "_TOTINDA", "_RFDRHV9",
]                                                           # already 0/1
CATEGORICAL_COLS = ["SEXVAR", "_RACE", "DIABETE4", "_SMOKER3", "MARITAL",
                    "EMPLOY1", "CHECKUP1"]                  # no natural order

RAW_FEATURES = ORDINAL_COLS + BINARY_COLS + CATEGORICAL_COLS + \
               ["_BMI5", "PHYSHLTH", "MENTHLTH", "_DRNKWK3"]
ALL_FEATURES = RAW_FEATURES  # chronic_count is derived inside the pipeline

CONDITION_COLS = ["CHCKDNY2", "CHCCOPD3", "ASTHMA3", "ADDEPEV3",
                  "CHCSCNC1", "CHCOCNC1", "HAVARTH4"]

# Fixed domain limits (NOT learned from data, so no leakage)
BMI_MAX = 70.0
DRINKS_PER_WEEK_MAX = 100.0



   def prepare_features(X: pd.DataFrame, derive_chronic_count: bool = True) -> pd.DataFrame:
    """Cap implausible values and (optionally) add the derived feature
    `chronic_count` = number of reported chronic conditions (unanswered = 0)."""
    X = X.copy()
    if "_BMI5" in X.columns:
        X["_BMI5"] = X["_BMI5"].clip(upper=BMI_MAX)
    if "_DRNKWK3" in X.columns:
        X["_DRNKWK3"] = X["_DRNKWK3"].clip(upper=DRINKS_PER_WEEK_MAX)
    if derive_chronic_count:
        X["chronic_count"] = X[CONDITION_COLS].sum(axis=1, skipna=True)
    return X


def build_preprocessor(features=None, derive_chronic_count: bool = True) -> Pipeline:
    """Leak-safe preprocessing for any subset of ALL_FEATURES.

    features: raw columns the model will receive (default: all 29).
    derive_chronic_count: add `chronic_count`; only possible when all seven
        condition columns are in `features`.
    """
    features = list(ALL_FEATURES if features is None else features)
    derive = derive_chronic_count and all(c in features for c in CONDITION_COLS)

    num_cols = [c for c in NUMERIC_COLS if c in features or (c == "chronic_count" and derive)]
    ord_cols = [c for c in ORDINAL_COLS if c in features]
    bin_cols = [c for c in BINARY_COLS if c in features]
    cat_cols = [c for c in CATEGORICAL_COLS if c in features]

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=-1)),  # -1 = "unknown" category
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    blocks = [("num", numeric, num_cols + ord_cols),
              ("bin", binary, bin_cols),
              ("cat", categorical, cat_cols)]
    encode = ColumnTransformer([b for b in blocks if len(b[2]) > 0])

    return Pipeline([
        ("prepare", FunctionTransformer(prepare_features, validate=False,
                                        kw_args={"derive_chronic_count": derive})),
        ("encode", encode),
    ])


def feature_names(fitted_preprocessor: Pipeline):
    """Readable names of the columns the fitted preprocessor outputs."""
    return fitted_preprocessor.named_steps["encode"].get_feature_names_out()

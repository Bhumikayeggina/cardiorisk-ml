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


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Cap implausible values and add the derived feature `chronic_count`
    (number of reported chronic conditions; unanswered counts as 0)."""
    X = X.copy()
    X["_BMI5"] = X["_BMI5"].clip(upper=BMI_MAX)
    X["_DRNKWK3"] = X["_DRNKWK3"].clip(upper=DRINKS_PER_WEEK_MAX)
    X["chronic_count"] = X[CONDITION_COLS].sum(axis=1, skipna=True)
    return X


def build_preprocessor() -> Pipeline:
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

    encode = ColumnTransformer([
        ("num", numeric, NUMERIC_COLS + ORDINAL_COLS),
        ("bin", binary, BINARY_COLS),
        ("cat", categorical, CATEGORICAL_COLS),
    ])

    return Pipeline([
        ("prepare", FunctionTransformer(prepare_features, validate=False)),
        ("encode", encode),
    ])


def feature_names(fitted_preprocessor: Pipeline):
    """Readable names of the columns the fitted preprocessor outputs."""
    return fitted_preprocessor.named_steps["encode"].get_feature_names_out()

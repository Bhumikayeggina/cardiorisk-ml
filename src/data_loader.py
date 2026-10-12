"""
Data acquisition and cleaning for CardioRisk (notebooks 01-02).

    python src/data_loader.py        # run from the repo root

downloads the 2024 BRFSS file straight from the CDC (nothing is saved except the
two parquet files below), keeps the 29 candidate questions plus the outcome, and
applies the per-column recoding documented in notebooks/01_data_cleaning.ipynb:

    data/raw/brfss2024_subset.parquet     raw codes, 30 columns
    data/processed/cleaned.parquet        cleaned, ready for modelling
"""

import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

BRFSS_2024_XPT_URL = "https://cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip"
TARGET_COL = "_MICHD"          # 1 = ever told coronary heart disease / heart attack

SHORTLIST_COLS = [
    # demographics
    "_AGE_G", "SEXVAR", "_RACE", "EDUCA", "INCOME3", "MARITAL", "EMPLOY1",
    # general health and function
    "GENHLTH", "_BMI5", "PHYSHLTH", "MENTHLTH", "DIFFWALK", "DIFFDRES", "DECIDE",
    # chronic conditions
    "DIABETE4", "CHCKDNY2", "CHCCOPD3", "ASTHMA3", "ADDEPEV3", "CHCSCNC1", "CHCOCNC1", "HAVARTH4",
    # lifestyle and access to care
    "_SMOKER3", "_TOTINDA", "_RFDRHV9", "DRNKANY6", "_DRNKWK3", "MEDCOST1", "CHECKUP1",
]

YES_NO = ["CHCKDNY2", "CHCCOPD3", "ASTHMA3", "ADDEPEV3", "CHCSCNC1", "CHCOCNC1",
          "HAVARTH4", "DIFFWALK", "DIFFDRES", "DECIDE", "MEDCOST1", "DRNKANY6"]

# codes meaning "don't know / refused" for each column (checked against the codebook)
UNKNOWN_CODES = {
    "_RACE": [9], "EDUCA": [9], "INCOME3": [77, 99], "GENHLTH": [7, 9],
    "DIABETE4": [7, 9], "_SMOKER3": [9], "MARITAL": [9], "EMPLOY1": [9],
    "CHECKUP1": [7, 9],
}


def download_and_load_brfss(url: str = BRFSS_2024_XPT_URL) -> pd.DataFrame:
    """Fetch the zip from the CDC, unzip in memory and parse the .XPT file.
    If the request fails, try https://www.cdc.gov/... (depends on the network)."""
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_sas(f, format="xport", encoding="latin-1")


def clean_brfss(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column recoding of the raw BRFSS codes (see notebook 01 for the reasoning)."""
    df = df[SHORTLIST_COLS + [TARGET_COL]].copy()

    # target: drop unknown outcome, then 1 = has CHD/MI, 0 = does not
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)

    # yes/no questions: 1 = yes, 2 = no; 7, 9 and blank become missing
    for c in YES_NO:
        df[c] = df[c].map({1.0: 1, 2.0: 0})

    # calculated risk flags, recoded so 1 = risk
    df["_TOTINDA"] = df["_TOTINDA"].map({1.0: 0, 2.0: 1})      # 1 = physically inactive
    df["_RFDRHV9"] = df["_RFDRHV9"].map({1.0: 0, 2.0: 1})      # 1 = heavy drinker

    # categorical columns: unknown / refused become missing
    for c, codes in UNKNOWN_CODES.items():
        df[c] = df[c].replace(codes, np.nan)

    # days of poor health: 88 = none
    for c in ["PHYSHLTH", "MENTHLTH"]:
        df[c] = df[c].replace({88: 0, 77: np.nan, 99: np.nan})

    # BMI is stored as BMI x 100
    df["_BMI5"] = df["_BMI5"] / 100

    # drinks per week: 99900 = unknown; tiny values are SAS floating-point zeros; stored x 100
    dw = df["_DRNKWK3"].copy()
    dw[dw == 99900] = np.nan
    dw[dw < 1e-6] = 0
    df["_DRNKWK3"] = dw / 100

    return df.reset_index(drop=True)


def main():
    raw = download_and_load_brfss()
    print("downloaded:", raw.shape)
    subset = raw[SHORTLIST_COLS + [TARGET_COL]].copy()
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    subset.to_parquet("data/raw/brfss2024_subset.parquet", index=False)

    clean = clean_brfss(subset)
    clean.to_parquet("data/processed/cleaned.parquet", index=False)
    print("cleaned:", clean.shape, " positive rate:", round(clean[TARGET_COL].mean(), 4))


if __name__ == "__main__":
    main()

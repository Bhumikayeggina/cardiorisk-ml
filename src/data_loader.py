"""
Phase 1-2: Load the raw BRFSS 2024 extract and produce a cleaned dataframe.

Usage:
    from src.data_loader import load_raw_brfss, clean_brfss

    df_raw = load_raw_brfss("data/raw/LLCP2024.XPT")
    df_clean = clean_brfss(df_raw, feature_cols=SHORTLIST_COLS)
    df_clean.to_parquet("data/processed/cleaned.parquet")
"""

import io
import zipfile

import pandas as pd
import requests

BRFSS_2024_XPT_URL = "https://cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip"
BRFSS_2024_CODEBOOK_URL = "https://cdc.gov/brfss/annual_data/2024/zip/codebook24_llcp-v2-508.zip"

# TODO: finalize against the 2024 codebook — variable names can shift slightly year to year.
TARGET_COL = "_MICHD"

SHORTLIST_COLS = [
    "_AGE_G", "SEXVAR", "_RACE", "EDUCA", "INCOME3",
    "GENHLTH", "_BMI5", "DIABETE4", "_RFHYPE6", "TOLDHI3",
    "_SMOKER3", "_TOTINDA", "SLEPTIM1", "_RFDRHV8",
    # add more from your notebooks/00_variable_shortlist.md
]

# BRFSS commonly uses these as "don't know / refused / missing" sentinel codes.
# Confirm the exact set per-column against the codebook before applying blindly.
SENTINEL_CODES = [7, 9, 77, 99, 777, 999, 7777, 9999]


def load_raw_brfss(path: str) -> pd.DataFrame:
    """Load the SAS transport (.XPT) BRFSS file from a local path. Convert once
    and cache as parquet afterwards — re-parsing XPT on a 457K x 345 table is slow."""
    return pd.read_sas(path, format="xport", encoding="latin-1")


def download_and_load_brfss(url: str = BRFSS_2024_XPT_URL) -> pd.DataFrame:
    """Download the BRFSS zip directly from the CDC, unzip in-memory, and parse
    the .XPT — no manual download step needed. Run this once, then cache the
    result with df.to_parquet() and load from disk on every subsequent run.

    If the request times out or 403s, try prefixing the URL with 'www.'
    (https://www.cdc.gov/... vs https://cdc.gov/...) — this varies by network.
    """
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        xpt_filename = z.namelist()[0]
        with z.open(xpt_filename) as f:
            return pd.read_sas(f, format="xport", encoding="latin-1")


def download_codebook(url: str = BRFSS_2024_CODEBOOK_URL, out_path: str = "data/raw/codebook24.zip") -> None:
    """Download the codebook zip for reference (open the PDF inside manually
    while writing cleaning code — not meant to be parsed programmatically)."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


def clean_brfss(df: pd.DataFrame, feature_cols: list[str] = SHORTLIST_COLS,
                 target_col: str = TARGET_COL) -> pd.DataFrame:
    """Subset to the feature shortlist + target, drop missing target rows,
    and null out sentinel 'unknown/refused' codes.

    NOTE: sentinel codes vary by column semantics (e.g. 7/9 vs 77/99 vs 777/999) —
    this generic pass is a starting point. Cross-check each column against the
    codebook and override per-column as needed before trusting the output.
    """
    cols = feature_cols + [target_col]
    out = df[cols].copy()

    out = out[out[target_col].notna()]
    out = out[~out[target_col].isin([7, 9])]  # unknown / refused on target

    for col in feature_cols:
        out.loc[out[col].isin(SENTINEL_CODES), col] = pd.NA

    out[target_col] = (out[target_col] == 1).astype(int)  # 1=Yes -> 1, else 0
    return out.reset_index(drop=True)


if __name__ == "__main__":
    raw = download_and_load_brfss()
    raw.to_parquet("data/raw/full_2024.parquet", index=False)
    clean = clean_brfss(raw)
    clean.to_parquet("data/processed/cleaned.parquet", index=False)
    print(f"Saved cleaned data: {clean.shape}")

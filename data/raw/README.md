# Raw data folder

The 2024 BRFSS dataset is too large to commit to GitHub. Download it yourself:

1. Go to the CDC BRFSS Annual Survey Data page and download the **2024 BRFSS Data (XPT format)**.
2. Also download the **2024 Calculated Variables / Codebook** for variable definitions.
3. Place the `.XPT` file here, then run `notebooks/01_data_cleaning.ipynb` (or `src/data_loader.py`) to convert it to `data/processed/cleaned.parquet`.

See `PROJECT_GUIDE.md` Phase 1 for details.

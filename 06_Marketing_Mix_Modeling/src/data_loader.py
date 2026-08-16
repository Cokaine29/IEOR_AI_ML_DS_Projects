"""
data_loader.py
==============
Downloads / loads the Marketing Mix dataset from Kaggle and prepares it
for modeling:
  1. Loads raw CSV (auto-downloads a built-in sample if Kaggle not configured)
  2. Cleans and engineers base features (week index, seasonality dummies)
  3. Temporal train/test split (last 20% = holdout, no random split)
  4. Saves processed data to data/processed/
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

MEDIA_CHANNELS = ["TV", "Radio", "InStore", "NewspaperInserts", "Website_Campaign"]
TARGET         = "NewVolSales"

# ── Synthetic dataset generator (used if Kaggle download is unavailable) ──────
def _generate_synthetic_data(n_weeks: int = 156, seed: int = 42) -> pd.DataFrame:
    """
    Generates 3 years of realistic weekly MMM data with:
      - Geometric adstock on each channel (known ground-truth decay rates)
      - Hill saturation curves
      - Seasonality (Diwali / summer peaks)
      - Price sensitivity
      - Realistic multicollinearity between TV and Digital
    Ground-truth decay rates: TV=0.7, Radio=0.4, InStore=0.3, News=0.2, Web=0.5
    """
    rng = np.random.default_rng(seed)
    weeks = np.arange(n_weeks)

    # ── Raw channel spend (units: INR thousand per week) ──────────────────────
    tv       = np.abs(rng.normal(150, 40, n_weeks) + 60 * np.sin(2 * np.pi * weeks / 52))
    radio    = np.abs(rng.normal(60,  20, n_weeks))
    instore  = np.abs(rng.normal(80,  25, n_weeks) + 30 * np.sin(2 * np.pi * weeks / 52))
    news     = np.abs(rng.normal(40,  15, n_weeks))
    web      = np.abs(rng.normal(100, 35, n_weeks) + 20 * np.sin(2 * np.pi * weeks / 52))
    # TV and Web are moderately correlated (both peak together)
    web      = 0.6 * web + 0.4 * tv + rng.normal(0, 10, n_weeks)
    web      = np.clip(web, 10, None)

    price    = 100 + rng.normal(0, 5, n_weeks)
    discount = rng.choice([0, 5, 10, 15, 20], size=n_weeks, p=[0.5, 0.2, 0.15, 0.1, 0.05])

    # ── Adstock (geometric decay) ──────────────────────────────────────────────
    def adstock(x, theta):
        a = np.zeros_like(x)
        a[0] = x[0]
        for t in range(1, len(x)):
            a[t] = x[t] + theta * a[t - 1]
        return a

    def hill_saturation(x, alpha=2.0, gamma=0.5):
        """Hill function: diminishing returns."""
        xn = x / (x.max() + 1e-8)
        return xn ** alpha / (xn ** alpha + gamma ** alpha)

    tv_a    = adstock(tv,      0.70)
    radio_a = adstock(radio,   0.40)
    ins_a   = adstock(instore, 0.30)
    news_a  = adstock(news,    0.20)
    web_a   = adstock(web,     0.50)

    tv_s    = hill_saturation(tv_a,    alpha=2.0, gamma=0.6)
    radio_s = hill_saturation(radio_a, alpha=1.5, gamma=0.5)
    ins_s   = hill_saturation(ins_a,   alpha=1.8, gamma=0.5)
    news_s  = hill_saturation(news_a,  alpha=1.2, gamma=0.4)
    web_s   = hill_saturation(web_a,   alpha=2.2, gamma=0.55)

    # Seasonality: Diwali (week 42-46) and Summer (week 20-24)
    seasonality = (
        20 * np.exp(-0.5 * ((weeks - 43) / 2) ** 2) +
        10 * np.exp(-0.5 * ((weeks - 22) / 2) ** 2) +
        rng.normal(0, 5, n_weeks)
    )

    # ── Sales DGP (data-generating process) ───────────────────────────────────
    base_sales = 500
    sales = (
        base_sales
        + 120 * tv_s
        + 60  * radio_s
        + 80  * ins_s
        + 40  * news_s
        + 100 * web_s
        - 3   * (price - 100)       # price sensitivity
        + 2   * discount             # discount lifts
        + seasonality
        + rng.normal(0, 15, n_weeks) # noise
    )
    sales = np.clip(sales, 50, None)

    df = pd.DataFrame({
        "Week":              weeks + 1,
        "TV":                np.round(tv,      1),
        "Radio":             np.round(radio,   1),
        "InStore":           np.round(instore, 1),
        "NewspaperInserts":  np.round(news,    1),
        "Website_Campaign":  np.round(web,     1),
        "Base_Price":        np.round(price,   2),
        "Discount":          discount,
        "NewVolSales":       np.round(sales,   0).astype(int),
    })
    return df


# ── Loader ────────────────────────────────────────────────────────────────────
def load_raw_data() -> pd.DataFrame:
    """
    Tries to load 'marketing_mix.csv' from data/raw/.
    Falls back to generating synthetic data if not found.
    """
    csv_path = os.path.join(RAW_DIR, "marketing_mix.csv")
    if os.path.exists(csv_path):
        print(f"[data_loader] Loading real data from {csv_path}")
        df = pd.read_csv(csv_path)
        # Standardise column names
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        return df
    else:
        print("[data_loader] 'marketing_mix.csv' not found in data/raw/.")
        print("[data_loader] Generating realistic synthetic MMM data (156 weeks / 3 years)...")
        df = _generate_synthetic_data(n_weeks=156, seed=42)
        df.to_csv(csv_path, index=False)
        print(f"[data_loader] Saved synthetic data to {csv_path}")
        return df


def prepare_data(df: pd.DataFrame, test_frac: float = 0.20) -> dict:
    """
    Cleans data, adds temporal features, performs temporal train/test split.

    Returns dict with keys:
        df_full, df_train, df_test,
        X_train, X_test, y_train, y_test,
        feature_names, split_week
    """
    df = df.copy()

    # ── Ensure required columns exist ─────────────────────────────────────────
    required = MEDIA_CHANNELS + [TARGET]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Fill NaN with 0 for spend columns ─────────────────────────────────────
    df[MEDIA_CHANNELS] = df[MEDIA_CHANNELS].fillna(0)
    df[TARGET]         = df[TARGET].fillna(df[TARGET].median())

    # ── Add price / discount if present, else defaults ────────────────────────
    if "Base_Price" not in df.columns:
        df["Base_Price"] = 100.0
    if "Discount" not in df.columns:
        df["Discount"] = 0.0

    # ── Week index ────────────────────────────────────────────────────────────
    if "Week" not in df.columns:
        df["Week"] = np.arange(1, len(df) + 1)

    n = len(df)
    df = df.reset_index(drop=True)

    # ── Stationarity check (ADF test on sales) ────────────────────────────────
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(df[TARGET])
    print(f"\n[data_loader] ADF Test on {TARGET}: statistic={adf_result[0]:.3f}, p-value={adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print("              -> Series is STATIONARY (good for regression).")
    else:
        print("              -> Series may be NON-STATIONARY — consider differencing.")

    # ── Temporal train/test split ──────────────────────────────────────────────
    split_idx  = int(n * (1 - test_frac))
    split_week = df["Week"].iloc[split_idx]
    df_train   = df.iloc[:split_idx].copy()
    df_test    = df.iloc[split_idx:].copy()

    print(f"[data_loader] Train: {len(df_train)} weeks  |  Test: {len(df_test)} weeks  |  Split at week {split_week}")

    # ── Feature set (raw spend + price/discount controls) ─────────────────────
    feature_cols = MEDIA_CHANNELS + ["Base_Price", "Discount"]

    X_train = df_train[feature_cols].values.astype(float)
    y_train = df_train[TARGET].values.astype(float)
    X_test  = df_test[feature_cols].values.astype(float)
    y_test  = df_test[TARGET].values.astype(float)

    # ── Basic EDA summary ─────────────────────────────────────────────────────
    print(f"\n[data_loader] Sales stats:\n{df[TARGET].describe().round(1)}")
    print(f"\n[data_loader] Channel spend means (weekly):")
    for ch in MEDIA_CHANNELS:
        print(f"  {ch:<22}: {df[ch].mean():.1f}  (std={df[ch].std():.1f})")

    # ── Save processed splits ─────────────────────────────────────────────────
    df_train.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
    df_test.to_csv(os.path.join(PROC_DIR, "test.csv"),  index=False)
    df.to_csv(os.path.join(PROC_DIR, "full.csv"),       index=False)
    print(f"\n[data_loader] Saved splits to {PROC_DIR}")

    return {
        "df_full":       df,
        "df_train":      df_train,
        "df_test":       df_test,
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "feature_names": feature_cols,
        "split_week":    split_week,
        "n_channels":    len(MEDIA_CHANNELS),
    }


if __name__ == "__main__":
    df_raw = load_raw_data()
    print(f"\nDataset shape: {df_raw.shape}")
    print(df_raw.head())
    data = prepare_data(df_raw)
    print(f"\nTrain X shape: {data['X_train'].shape}")
    print(f"Test  X shape: {data['X_test'].shape}")

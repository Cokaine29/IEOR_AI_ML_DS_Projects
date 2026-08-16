"""
transforms.py
=============
Adstock, saturation, and seasonality transformations for MMM.

Key functions:
  - adstock_geometric(x, theta)         : geometric decay carryover
  - hill_saturation(x, alpha, gamma)    : diminishing returns (Hill function)
  - apply_transforms(df, params)        : apply both transforms to all channels
  - grid_search_transforms(X, y, ...)   : find optimal adstock+saturation params
                                          via TimeSeriesSplit cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import itertools
import warnings
warnings.filterwarnings("ignore")


MEDIA_CHANNELS = ["TV", "Radio", "InStore", "NewspaperInserts", "Website_Campaign"]


# ── Core transformation functions ─────────────────────────────────────────────

def adstock_geometric(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Geometric adstock (carryover) transformation.

    Models the lagged, decaying effect of advertising spend:
        A_t = x_t + theta * A_{t-1}

    Parameters
    ----------
    x     : raw spend array (T,)
    theta : decay rate in [0, 1). 0 = no carryover, 0.9 = very long memory
            Typical values: TV=0.6-0.8, Search=0.1-0.3, Social=0.3-0.5

    Returns
    -------
    adstocked array (T,) -- same shape as x
    """
    assert 0.0 <= theta < 1.0, f"theta must be in [0,1), got {theta}"
    result = np.zeros_like(x, dtype=float)
    result[0] = x[0]
    for t in range(1, len(x)):
        result[t] = x[t] + theta * result[t - 1]
    return result


def hill_saturation(x: np.ndarray, alpha: float = 2.0, gamma: float = 0.5,
                    ref_max: float = None) -> np.ndarray:
    """
    Hill function saturation (diminishing returns).

    Models the fact that doubling ad spend does NOT double sales:
        S(x) = x^alpha / (x^alpha + K^alpha)

    where K = gamma * max(x) is the half-saturation point.

    Parameters
    ----------
    x       : adstocked spend array (T,)
    alpha   : shape/steepness parameter > 0. Higher = sharper S-curve.
    gamma   : half-saturation parameter as fraction of max spend. 0.5 = saturation
              kicks in at 50% of peak spend.
    ref_max : if provided, normalize by this value (from training data) instead
              of x.max(). CRITICAL: must be set on test/inference data so that
              the normalization scale is consistent with training.

    Returns
    -------
    saturated array in [0, 1) -- same shape as x
    """
    assert alpha > 0, f"alpha must be positive, got {alpha}"
    assert 0 < gamma < 1, f"gamma must be in (0,1), got {gamma}"
    max_val = ref_max if ref_max is not None else (x.max() + 1e-8)
    x_norm  = x / (max_val + 1e-8)
    return x_norm ** alpha / (x_norm ** alpha + gamma ** alpha)


def apply_adstock(df: pd.DataFrame, thetas: dict) -> pd.DataFrame:
    """
    Apply geometric adstock to each media channel in df.

    Parameters
    ----------
    df     : DataFrame with raw spend columns
    thetas : dict mapping channel name -> theta (decay rate)

    Returns
    -------
    DataFrame with adstocked channel columns (same names, values replaced)
    """
    df_out = df.copy()
    for ch in MEDIA_CHANNELS:
        if ch in df_out.columns:
            theta = thetas.get(ch, 0.0)
            df_out[ch] = adstock_geometric(df_out[ch].values, theta)
    return df_out


def apply_saturation(df: pd.DataFrame, alphas: dict, gammas: dict,
                     ref_maxes: dict = None) -> pd.DataFrame:
    """
    Apply Hill saturation to each (already adstocked) media channel.

    Parameters
    ----------
    df        : DataFrame with adstocked spend columns
    alphas    : dict mapping channel name -> alpha
    gammas    : dict mapping channel name -> gamma
    ref_maxes : dict mapping channel name -> ref_max (from training adstocked max).
                MUST be set when transforming test/inference data.

    Returns
    -------
    DataFrame with saturated channel columns (values in [0,1))
    """
    df_out = df.copy()
    for ch in MEDIA_CHANNELS:
        if ch in df_out.columns:
            alpha   = alphas.get(ch, 2.0)
            gamma   = gammas.get(ch, 0.5)
            ref_max = ref_maxes.get(ch) if ref_maxes else None
            df_out[ch] = hill_saturation(df_out[ch].values, alpha, gamma, ref_max)
    return df_out


def apply_transforms(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Apply both adstock and saturation in one call.

    Parameters
    ----------
    df     : DataFrame with raw spend columns
    params : dict with keys 'thetas', 'alphas', 'gammas', and optionally 'ref_maxes'
             e.g. {'thetas': {'TV': 0.7, ...}, 'alphas': {...}, 'gammas': {...},
                   'ref_maxes': {'TV': 320.0, ...}}

    Returns
    -------
    Transformed DataFrame
    """
    df_a = apply_adstock(df, params["thetas"])
    df_s = apply_saturation(df_a, params["alphas"], params["gammas"],
                            ref_maxes=params.get("ref_maxes"))
    return df_s


# ── Grid search for optimal transform parameters ──────────────────────────────

THETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
GAMMA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def _build_feature_matrix(df_train: pd.DataFrame,
                           thetas: dict, alphas: dict, gammas: dict,
                           control_cols: list) -> np.ndarray:
    """Apply transforms and return feature matrix for regression."""
    df_t = apply_adstock(df_train, thetas)
    df_t = apply_saturation(df_t, alphas, gammas)
    feature_cols = MEDIA_CHANNELS + control_cols
    feature_cols = [c for c in feature_cols if c in df_t.columns]
    return df_t[feature_cols].values.astype(float)


def grid_search_transforms(df_train: pd.DataFrame,
                            target_col: str = "NewVolSales",
                            control_cols: list = None,
                            n_splits: int = 3,
                            verbose: bool = True) -> dict:
    """
    Grid search over adstock decay rates and saturation parameters to find
    the combination that maximizes out-of-fold R² using TimeSeriesSplit CV.

    Strategy: search theta per channel independently first (faster),
    then fix thetas and search alpha/gamma per channel.

    Returns
    -------
    dict with keys:
      'thetas': {channel: best_theta},
      'alphas': {channel: best_alpha},
      'gammas': {channel: best_gamma},
      'best_r2': float
    """
    if control_cols is None:
        control_cols = ["Base_Price", "Discount"]
    control_cols = [c for c in control_cols if c in df_train.columns]

    y = df_train[target_col].values.astype(float)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # ── Stage 1: Search theta per channel (alpha=2.0, gamma=0.5 fixed) ────────
    if verbose:
        print("\n[transforms] Stage 1: Grid-searching adstock decay rates (theta)...")

    best_thetas = {}
    for ch in MEDIA_CHANNELS:
        if ch not in df_train.columns:
            best_thetas[ch] = 0.0
            continue

        best_r2    = -np.inf
        best_theta = 0.0

        for theta in THETA_GRID:
            thetas_trial = {c: (theta if c == ch else 0.0) for c in MEDIA_CHANNELS}
            alphas_fixed  = {c: 2.0 for c in MEDIA_CHANNELS}
            gammas_fixed  = {c: 0.5 for c in MEDIA_CHANNELS}

            X = _build_feature_matrix(df_train, thetas_trial, alphas_fixed, gammas_fixed, control_cols)
            scaler = StandardScaler()

            fold_r2 = []
            for tr_idx, val_idx in tscv.split(X):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]
                X_tr_s  = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)
                model = Ridge(alpha=1.0)
                model.fit(X_tr_s, y_tr)
                y_hat = model.predict(X_val_s)
                fold_r2.append(r2_score(y_val, y_hat))

            mean_r2 = np.mean(fold_r2)
            if mean_r2 > best_r2:
                best_r2    = mean_r2
                best_theta = theta

        best_thetas[ch] = best_theta
        if verbose:
            print(f"  {ch:<22}: theta={best_theta:.1f}  (cv R2={best_r2:.3f})")

    # ── Stage 2: Search alpha/gamma per channel (best thetas fixed) ───────────
    if verbose:
        print("\n[transforms] Stage 2: Grid-searching saturation params (alpha, gamma)...")

    best_alphas = {}
    best_gammas = {}

    for ch in MEDIA_CHANNELS:
        if ch not in df_train.columns:
            best_alphas[ch] = 2.0
            best_gammas[ch] = 0.5
            continue

        best_r2    = -np.inf
        best_alpha = 2.0
        best_gamma = 0.5

        for alpha, gamma in itertools.product(ALPHA_GRID, GAMMA_GRID):
            alphas_trial = {c: (alpha if c == ch else 2.0) for c in MEDIA_CHANNELS}
            gammas_trial = {c: (gamma if c == ch else 0.5) for c in MEDIA_CHANNELS}

            X = _build_feature_matrix(df_train, best_thetas, alphas_trial, gammas_trial, control_cols)
            scaler = StandardScaler()

            fold_r2 = []
            for tr_idx, val_idx in tscv.split(X):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]
                X_tr_s  = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)
                model = Ridge(alpha=1.0)
                model.fit(X_tr_s, y_tr)
                y_hat = model.predict(X_val_s)
                fold_r2.append(r2_score(y_val, y_hat))

            mean_r2 = np.mean(fold_r2)
            if mean_r2 > best_r2:
                best_r2    = mean_r2
                best_alpha = alpha
                best_gamma = gamma

        best_alphas[ch] = best_alpha
        best_gammas[ch] = best_gamma
        if verbose:
            print(f"  {ch:<22}: alpha={best_alpha:.1f}  gamma={best_gamma:.1f}  (cv R2={best_r2:.3f})")

    final_params = {
        "thetas": best_thetas,
        "alphas": best_alphas,
        "gammas": best_gammas,
    }

    # Final CV score with all best params
    X_final = _build_feature_matrix(df_train, best_thetas, best_alphas, best_gammas, control_cols)
    scaler  = StandardScaler()
    fold_r2 = []
    for tr_idx, val_idx in tscv.split(X_final):
        X_tr_s  = scaler.fit_transform(X_final[tr_idx])
        X_val_s = scaler.transform(X_final[val_idx])
        model   = Ridge(alpha=1.0)
        model.fit(X_tr_s, y[tr_idx])
        fold_r2.append(r2_score(y[val_idx], model.predict(X_val_s)))

    final_params["best_cv_r2"] = float(np.mean(fold_r2))
    if verbose:
        print(f"\n[transforms] Final combined CV R2: {final_params['best_cv_r2']:.4f}")

    return final_params


if __name__ == "__main__":
    # Quick smoke test
    x = np.array([100, 50, 30, 20, 10], dtype=float)
    a = adstock_geometric(x, theta=0.5)
    print("Adstock test:", a)
    s = hill_saturation(x, alpha=2.0, gamma=0.5)
    print("Saturation test:", s)

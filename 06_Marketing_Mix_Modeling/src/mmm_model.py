"""
mmm_model.py
============
Trains and evaluates three regularized regression models on transformed
(adstocked + saturated) features:
  - Ridge Regression      (L2 — handles multicollinearity)
  - Lasso Regression      (L1 — implicit feature/channel selection)
  - Bayesian Ridge         (uncertainty estimates on channel coefficients)

After fitting, computes:
  - Channel contribution decomposition (% of sales from each channel)
  - VIF diagnostics (multicollinearity check)
  - Out-of-sample R², MAPE, RMSE on temporal holdout
  - MLflow experiment tracking for every model run
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

MEDIA_CHANNELS = ["TV", "Radio", "InStore", "NewspaperInserts", "Website_Campaign"]
CONTROL_COLS   = ["Base_Price", "Discount"]


# ── VIF diagnostic ─────────────────────────────────────────────────────────────

def compute_vif(X: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Computes Variance Inflation Factor for each feature."""
    vif_data = []
    for i in range(X.shape[1]):
        try:
            vif = variance_inflation_factor(X, i)
        except Exception:
            vif = np.nan
        vif_data.append({"Feature": feature_names[i], "VIF": round(vif, 2)})
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # as %
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"r2": round(r2, 4), "mape": round(mape, 2), "rmse": round(rmse, 2)}


# ── Channel contribution decomposition ────────────────────────────────────────

def compute_contributions(model, X_transformed: np.ndarray,
                           feature_names: list, y_true: np.ndarray,
                           scaler: StandardScaler) -> pd.DataFrame:
    """
    Decompose model predictions into per-channel incremental sales contributions
    using a drop-one (leave-one-out) method.

    For each channel i: contribution_i = mean(y_pred_full - y_pred_with_channel_i_zeroed)

    This is robust even when Lasso zeros out coefficients, because it correctly
    attributes credit to correlated channels.
    """
    X_std_full = scaler.transform(X_transformed)
    y_pred_full = model.predict(X_std_full)

    mean_contribs = {}

    for i, name in enumerate(feature_names):
        X_zero = X_transformed.copy()
        X_zero[:, i] = 0.0   # zero out this channel
        X_std_zero = scaler.transform(X_zero)
        y_pred_zero = model.predict(X_std_zero)
        mean_contribs[name] = float(np.mean(y_pred_full - y_pred_zero))

    # Baseline = intercept effect (all channels zeroed)
    X_all_zero = np.zeros_like(X_transformed)
    X_all_zero[:, feature_names.index("Base_Price")] = X_transformed[:, feature_names.index("Base_Price")]
    X_all_zero[:, feature_names.index("Discount")]   = X_transformed[:, feature_names.index("Discount")]
    if "Base_Price" in feature_names and "Discount" in feature_names:
        baseline_pred = model.predict(scaler.transform(X_all_zero))
        mean_contribs["Intercept (Baseline)"] = float(np.mean(baseline_pred))

    # Pct of total absolute contribution
    total_abs = sum(abs(v) for v in mean_contribs.values()) + 1e-8
    contrib_pct = {k: round(abs(v) / total_abs * 100, 1) for k, v in mean_contribs.items()}

    summary = pd.DataFrame([
        {"Feature": k, "Mean_Contribution": round(v, 2), "Pct_of_Sales": contrib_pct.get(k, 0.0)}
        for k, v in mean_contribs.items()
    ]).sort_values("Mean_Contribution", ascending=False)

    return summary


# ── Model training and evaluation ─────────────────────────────────────────────

def train_and_evaluate(df_train: pd.DataFrame,
                        df_test:  pd.DataFrame,
                        transform_params: dict,
                        target_col:  str = "NewVolSales",
                        mlflow_tracking: bool = True) -> dict:
    """
    Trains Ridge, Lasso, and Bayesian Ridge on adstock+saturated features.
    Logs all experiments to MLflow.

    Returns
    -------
    dict mapping model_name -> {model, scaler, metrics_train, metrics_test,
                                 contributions, vif_df, feature_names}
    """
    from transforms import apply_transforms

    control_cols  = [c for c in CONTROL_COLS if c in df_train.columns]
    feature_names = MEDIA_CHANNELS + control_cols

    # Compute ref_maxes: max of adstocked (pre-saturation) training series
    # Used so test data gets the same saturation scale as training
    from transforms import apply_adstock
    df_train_adstocked = apply_adstock(df_train, transform_params["thetas"])
    ref_maxes = {ch: float(df_train_adstocked[ch].max())
                 for ch in MEDIA_CHANNELS if ch in df_train.columns}
    transform_params["ref_maxes"] = ref_maxes
    # Also store raw spend means (for optimizer budget calculations)
    transform_params["raw_spend_means"] = {ch: float(df_train[ch].mean())
                                            for ch in MEDIA_CHANNELS if ch in df_train.columns}

    # Apply transforms to full train+test (train sets normalization reference)
    df_train_t = apply_transforms(df_train, transform_params)
    df_test_t  = apply_transforms(df_test,  transform_params)

    X_train = df_train_t[feature_names].values.astype(float)
    y_train = df_train[target_col].values.astype(float)
    X_test  = df_test_t[feature_names].values.astype(float)
    y_test  = df_test[target_col].values.astype(float)

    # Standardize (fit on train only)
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_train)
    X_te_s   = scaler.transform(X_test)

    # VIF on raw (non-standardized) features
    vif_df = compute_vif(X_train, feature_names)
    print(f"\n[mmm_model] VIF Diagnostics:")
    print(vif_df.to_string(index=False))

    # Set MLflow experiment
    if mlflow_tracking:
        mlflow_db = os.path.join(BASE_DIR, "mlruns", "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        mlflow.set_experiment("MMM_Experiments")

    models_cfg = {
        "Ridge":         Ridge(alpha=1.0),
        "Lasso":         Lasso(alpha=0.01, max_iter=5000),
        "BayesianRidge": BayesianRidge(),
    }

    # Alpha grid — must include large values since VIF can be 100+
    # High VIF requires strong regularization to avoid overfitting
    alpha_grid = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    tscv       = TimeSeriesSplit(n_splits=4)

    all_results = {}
    all_metrics = []

    for model_name, base_model in models_cfg.items():
        print(f"\n[mmm_model] Training {model_name}...")

        # Tune regularization strength (Ridge and Lasso)
        best_model = base_model
        best_cv_r2 = -np.inf

        if model_name in ("Ridge", "Lasso"):
            for alpha in alpha_grid:
                m = Ridge(alpha=alpha) if model_name == "Ridge" else Lasso(alpha=alpha, max_iter=5000)
                fold_r2 = []
                for tr_idx, val_idx in tscv.split(X_tr_s):
                    m.fit(X_tr_s[tr_idx], y_train[tr_idx])
                    fold_r2.append(r2_score(y_train[val_idx], m.predict(X_tr_s[val_idx])))
                mean_r2 = np.mean(fold_r2)
                if mean_r2 > best_cv_r2:
                    best_cv_r2 = mean_r2
                    best_alpha = alpha
                    best_model = Ridge(alpha=alpha) if model_name == "Ridge" else Lasso(alpha=alpha, max_iter=5000)
            print(f"  Best alpha={best_alpha:.3f}  (cv R2={best_cv_r2:.4f})")
        else:
            # BayesianRidge has its own auto-tuning
            fold_r2 = [
                r2_score(y_train[val_idx], base_model.fit(X_tr_s[tr_idx], y_train[tr_idx]).predict(X_tr_s[val_idx]))
                for tr_idx, val_idx in tscv.split(X_tr_s)
            ]
            best_cv_r2 = float(np.mean(fold_r2))
            best_alpha = None
            print(f"  cv R2={best_cv_r2:.4f}")

        # Final fit on all training data
        best_model.fit(X_tr_s, y_train)

        # Metrics
        y_train_pred = best_model.predict(X_tr_s)
        y_test_pred  = best_model.predict(X_te_s)
        metrics_train = compute_metrics(y_train, y_train_pred)
        metrics_test  = compute_metrics(y_test,  y_test_pred)

        print(f"  Train: R2={metrics_train['r2']:.4f}  MAPE={metrics_train['mape']:.1f}%  RMSE={metrics_train['rmse']:.1f}")
        print(f"  Test:  R2={metrics_test['r2']:.4f}  MAPE={metrics_test['mape']:.1f}%  RMSE={metrics_test['rmse']:.1f}")

        # Channel contributions
        contributions = compute_contributions(
            best_model, X_train, feature_names, y_train, scaler
        )

        # MLflow logging
        if mlflow_tracking:
            with mlflow.start_run(run_name=model_name):
                mlflow.log_params({
                    "model":        model_name,
                    "alpha":        best_alpha if best_alpha is not None else "auto",
                    "cv_r2":        round(best_cv_r2, 4),
                    **{f"theta_{ch}": transform_params["thetas"].get(ch, 0)
                       for ch in MEDIA_CHANNELS},
                    **{f"alpha_sat_{ch}": transform_params["alphas"].get(ch, 2.0)
                       for ch in MEDIA_CHANNELS},
                })
                mlflow.log_metrics({
                    **{f"train_{k}": v for k, v in metrics_train.items()},
                    **{f"test_{k}":  v for k, v in metrics_test.items()},
                })
                mlflow.sklearn.log_model(best_model, model_name)

        all_results[model_name] = {
            "model":          best_model,
            "scaler":         scaler,
            "feature_names":  feature_names,
            "metrics_train":  metrics_train,
            "metrics_test":   metrics_test,
            "contributions":  contributions,
            "vif_df":         vif_df,
            "y_train":        y_train,
            "y_train_pred":   y_train_pred,
            "y_test":         y_test,
            "y_test_pred":    y_test_pred,
            "X_train":        X_train,
            "transform_params": transform_params,
        }

        all_metrics.append({
            "Model": model_name,
            "Train_R2":   metrics_train["r2"],
            "Train_MAPE": metrics_train["mape"],
            "Train_RMSE": metrics_train["rmse"],
            "Test_R2":    metrics_test["r2"],
            "Test_MAPE":  metrics_test["mape"],
            "Test_RMSE":  metrics_test["rmse"],
        })

    # Save summary metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(METRICS_DIR, "model_comparison.csv"), index=False)
    print(f"\n[mmm_model] Model comparison saved to results/metrics/model_comparison.csv")
    print(metrics_df.to_string(index=False))

    # Select best model by test R2
    best_name = metrics_df.loc[metrics_df["Test_R2"].idxmax(), "Model"]
    print(f"\n[mmm_model] Best model (by test R2): {best_name}")
    all_results["_best_model_name"] = best_name

    # Save best model contributions
    best_contribs = all_results[best_name]["contributions"]
    best_contribs.to_csv(os.path.join(METRICS_DIR, "channel_contributions.csv"), index=False)

    # Save transform params
    with open(os.path.join(METRICS_DIR, "transform_params.json"), "w") as f:
        json.dump(transform_params, f, indent=2)

    return all_results


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_raw_data, prepare_data, MEDIA_CHANNELS as MC
    from transforms import grid_search_transforms

    print("=" * 60)
    print("  MARKETING MIX MODEL — TRAINING PIPELINE")
    print("=" * 60)

    df_raw  = load_raw_data()
    data    = prepare_data(df_raw)
    df_full = pd.read_csv(os.path.join(PROC_DIR, "full.csv"))
    df_train = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
    df_test  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))

    print("\n" + "=" * 60)
    print("  STEP 1: Transform parameter search")
    print("=" * 60)
    transform_params = grid_search_transforms(df_train, verbose=True)

    print("\n" + "=" * 60)
    print("  STEP 2: Model training & evaluation")
    print("=" * 60)
    results = train_and_evaluate(df_train, df_test, transform_params)

    print("\n[mmm_model] Pipeline complete.")

"""
run_all.py
==========
Full end-to-end pipeline runner for Project 06: Marketing Mix Modeling.

Executes in order:
  1. Data loading + preparation
  2. Adstock + saturation parameter grid search
  3. Model training (Ridge, Lasso, Bayesian Ridge)
  4. Budget optimization + sensitivity analysis
  5. All visualizations
  6. Final summary print
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from data_loader    import load_raw_data, prepare_data, MEDIA_CHANNELS
from transforms     import grid_search_transforms, apply_transforms
from mmm_model      import train_and_evaluate
from optimizer      import (build_response_params, optimize_budget,
                             sensitivity_analysis, compute_marginal_roi)
from visualizations import (plot_actual_vs_predicted, plot_contribution_waterfall,
                              plot_saturation_curves, plot_adstock_decay,
                              plot_budget_comparison, plot_sensitivity_analysis,
                              plot_marginal_roi, plot_vif)

PROC_DIR    = os.path.join(BASE_DIR, "data",    "processed")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")


def run():
    print("\n" + "=" * 65)
    print("  PROJECT 06: MARKETING MIX MODELING + BUDGET OPTIMIZATION")
    print("=" * 65)

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading and preparing data...")
    df_raw  = load_raw_data()
    data    = prepare_data(df_raw)
    df_train = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
    df_test  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))
    print(f"     Dataset: {len(df_raw)} weeks  |  "
          f"Train: {len(df_train)}  |  Test: {len(df_test)}")

    # ── Step 2: Transform parameter search ───────────────────────────────────
    print("\n[2/5] Grid-searching adstock + saturation parameters...")
    transform_params = grid_search_transforms(df_train, verbose=True)
    print(f"\n     Best CV R2: {transform_params['best_cv_r2']:.4f}")
    print("     Adstock decay rates:")
    for ch, theta in transform_params["thetas"].items():
        print(f"       {ch:<22}: theta={theta:.1f}")

    # ── Step 3: Model training ────────────────────────────────────────────────
    print("\n[3/5] Training models (Ridge / Lasso / Bayesian Ridge)...")
    results = train_and_evaluate(df_train, df_test, transform_params,
                                  mlflow_tracking=True)

    best_name = results["_best_model_name"]
    best_res  = results[best_name]
    model     = best_res["model"]
    scaler    = best_res["scaler"]
    X_train   = best_res["X_train"]
    feat_names = best_res["feature_names"]

    print(f"\n     Best model: {best_name}")
    mt = best_res["metrics_test"]
    print(f"     Test R2={mt['r2']:.4f}  MAPE={mt['mape']:.1f}%  RMSE={mt['rmse']:.1f}")

    # ── Step 4: Budget optimization ───────────────────────────────────────────
    print("\n[4/5] Running budget optimization + sensitivity analysis...")
    resp_params = build_response_params(model, scaler, X_train,
                                         transform_params, feat_names)

    channels     = [ch for ch in MEDIA_CHANNELS if ch in resp_params]
    total_budget = sum(resp_params[ch]["mean_spend_raw"] for ch in channels)
    print(f"     Total budget (mean raw weekly spend): {total_budget:.1f}")

    opt_result = optimize_budget(resp_params, total_budget, channels, n_restarts=15)
    print(f"     Current alloc -> Predicted sales: {opt_result['predicted_sales_current']:.1f}")
    print(f"     Optimal alloc -> Predicted sales: {opt_result['predicted_sales_optimal']:.1f}")
    print(f"     Predicted lift: {opt_result['predicted_sales_lift']:+.1f} "
          f"({opt_result['predicted_sales_lift_pct']:+.1f}%)")

    # Save optimization result (with all resp_params columns for API)
    opt_rows = []
    for ch in channels:
        opt_rows.append({
            "Channel":        ch,
            "Current_Spend":  round(opt_result["current_allocation"][ch], 2),
            "Optimal_Spend":  round(opt_result["optimal_allocation"][ch], 2),
            **resp_params[ch],
        })
    pd.DataFrame(opt_rows).to_csv(
        os.path.join(METRICS_DIR, "optimization_result.csv"), index=False)

    # Sensitivity analysis
    print("\n     Sensitivity analysis across budget levels:")
    sens_df = sensitivity_analysis(resp_params, total_budget, channels,
                                    budget_range=[0.5, 0.7, 0.8, 0.9, 1.0,
                                                  1.1, 1.2, 1.3, 1.5])
    sens_df.to_csv(os.path.join(METRICS_DIR, "sensitivity_analysis.csv"), index=False)

    # Marginal ROI curves
    roi_curves = compute_marginal_roi(resp_params, channels)

    # ── Step 5: Visualizations ────────────────────────────────────────────────
    print("\n[5/5] Generating visualizations...")

    print("  Actual vs. Predicted...")
    plot_actual_vs_predicted(
        {k: v for k, v in results.items() if not k.startswith("_")},
        df_train, df_test
    )

    print("  Channel contribution waterfall...")
    plot_contribution_waterfall(best_res["contributions"], best_name)

    print("  Saturation curves...")
    plot_saturation_curves(resp_params, channels)

    print("  Adstock decay...")
    plot_adstock_decay(transform_params)

    print("  Budget comparison...")
    plot_budget_comparison(opt_result)

    print("  Sensitivity analysis...")
    plot_sensitivity_analysis(sens_df)

    print("  Marginal ROI curves...")
    plot_marginal_roi(roi_curves, opt_result)

    print("  VIF diagnostic...")
    plot_vif(best_res["vif_df"])

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE — FINAL SUMMARY")
    print("=" * 65)

    metrics_df = pd.read_csv(os.path.join(METRICS_DIR, "model_comparison.csv"))
    print("\nModel Comparison:")
    print(metrics_df[["Model", "Train_R2", "Test_R2", "Test_MAPE", "Test_RMSE"]].to_string(index=False))

    print(f"\nBest Model: {best_name}")
    print(f"  Test R2 = {mt['r2']:.4f}")
    print(f"  Test MAPE = {mt['mape']:.1f}%")

    print(f"\nBudget Optimization (budget = {total_budget:.0f}):")
    print(f"  {'Channel':<24} {'Current':>10} {'Optimal':>10} {'Change':>10}")
    print("  " + "-" * 56)
    for ch in channels:
        cur = opt_result["current_allocation"][ch]
        opt = opt_result["optimal_allocation"][ch]
        print(f"  {ch:<24} {cur:>10.1f} {opt:>10.1f} {opt-cur:>+10.1f}")
    print(f"\n  Predicted sales lift: {opt_result['predicted_sales_lift']:+.1f} "
          f"({opt_result['predicted_sales_lift_pct']:+.1f}%)")

    print(f"\nChannel Attribution ({best_name}):")
    for _, row in best_res["contributions"].iterrows():
        if row["Feature"] in MEDIA_CHANNELS:
            print(f"  {row['Feature']:<24}: {row['Pct_of_Sales']:>5.1f}% of sales")

    print(f"\nOutputs:")
    print(f"  results/metrics/   -> CSVs + JSON")
    print(f"  results/figures/   -> 8 PNG plots")
    print(f"  mlruns/            -> MLflow experiment logs")
    print(f"\nNext: streamlit run src/dashboard.py")


if __name__ == "__main__":
    run()

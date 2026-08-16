"""
optimizer.py
============
Budget reallocation optimization using the fitted MMM response curves.

The core formulation (Nonlinear Program):
    max  sum_i { beta_i * Hill(Adstock(x_i; theta_i); alpha_i, gamma_i) }
    s.t. sum_i x_i = B          (total budget constraint)
         x_i >= 0               (no negative spend)

Solved via scipy.optimize (SLSQP) — analytically tractable convex NLP
since Hill functions are concave for alpha >= 1.

Additional outputs:
  - Sensitivity analysis: optimal allocation as budget varies ±30%
  - Scenario comparison: actual vs. optimal allocation -> predicted sales lift
  - Marginal ROI curves: incremental return per unit of extra spend per channel
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
import warnings
warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

MEDIA_CHANNELS = ["TV", "Radio", "InStore", "NewspaperInserts", "Website_Campaign"]


# ── Channel response function (adstock + saturation + linear coefficient) ──────

def channel_response(spend: float, coef: float,
                      theta: float, alpha: float, gamma: float,
                      max_spend_ref: float) -> float:
    """
    Predict incremental sales from a given spend level for one channel.
    This is the 'ROI curve' for that channel.

    spend         : spend level (same units as training data)
    coef          : fitted regression coefficient (from best model, unstandardized)
    theta         : adstock decay rate
    alpha, gamma  : Hill saturation parameters
    max_spend_ref : reference max spend (used for Hill normalization, from training data)

    Notes
    -----
    For optimization we use the SINGLE-PERIOD (no time dimension) approximation:
        effective_spend = spend / (1 - theta)   (steady-state adstock)
    This is the standard approach when optimizing a "next period" budget.
    """
    # Steady-state adstock approximation (assuming constant spend going forward)
    eff_spend = spend / (1.0 - theta + 1e-8)

    # Hill saturation
    x_norm = eff_spend / (max_spend_ref + 1e-8)
    saturated = x_norm ** alpha / (x_norm ** alpha + gamma ** alpha)

    return coef * saturated


def build_response_params(model, scaler, X_train: np.ndarray,
                           transform_params: dict,
                           feature_names: list) -> dict:
    """
    Extract per-channel response parameters from the fitted model.

    X_train      : transformed (adstocked+saturated) training features
    transform_params : must include 'ref_maxes' (adstocked max per channel, from training)
                       and 'raw_spend_means' (raw spend mean per channel)

    The response function for channel i, given raw spend x_i:
        eff   = x_i / (1 - theta_i)               # steady-state adstock
        sat   = Hill(eff / ref_max_i; alpha, gamma) # same normalization as training
        delta = (sat - mean_sat_i) * coef_std_i / std_sat_i  # de-standardize

    Returns dict:
      {channel: {coef_std, std_sat, mean_sat, theta, alpha, gamma,
                 ref_max_adstocked, mean_spend_raw}}
    """
    coefs_std = model.coef_      # coefficients in standardized feature space
    feat_stds  = scaler.scale_   # std of each feature (after saturation, before StandardScaler)
    feat_means = scaler.mean_    # mean of each feature

    resp_params = {}
    for ch in MEDIA_CHANNELS:
        if ch not in feature_names:
            continue
        fi = feature_names.index(ch)
        resp_params[ch] = {
            "coef_std":          float(coefs_std[fi]),
            "std_sat":           float(feat_stds[fi]),
            "mean_sat":          float(feat_means[fi]),
            "theta":             float(transform_params["thetas"].get(ch, 0.0)),
            "alpha":             float(transform_params["alphas"].get(ch, 2.0)),
            "gamma":             float(transform_params["gammas"].get(ch, 0.5)),
            "ref_max_adstocked": float(transform_params.get("ref_maxes", {}).get(ch, X_train[:, fi].max() * 10)),
            "mean_spend_raw":    float(transform_params.get("raw_spend_means", {}).get(ch,
                                       X_train[:, fi].mean())),
        }

    return resp_params


# ── Channel response function ──────────────────────────────────────────────────

def channel_response(spend: float, p: dict) -> float:
    """
    Predict incremental sales from a given RAW spend level for one channel.
    This is the 'ROI curve' for that channel.

    spend : raw spend level (same units as original training data, e.g. INR thousands)
    p     : dict from build_response_params with keys:
              coef_std, std_sat, mean_sat, theta, alpha, gamma,
              ref_max_adstocked, mean_spend_raw

    Method:
      1. Steady-state adstock: eff = spend / (1 - theta)
         Assumes company has been spending `spend` constantly -> series reaches steady state
      2. Hill saturation using training-set ref_max (same normalization as model training):
         x_norm = eff / ref_max_adstocked
         sat    = x_norm^alpha / (x_norm^alpha + gamma^alpha)
      3. De-standardize:
         contribution = coef_std * (sat - mean_sat) / std_sat
         This gives the contribution in original sales units (centered around training mean).
    """
    theta = p["theta"]
    alpha = p["alpha"]
    gamma = p["gamma"]
    ref   = p["ref_max_adstocked"]

    # Steady-state adstock
    eff_spend = spend / (1.0 - theta + 1e-8)

    # Hill saturation (same normalization as training)
    x_norm    = eff_spend / (ref + 1e-8)
    sat       = x_norm ** alpha / (x_norm ** alpha + gamma ** alpha)

    # De-standardize: from standardized space back to sales units
    contribution = p["coef_std"] * (sat - p["mean_sat"]) / (p["std_sat"] + 1e-8)
    return contribution


def total_sales_objective(x: np.ndarray, resp_params: dict,
                           channels: list) -> float:
    """Negative total predicted sales (minimized by scipy)."""
    total = 0.0
    for i, ch in enumerate(channels):
        total += channel_response(x[i], resp_params[ch])
    return -total  # negative because we minimize


def optimize_budget(resp_params: dict,
                    total_budget: float,
                    channels: list = None,
                    n_restarts: int = 10) -> dict:
    """
    Solve the budget reallocation NLP.

    Parameters
    ----------
    resp_params   : output of build_response_params()
    total_budget  : total marketing budget in RAW SPEND UNITS (e.g. weekly INR thousands)
    n_restarts    : number of random restarts to avoid local optima

    Returns
    -------
    dict with:
      'optimal_allocation': {channel: optimal_spend},
      'predicted_sales_lift': float
      'predicted_sales_lift_pct': float
      'convergence': bool
    """
    if channels is None:
        channels = [ch for ch in MEDIA_CHANNELS if ch in resp_params]

    n = len(channels)
    best_result = None
    best_obj    = np.inf

    # Budget equality constraint: sum(x) = total_budget
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - total_budget}]
    bounds      = Bounds(lb=np.zeros(n), ub=np.full(n, total_budget))

    for restart in range(n_restarts):
        # Random initialization that sums to total_budget
        x0 = np.random.dirichlet(np.ones(n)) * total_budget

        result = minimize(
            total_sales_objective,
            x0,
            args=(resp_params, channels),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        if result.fun < best_obj:
            best_obj    = result.fun
            best_result = result

    # Predicted sales at optimal vs. current (mean raw spend) allocation
    optimal_spend = dict(zip(channels, best_result.x))

    # Current (actual mean raw spend) allocation — scaled to match total_budget
    current_total_raw = sum(resp_params[ch]["mean_spend_raw"] for ch in channels)
    current_spend = {ch: resp_params[ch]["mean_spend_raw"] * (total_budget / (current_total_raw + 1e-8))
                     for ch in channels}

    def pred_sales(alloc: dict) -> float:
        return sum(channel_response(alloc[ch], resp_params[ch]) for ch in channels)

    sales_optimal = pred_sales(optimal_spend)
    sales_current = pred_sales(current_spend)
    sales_lift    = sales_optimal - sales_current
    sales_lift_pct = (sales_lift / (abs(sales_current) + 1e-8)) * 100

    return {
        "optimal_allocation":      optimal_spend,
        "current_allocation":      current_spend,
        "predicted_sales_optimal": sales_optimal,
        "predicted_sales_current": sales_current,
        "predicted_sales_lift":    sales_lift,
        "predicted_sales_lift_pct": sales_lift_pct,
        "convergence":             best_result.success,
        "total_budget":            total_budget,
        "channels":                channels,
    }



# ── Sensitivity analysis ───────────────────────────────────────────────────────

def sensitivity_analysis(resp_params: dict,
                          base_budget: float,
                          channels: list = None,
                          budget_range: list = None) -> pd.DataFrame:
    """
    Run optimization across a range of total budgets.
    Shows how optimal channel allocation shifts as budget changes.

    Returns DataFrame with columns: budget, channel1_spend, ..., predicted_sales
    """
    if channels is None:
        channels = [ch for ch in MEDIA_CHANNELS if ch in resp_params]
    if budget_range is None:
        budget_range = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

    records = []
    for pct in budget_range:
        budget = base_budget * pct
        result = optimize_budget(resp_params, budget, channels, n_restarts=5)
        row = {"Budget_pct": pct, "Total_Budget": round(budget, 1),
               "Predicted_Sales": round(result["predicted_sales_optimal"], 1)}
        for ch in channels:
            row[ch] = round(result["optimal_allocation"][ch], 1)
        records.append(row)
        print(f"  Budget {pct*100:.0f}%: Sales={result['predicted_sales_optimal']:.1f}  "
              f"({'+' if result['predicted_sales_lift_pct'] >= 0 else ''}"
              f"{result['predicted_sales_lift_pct']:.1f}% vs proportional)")

    return pd.DataFrame(records)


# ── Marginal ROI curves ────────────────────────────────────────────────────────

def compute_marginal_roi(resp_params: dict,
                          channels: list = None,
                          n_points: int = 50) -> dict:
    """
    Compute the marginal ROI curve for each channel:
    dSales/dSpend as a function of spend level.

    Returns dict: {channel: {"spend": array, "marginal_roi": array, "total_sales": array}}
    """
    if channels is None:
        channels = [ch for ch in MEDIA_CHANNELS if ch in resp_params]

    curves = {}
    for ch in channels:
        p     = resp_params[ch]
        spend = np.linspace(0, p["mean_spend_raw"] * 3, n_points)
        sales = np.array([channel_response(s, p) for s in spend])
        # Marginal ROI = dSales/dSpend (numerical derivative)
        mroi = np.gradient(sales, spend)
        curves[ch] = {"spend": spend, "sales": sales, "marginal_roi": mroi}

    return curves


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_raw_data, prepare_data
    from transforms import grid_search_transforms, apply_transforms
    from mmm_model import train_and_evaluate
    import pandas as pd

    print("=" * 60)
    print("  BUDGET OPTIMIZATION PIPELINE")
    print("=" * 60)

    df_raw   = load_raw_data()
    data     = prepare_data(df_raw)
    df_train = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
    df_test  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))

    transform_params = grid_search_transforms(df_train, verbose=False)
    results          = train_and_evaluate(df_train, df_test, transform_params,
                                           mlflow_tracking=True)

    best_name = results["_best_model_name"]
    best_res  = results[best_name]
    model     = best_res["model"]
    scaler    = best_res["scaler"]
    X_train   = best_res["X_train"]
    feat_names = best_res["feature_names"]

    resp_params = build_response_params(model, scaler, X_train, transform_params, feat_names)

    # Total budget = sum of average weekly spend across all channels
    channels     = [ch for ch in MEDIA_CHANNELS if ch in resp_params]
    total_budget = sum(resp_params[ch]["mean_spend_raw"] for ch in channels)
    print(f"     Total budget (mean raw weekly spend): {total_budget:.1f}")

    print("\n[optimizer] Solving budget optimization...")
    opt_result = optimize_budget(resp_params, total_budget, channels)

    print(f"\n--- Optimization Result ---")
    print(f"Convergence: {opt_result['convergence']}")
    print(f"Current  allocation -> Predicted sales: {opt_result['predicted_sales_current']:.1f}")
    print(f"Optimal  allocation -> Predicted sales: {opt_result['predicted_sales_optimal']:.1f}")
    print(f"Predicted lift: {opt_result['predicted_sales_lift']:.1f}  "
          f"({opt_result['predicted_sales_lift_pct']:+.1f}%)")

    print(f"\n{'Channel':<24} {'Current':>10} {'Optimal':>10} {'Change':>10}")
    print("-" * 56)
    for ch in channels:
        cur = opt_result["current_allocation"][ch]
        opt = opt_result["optimal_allocation"][ch]
        print(f"  {ch:<22} {cur:>10.1f} {opt:>10.1f} {opt-cur:>+10.1f}")

    # Save optimization result
    opt_df = pd.DataFrame([{
        "Channel": ch,
        "Current_Spend":  opt_result["current_allocation"][ch],
        "Optimal_Spend":  opt_result["optimal_allocation"][ch],
        "Change":         opt_result["optimal_allocation"][ch] - opt_result["current_allocation"][ch],
        **{k: v for k, v in resp_params[ch].items() if k != "max_spend_ref"},
    } for ch in channels])
    opt_df.to_csv(os.path.join(METRICS_DIR, "optimization_result.csv"), index=False)

    # Sensitivity analysis
    print(f"\n[optimizer] Sensitivity analysis (budget range 50%-150%)...")
    sens_df = sensitivity_analysis(resp_params, total_budget, channels)
    sens_df.to_csv(os.path.join(METRICS_DIR, "sensitivity_analysis.csv"), index=False)

    # Marginal ROI
    roi_curves = compute_marginal_roi(resp_params, channels)
    for ch, curve in roi_curves.items():
        pd.DataFrame({"spend": curve["spend"], "sales": curve["sales"],
                      "marginal_roi": curve["marginal_roi"]}).to_csv(
            os.path.join(METRICS_DIR, f"roi_curve_{ch}.csv"), index=False)

    print(f"\n[optimizer] All results saved to {METRICS_DIR}")

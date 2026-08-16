"""
visualizations.py
=================
All plotting utilities for the MMM project:
  1. Actual vs. Predicted sales (train/test)
  2. Channel contribution waterfall chart
  3. Saturation (ROI) curves per channel
  4. Adstock decay visualization
  5. Optimal vs. actual budget allocation (bar chart)
  6. Sensitivity analysis (budget sweep)
  7. Marginal ROI curves overlay
  8. VIF heatmap / bar chart
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = {
    "TV":               "#E63946",
    "Radio":            "#457B9D",
    "InStore":          "#2A9D8F",
    "NewspaperInserts": "#E9C46A",
    "Website_Campaign": "#F4A261",
    "Intercept (Baseline)": "#A8DADC",
}
DEFAULT_COLORS = list(PALETTE.values())

MEDIA_CHANNELS = ["TV", "Radio", "InStore", "NewspaperInserts", "Website_Campaign"]


def savefig(name: str, dpi: int = 150):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 1. Actual vs. Predicted ───────────────────────────────────────────────────

def plot_actual_vs_predicted(results: dict, df_train: pd.DataFrame,
                              df_test: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig.suptitle("Actual vs. Predicted Sales — Model Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, (model_name, res) in zip(axes, {k: v for k, v in results.items()
                                             if not k.startswith("_")}.items()):
        n_train = len(res["y_train"])
        n_test  = len(res["y_test"])

        ax.plot(range(n_train), res["y_train"],       color="black", lw=1.5,
                label="Actual (train)", alpha=0.8)
        ax.plot(range(n_train), res["y_train_pred"],  color="steelblue", lw=1.5,
                ls="--", label=f"{model_name} fit (train)")
        ax.plot(range(n_train, n_train + n_test), res["y_test"],
                color="black", lw=1.5, alpha=0.8)
        ax.plot(range(n_train, n_train + n_test), res["y_test_pred"],
                color="crimson", lw=1.5, ls="--", label=f"{model_name} pred (test)")

        ax.axvline(n_train - 0.5, color="gray", ls=":", lw=1, alpha=0.7)
        ax.text(n_train + 0.5, ax.get_ylim()[0], "Test ->", fontsize=8, color="gray")

        mt = res["metrics_test"]
        ax.set_title(f"{model_name}  |  Test: R2={mt['r2']:.3f}  "
                     f"MAPE={mt['mape']:.1f}%  RMSE={mt['rmse']:.0f}",
                     fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylabel("Sales Volume", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("actual_vs_predicted.png")


# ── 2. Channel contribution waterfall ─────────────────────────────────────────

def plot_contribution_waterfall(contributions: pd.DataFrame, model_name: str = ""):
    df = contributions.sort_values("Mean_Contribution", ascending=True)
    colors = [PALETTE.get(f, "#888888") for f in df["Feature"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["Feature"], df["Mean_Contribution"], color=colors, edgecolor="white")

    for bar, pct in zip(bars, df["Pct_of_Sales"]):
        w = bar.get_width()
        ax.text(w + max(df["Mean_Contribution"]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Mean Contribution to Sales", fontsize=11)
    ax.set_title(f"Channel Sales Attribution — {model_name}\n"
                 "(% of total attributed sales from each channel)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig(f"contribution_waterfall_{model_name.replace(' ', '_')}.png")


# ── 3. Saturation curves ───────────────────────────────────────────────────────

def plot_saturation_curves(resp_params: dict, channels: list = None):
    if channels is None:
        channels = [ch for ch in MEDIA_CHANNELS if ch in resp_params]

    n   = len(channels)
    fig = plt.figure(figsize=(16, 4 * ((n + 2) // 3)))
    fig.suptitle("Channel Saturation Curves (Diminishing Returns)\n"
                 "Each curve shows predicted sales as spend increases",
                 fontsize=13, fontweight="bold")

    for i, ch in enumerate(channels):
        ax  = fig.add_subplot((n + 2) // 3, 3, i + 1)
        p   = resp_params[ch]

        # Use mean_spend_raw as reference for spend axis
        max_raw = p.get("mean_spend_raw", p.get("max_spend_ref", 100)) * 3
        spend   = np.linspace(0, max_raw, 200)
        sales   = []
        for s in spend:
            eff   = s / (1.0 - p["theta"] + 1e-8)
            ref   = p.get("ref_max_adstocked", p.get("max_spend_ref", 1.0))
            xn    = eff / (ref + 1e-8)
            sat   = xn ** p["alpha"] / (xn ** p["alpha"] + p["gamma"] ** p["alpha"])
            sales.append(p["coef_std"] * (sat - p.get("mean_sat", 0)) / (p.get("std_sat", 1) + 1e-8))
        sales = np.array(sales)

        color = PALETTE.get(ch, "#555555")
        ax.plot(spend, sales, color=color, lw=2.5)
        ax.axvline(p.get("mean_spend_raw", 50), color="black", ls="--", lw=1, alpha=0.6,
                   label=f"Avg spend={p.get('mean_spend_raw', 0):.0f}")
        ax.fill_between(spend, sales, alpha=0.1, color=color)
        ax.set_title(ch, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Weekly Spend", fontsize=9)
        ax.set_ylabel("Incremental Sales", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("saturation_curves.png")


# ── 4. Adstock decay visualization ────────────────────────────────────────────

def plot_adstock_decay(transform_params: dict, n_weeks: int = 12):
    thetas = transform_params["thetas"]
    channels = [ch for ch in MEDIA_CHANNELS if ch in thetas]

    fig, ax = plt.subplots(figsize=(10, 5))
    weeks   = np.arange(n_weeks)

    for ch in channels:
        theta = thetas[ch]
        decay = theta ** weeks        # impulse response: 1 unit spent at t=0
        color = PALETTE.get(ch, "#555555")
        ax.plot(weeks, decay, marker="o", lw=2, ms=5, color=color,
                label=f"{ch}  (theta={theta:.1f})")

    ax.set_xlabel("Weeks after Advertising", fontsize=11)
    ax.set_ylabel("Remaining Effect (fraction of original)", fontsize=11)
    ax.set_title("Adstock Decay by Channel\n"
                 "How long does 1 unit of advertising spend keep affecting sales?",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    savefig("adstock_decay.png")


# ── 5. Optimal vs. actual budget allocation ───────────────────────────────────

def plot_budget_comparison(opt_result: dict):
    channels  = opt_result["channels"]
    current   = [opt_result["current_allocation"][ch]  for ch in channels]
    optimal   = [opt_result["optimal_allocation"][ch]  for ch in channels]

    x    = np.arange(len(channels))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    b1 = ax.bar(x - w/2, current, w, label="Current Allocation",
                color=[PALETTE.get(ch, "#888") for ch in channels], alpha=0.6, edgecolor="white")
    b2 = ax.bar(x + w/2, optimal, w, label="Optimal Allocation",
                color=[PALETTE.get(ch, "#888") for ch in channels], alpha=1.0, edgecolor="white")

    # Annotate change
    for xi, (c, o) in enumerate(zip(current, optimal)):
        change = o - c
        ax.text(xi + w/2, o + max(optimal) * 0.01,
                f"{'+' if change >= 0 else ''}{change:.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="green" if change > 0 else "red")

    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Weekly Spend (budget units)", fontsize=11)
    ax.set_title(
        f"Budget Reallocation: Current vs. Optimal\n"
        f"Predicted sales lift: {opt_result['predicted_sales_lift']:+.1f} units  "
        f"({opt_result['predicted_sales_lift_pct']:+.1f}%)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig("budget_comparison.png")


# ── 6. Sensitivity analysis ───────────────────────────────────────────────────

def plot_sensitivity_analysis(sens_df: pd.DataFrame):
    channels = [c for c in MEDIA_CHANNELS if c in sens_df.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: predicted sales vs. budget
    ax1.plot(sens_df["Budget_pct"] * 100, sens_df["Predicted_Sales"],
             marker="o", lw=2, color="steelblue")
    ax1.axvline(100, color="black", ls="--", lw=1, alpha=0.5, label="Current budget")
    ax1.set_xlabel("Budget Level (% of current)", fontsize=11)
    ax1.set_ylabel("Predicted Sales", fontsize=11)
    ax1.set_title("Sales vs. Budget Level\n(Optimal allocation at each budget)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: channel allocations vs. budget (stacked area)
    budget_pcts = sens_df["Budget_pct"] * 100
    bottoms = np.zeros(len(sens_df))
    for ch in channels:
        if ch not in sens_df.columns:
            continue
        vals = sens_df[ch].values
        ax2.fill_between(budget_pcts, bottoms, bottoms + vals,
                         alpha=0.8, color=PALETTE.get(ch, "#888"), label=ch)
        bottoms += vals

    ax2.axvline(100, color="black", ls="--", lw=1, alpha=0.5)
    ax2.set_xlabel("Budget Level (% of current)", fontsize=11)
    ax2.set_ylabel("Optimal Channel Spend", fontsize=11)
    ax2.set_title("Channel Allocation Shifts with Budget\n(How reallocation changes as total budget varies)",
                  fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("sensitivity_analysis.png")


# ── 7. Marginal ROI curves overlay ────────────────────────────────────────────

def plot_marginal_roi(roi_curves: dict, opt_result: dict = None):
    channels = list(roi_curves.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total sales curves
    for ch in channels:
        c = roi_curves[ch]
        axes[0].plot(c["spend"], c["sales"], lw=2, color=PALETTE.get(ch, "#888"), label=ch)
    axes[0].set_xlabel("Weekly Spend (raw units)", fontsize=11)
    axes[0].set_ylabel("Predicted Incremental Sales", fontsize=11)
    axes[0].set_title("Channel Response Curves\n(Diminishing returns visible as curves flatten)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: marginal ROI curves
    for ch in channels:
        c = roi_curves[ch]
        mroi = np.clip(c["marginal_roi"], 0, None)   # clip negatives (numerical artefact)
        axes[1].plot(c["spend"], mroi, lw=2, color=PALETTE.get(ch, "#888"), label=ch)
        if opt_result is not None and ch in opt_result["optimal_allocation"]:
            opt_x = opt_result["optimal_allocation"][ch]
            # Find marginal ROI at optimal point
            idx = np.searchsorted(c["spend"], opt_x)
            idx = min(idx, len(mroi) - 1)
            axes[1].axvline(opt_x, color=PALETTE.get(ch, "#888"), ls=":", lw=1, alpha=0.6)

    axes[1].set_xlabel("Weekly Spend", fontsize=11)
    axes[1].set_ylabel("Marginal ROI (sales per unit spend)", fontsize=11)
    axes[1].set_title("Marginal ROI Curves\n(Where curves converge = equal marginal returns = optimal split)",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("marginal_roi_curves.png")


# ── 8. VIF bar chart ──────────────────────────────────────────────────────────

def plot_vif(vif_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors  = ["crimson" if v > 10 else ("orange" if v > 5 else "steelblue")
               for v in vif_df["VIF"]]
    ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors, edgecolor="white")
    ax.axvline(5,  color="orange", ls="--", lw=1.5, label="VIF=5 (moderate)")
    ax.axvline(10, color="red",    ls="--", lw=1.5, label="VIF=10 (high)")
    ax.set_xlabel("Variance Inflation Factor (VIF)", fontsize=11)
    ax.set_title("Multicollinearity Diagnostic (VIF)\nVIF>10 = severe multicollinearity",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    savefig("vif_diagnostic.png")


if __name__ == "__main__":
    print("Visualizations module — run via main.py or run_all.py")

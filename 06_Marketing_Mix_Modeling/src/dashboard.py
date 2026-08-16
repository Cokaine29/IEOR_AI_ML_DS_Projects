"""
dashboard.py
============
Streamlit interactive dashboard for the MMM project.

Sections:
  1. Sidebar: budget slider + model selector
  2. KPI cards: predicted sales, lift, convergence
  3. Budget allocation chart (current vs. optimal)
  4. Channel contribution waterfall
  5. Saturation curves (interactive per channel)
  6. Sensitivity analysis plot
  7. Marginal ROI curves

Usage:
  streamlit run src/dashboard.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

sys.path.insert(0, os.path.dirname(__file__))
from optimizer import optimize_budget, sensitivity_analysis, compute_marginal_roi, channel_response

MEDIA_CHANNELS = ["TV", "Radio", "InStore", "NewspaperInserts", "Website_Campaign"]
PALETTE = {
    "TV":               "#E63946",
    "Radio":            "#457B9D",
    "InStore":          "#2A9D8F",
    "NewspaperInserts": "#E9C46A",
    "Website_Campaign": "#F4A261",
}


# ── Load state ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    tp_path  = os.path.join(METRICS_DIR, "transform_params.json")
    opt_path = os.path.join(METRICS_DIR, "optimization_result.csv")
    contrib_path = os.path.join(METRICS_DIR, "channel_contributions.csv")
    model_path   = os.path.join(METRICS_DIR, "model_comparison.csv")

    try:
        with open(tp_path) as f:
            transform_params = json.load(f)
        opt_df      = pd.read_csv(opt_path)
        contrib_df  = pd.read_csv(contrib_path) if os.path.exists(contrib_path) else pd.DataFrame()
        model_df    = pd.read_csv(model_path)   if os.path.exists(model_path)   else pd.DataFrame()

        resp_params = {}
        for _, row in opt_df.iterrows():
            ch = row["Channel"]
            resp_params[ch] = {
                "coef":          float(row["coef"]),
                "theta":         float(row["theta"]),
                "alpha":         float(row["alpha"]),
                "gamma":         float(row["gamma"]),
                "mean_spend":    float(row["mean_spend"]),
                "max_spend_ref": float(opt_df["Current_Spend"].max() * 1.5),
            }
        base_budget = float(opt_df["Current_Spend"].sum())
        return resp_params, transform_params, contrib_df, model_df, base_budget
    except FileNotFoundError:
        return {}, {}, pd.DataFrame(), pd.DataFrame(), 500.0


# ── App layout ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Marketing Mix Modeling Dashboard",
        page_icon="📊",
        layout="wide",
    )

    # Header
    st.markdown(
        "<h1 style='text-align:center; color:#E63946;'>📊 Marketing Mix Modeling</h1>"
        "<p style='text-align:center; color:#555; font-size:16px;'>"
        "Budget Reallocation Optimization — IEOR IIT Bombay</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    resp_params, transform_params, contrib_df, model_df, base_budget = load_data()

    if not resp_params:
        st.error("No results found. Please run the full pipeline first:\n"
                 "`python src/mmm_model.py`  then  `python src/optimizer.py`")
        return

    channels = [ch for ch in MEDIA_CHANNELS if ch in resp_params]

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header("Budget Controls")
    budget_pct    = st.sidebar.slider("Total Budget (% of current)", 50, 150, 100, step=5)
    total_budget  = base_budget * (budget_pct / 100)
    st.sidebar.metric("Weekly Budget", f"{total_budget:.0f}")
    n_restarts    = st.sidebar.select_slider("Optimizer restarts", [5, 10, 20], value=10)

    st.sidebar.markdown("---")
    st.sidebar.header("Model Performance")
    if not model_df.empty:
        st.sidebar.dataframe(
            model_df[["Model", "Test_R2", "Test_MAPE"]].rename(
                columns={"Test_R2": "R²", "Test_MAPE": "MAPE (%)"}
            ).set_index("Model"),
            use_container_width=True,
        )

    # ── Run optimization ───────────────────────────────────────────────────────
    with st.spinner("Optimizing budget allocation..."):
        opt_result = optimize_budget(resp_params, total_budget, channels, n_restarts)

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Budget", f"{total_budget:.0f}")
    c2.metric("Sales (Current Alloc)", f"{opt_result['predicted_sales_current']:.0f}")
    c3.metric("Sales (Optimal Alloc)", f"{opt_result['predicted_sales_optimal']:.0f}",
              delta=f"{opt_result['predicted_sales_lift']:+.0f}")
    c4.metric("Predicted Lift",
              f"{opt_result['predicted_sales_lift_pct']:+.1f}%",
              delta_color="normal")

    st.markdown("---")

    # ── Budget Allocation Chart ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Budget Allocation: Current vs. Optimal")
        fig, ax = plt.subplots(figsize=(7, 4))
        x  = np.arange(len(channels))
        w  = 0.35
        ax.bar(x - w/2, [opt_result["current_allocation"][c] for c in channels],
               w, color=[PALETTE.get(c, "#888") for c in channels], alpha=0.6,
               label="Current", edgecolor="white")
        ax.bar(x + w/2, [opt_result["optimal_allocation"][c] for c in channels],
               w, color=[PALETTE.get(c, "#888") for c in channels], alpha=1.0,
               label="Optimal", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Spend", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Channel Attribution (% of Sales)")
        if not contrib_df.empty:
            media_contribs = contrib_df[contrib_df["Feature"].isin(MEDIA_CHANNELS)]
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            colors = [PALETTE.get(r["Feature"], "#888") for _, r in media_contribs.iterrows()]
            ax2.barh(media_contribs["Feature"], media_contribs["Pct_of_Sales"],
                     color=colors, edgecolor="white")
            ax2.set_xlabel("% of Attributed Sales", fontsize=10)
            ax2.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    st.markdown("---")

    # ── Saturation Curves ──────────────────────────────────────────────────────
    st.subheader("Channel Response Curves (Diminishing Returns)")
    sel_channel = st.selectbox("Select channel", channels)
    if sel_channel and sel_channel in resp_params:
        p     = resp_params[sel_channel]
        spend = np.linspace(0, p["max_spend_ref"] * 1.5, 200)
        sales = []
        for s in spend:
            eff  = s / (1.0 - p["theta"] + 1e-8)
            xn   = eff / (p["max_spend_ref"] + 1e-8)
            sat  = xn ** p["alpha"] / (xn ** p["alpha"] + p["gamma"] ** p["alpha"])
            sales.append(p["coef"] * sat)
        sales = np.array(sales)

        fig3, ax3 = plt.subplots(figsize=(9, 3.5))
        ax3.plot(spend, sales, color=PALETTE.get(sel_channel, "#555"), lw=2.5)
        ax3.fill_between(spend, sales, alpha=0.1, color=PALETTE.get(sel_channel, "#555"))
        ax3.axvline(p["mean_spend"], color="black", ls="--", lw=1.5,
                    label=f"Avg spend = {p['mean_spend']:.0f}")
        ax3.axvline(opt_result["optimal_allocation"].get(sel_channel, 0),
                    color="green", ls="-.", lw=1.5, label="Optimal spend")
        ax3.set_xlabel("Weekly Spend", fontsize=11)
        ax3.set_ylabel("Incremental Sales", fontsize=11)
        ax3.set_title(f"{sel_channel} — Response Curve (theta={p['theta']:.1f}, "
                      f"alpha={p['alpha']:.1f}, gamma={p['gamma']:.1f})", fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # ── Reallocation detail table ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Reallocation Detail")
    rows = []
    for ch in channels:
        cur = opt_result["current_allocation"][ch]
        opt = opt_result["optimal_allocation"][ch]
        rows.append({
            "Channel": ch, "Current": round(cur, 1), "Optimal": round(opt, 1),
            "Change": round(opt - cur, 1),
            "Change %": f"{(opt - cur) / (cur + 1e-8) * 100:+.1f}%",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Channel"), use_container_width=True)

    st.markdown(
        "<p style='color:#999; font-size:12px; text-align:center;'>"
        "Project 6: Marketing Mix Modeling | IEOR IIT Bombay | Ridge + Lasso + Bayesian Ridge + NLP Optimization"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

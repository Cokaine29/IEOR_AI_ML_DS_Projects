"""
sector_backtester.py
====================
Runs a rolling out-of-sample backtest across:
  - 6 market sectors
  - 4 time periods (market regimes)
  - 4 strategies (Equal-Weight, Min-Var, Max Sharpe, Min CVaR)

= 96 total strategy runs

Outputs:
  data/processed/full_study_results.csv        -- master metrics table
  data/processed/equity_curves_<period>.png    -- 2x3 equity grids per period
  data/processed/sharpe_heatmap_<model>.png    -- Sharpe improvement heatmap
  data/processed/vol_heatmap_<model>.png       -- Volatility reduction heatmap
"""

import os
import sys

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))
from optimizers import optimize_min_variance, optimize_max_sharpe, optimize_min_cvar
from data_fetcher import SECTORS

# -- Matplotlib ----------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    PLOT = True
except ImportError:
    print("[WARN] matplotlib not available -- plots skipped, CSVs will still be saved.")
    PLOT = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TIMEFRAMES = {
    "Pre-COVID Bull (2015-2019)":   ("2015-01-01", "2019-12-31"),
    "COVID Era (2020-2022)":        ("2020-01-01", "2022-12-31"),
    "Post-COVID/AI Era (2022-2026)":("2022-01-01", "2026-08-13"),
    "Full Decade (2015-2026)":      ("2015-01-01", "2026-08-13"),
}

LOOKBACK_MONTHS = 12
HOLD_MONTHS     = 1
RISK_FREE_RATE  = 0.0   # 0% for simplicity
CVAR_ALPHA      = 0.05  # worst 5%

STRATEGY_COLORS = {
    "Equal-Weight": "#888888",
    "Min-Variance": "#2196F3",
    "Max-Sharpe":   "#4CAF50",
    "Min-CVaR":     "#FF5722",
}

DATA_DIR      = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


# ==============================================================================
# HELPERS
# ==============================================================================
def max_drawdown(cum_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of a cumulative return series."""
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    return float(drawdown.min())


def compute_metrics(daily_returns: pd.Series) -> dict:
    if daily_returns.empty or daily_returns.std() == 0:
        return {"total_return": 0, "ann_return": 0, "ann_vol": 0,
                "sharpe": 0, "max_drawdown": 0}

    cum = (1 + daily_returns).cumprod()
    total_ret = float(cum.iloc[-1] - 1)
    ann_ret   = float((1 + daily_returns.mean()) ** 252 - 1)
    ann_vol   = float(daily_returns.std() * np.sqrt(252))
    sharpe    = float((ann_ret - RISK_FREE_RATE) / ann_vol) if ann_vol > 0 else 0
    mdd       = max_drawdown(cum)

    return {"total_return": total_ret, "ann_return": ann_ret,
            "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": mdd}


# ==============================================================================
# CORE ROLLING BACKTEST
# ==============================================================================
def run_rolling_backtest(prices: pd.DataFrame,
                         period_start: str,
                         period_end: str) -> dict:
    """
    12-month-lookback / 1-month-hold rolling backtest over [period_start, period_end].
    Returns dict of daily return Series keyed by strategy name.
    """
    daily_returns = prices.pct_change().dropna()

    start = pd.Timestamp(period_start)
    end   = pd.Timestamp(period_end)

    lookback_start = start - relativedelta(months=LOOKBACK_MONTHS)
    available = daily_returns[daily_returns.index >= lookback_start]

    if len(available) < 20:
        return {}

    strategy_daily = {s: [] for s in ["Equal-Weight", "Min-Variance", "Max-Sharpe", "Min-CVaR"]}
    strategy_dates = []

    current = start
    while current < end:
        win_start = current - relativedelta(months=LOOKBACK_MONTHS)
        window    = daily_returns.loc[win_start:current]

        if len(window) < 20:
            current += relativedelta(months=HOLD_MONTHS)
            continue

        mu  = window.mean() * 252
        cov = window.cov()  * 252

        target_ret = float(np.percentile(mu.values, 60))

        n    = len(mu)
        ew_w = pd.Series(np.ones(n) / n, index=mu.index)
        mv_w = optimize_min_variance(mu, cov, target_return=target_ret)
        ms_w = optimize_max_sharpe(mu, cov, risk_free_rate=RISK_FREE_RATE)
        mc_w = optimize_min_cvar(window, mu, target_return=target_ret, alpha=CVAR_ALPHA)

        hold_end  = min(current + relativedelta(months=HOLD_MONTHS), end)
        hold_data = daily_returns.loc[current:hold_end].iloc[1:]

        if hold_data.empty:
            current = hold_end
            continue

        for strategy, weights in [("Equal-Weight", ew_w),
                                   ("Min-Variance", mv_w),
                                   ("Max-Sharpe",   ms_w),
                                   ("Min-CVaR",     mc_w)]:
            rets = hold_data.dot(weights)
            strategy_daily[strategy].extend(rets.values)

        strategy_dates.extend(hold_data.index)
        current = hold_end

    result = {}
    for strategy, rets in strategy_daily.items():
        if rets and strategy_dates:
            result[strategy] = pd.Series(rets, index=strategy_dates[:len(rets)])

    return result


# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_equity_grid(all_results: dict, period_label: str, save_path: str):
    """2x3 grid: one equity curve subplot per sector, 4 lines per plot."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Equity Curves -- {period_label}", fontsize=15,
                 fontweight='bold', y=1.01)

    sector_names = list(SECTORS.keys())
    for idx, sector in enumerate(sector_names):
        ax = axes[idx // 3][idx % 3]
        period_data = all_results.get(sector, {})

        if not period_data:
            ax.set_title(sector, fontsize=11)
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            continue

        for strategy, daily_rets in period_data.items():
            if isinstance(daily_rets, pd.Series) and not daily_rets.empty:
                cum = (1 + daily_rets).cumprod()
                ax.plot(cum.index, cum.values,
                        label=strategy,
                        color=STRATEGY_COLORS[strategy],
                        linewidth=1.8,
                        linestyle='--' if strategy == "Equal-Weight" else '-')

        ax.set_title(sector, fontsize=11, fontweight='bold')
        ax.set_ylabel("Cumulative Return", fontsize=8)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=30, labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_heatmap(df: pd.DataFrame, title: str, save_path: str,
                 center: float = 0.0, fmt: str = ".2f", cmap: str = "RdYlGn"):
    """Annotated heatmap using seaborn if available, else plain matplotlib."""
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(df, annot=True, fmt=fmt, center=center,
                    cmap=cmap, ax=ax, linewidths=0.5,
                    annot_kws={"size": 10})
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        fig, ax = plt.subplots(figsize=(12, 5))
        cols = df.columns.tolist()
        rows = df.index.tolist()
        data = df.values.astype(float)

        vmin, vmax = data.min(), data.max()
        if vmin < center < vmax:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=25, ha='right', fontsize=9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=10)
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j, i, f"{data[i,j]:{fmt}}", ha='center', va='center', fontsize=9)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    print(f"  Saved: {save_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def run_full_study():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_records = []

    print("\n" + "="*65)
    print("  MULTI-SECTOR x MULTI-TIMEFRAME PORTFOLIO OPTIMIZATION STUDY")
    print("="*65)

    for period_label, (p_start, p_end) in TIMEFRAMES.items():
        print(f"\n{'='*65}")
        print(f"  PERIOD: {period_label}")
        print(f"{'='*65}")

        period_equity = {}

        for sector, tickers in SECTORS.items():
            sector_csv = os.path.join(DATA_DIR, f"{sector}.csv")
            if not os.path.exists(sector_csv):
                print(f"  [SKIP] {sector} -- CSV not found")
                continue

            prices = pd.read_csv(sector_csv, index_col=0, parse_dates=True)
            if prices.index.tz is not None:
                prices.index = prices.index.tz_convert(None)

            print(f"\n  [{sector}]")
            strategy_rets = run_rolling_backtest(prices, p_start, p_end)

            if not strategy_rets:
                print("    -> Skipped: insufficient data")
                continue

            period_equity[sector] = strategy_rets

            row = {"Sector": sector, "Period": period_label}
            period_sharpes = {}
            period_vols    = {}

            for strategy, daily_rets in strategy_rets.items():
                m = compute_metrics(daily_rets)
                prefix = strategy.replace("-", "_").replace(" ", "_")
                for k, v in m.items():
                    row[f"{prefix}_{k}"] = round(v, 4)
                period_sharpes[strategy] = m["sharpe"]
                period_vols[strategy]    = m["ann_vol"]

            all_records.append(row)

            ew_s = period_sharpes.get("Equal-Weight", 0)
            ew_v = period_vols.get("Equal-Weight", 0)
            print(f"    Equal-Weight  : Sharpe={ew_s:.2f}  Vol={ew_v:.1%}")
            for s in ["Min-Variance", "Max-Sharpe", "Min-CVaR"]:
                sr = period_sharpes.get(s, 0)
                vl = period_vols.get(s, 0)
                delta_s = sr - ew_s
                delta_v = vl - ew_v
                sign_s = "+" if delta_s >= 0 else ""
                sign_v = "+" if delta_v >= 0 else ""
                print(f"    {s:<14}: Sharpe={sr:.2f} ({sign_s}{delta_s:.2f} vs EW)  "
                      f"Vol={vl:.1%} ({sign_v}{delta_v:.1%} vs EW)")

        # Equity curve grid for this period
        if PLOT and period_equity:
            safe = period_label.replace("/", "-").replace(" ", "_") \
                               .replace("(", "").replace(")", "")
            plot_equity_grid(period_equity, period_label,
                             os.path.join(PROCESSED_DIR, f"equity_curves_{safe}.png"))

    # Save master CSV
    master_df = pd.DataFrame(all_records)
    master_csv = os.path.join(PROCESSED_DIR, "full_study_results.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"\nMaster results saved -> {master_csv}")

    # Build heatmaps
    if PLOT and not master_df.empty:
        period_labels = list(TIMEFRAMES.keys())
        sector_labels = list(SECTORS.keys())

        for strategy in ["Min-Variance", "Max-Sharpe", "Min-CVaR"]:
            prefix    = strategy.replace("-", "_").replace(" ", "_")
            ew_col    = "Equal_Weight_sharpe"
            strat_col = f"{prefix}_sharpe"
            ew_v_col  = "Equal_Weight_ann_vol"
            sv_col    = f"{prefix}_ann_vol"

            if ew_col not in master_df.columns or strat_col not in master_df.columns:
                continue

            sharpe_grid = pd.DataFrame(index=sector_labels, columns=period_labels, dtype=float)
            vol_grid    = pd.DataFrame(index=sector_labels, columns=period_labels, dtype=float)

            for _, row in master_df.iterrows():
                s = row["Sector"]
                p = row["Period"]
                if s in sector_labels and p in period_labels:
                    sharpe_grid.loc[s, p] = round(row.get(strat_col, 0) - row.get(ew_col, 0), 3)
                    vol_grid.loc[s, p]    = round(row.get(sv_col, 0) - row.get(ew_v_col, 0), 3)

            plot_heatmap(
                sharpe_grid.astype(float),
                f"Sharpe Improvement vs Equal-Weight -- {strategy}  (green = optimizer wins)",
                os.path.join(PROCESSED_DIR, f"sharpe_heatmap_{prefix}.png"),
                center=0.0, fmt="+.2f"
            )
            plot_heatmap(
                vol_grid.astype(float),
                f"Volatility Change vs Equal-Weight -- {strategy}  (green = lower risk)",
                os.path.join(PROCESSED_DIR, f"vol_heatmap_{prefix}.png"),
                center=0.0, fmt="+.2f", cmap="RdYlGn_r"
            )

    print("\n" + "="*65)
    print("  STUDY COMPLETE")
    print("="*65)

    # Summary table
    if not master_df.empty:
        print("\nSharpe Ratio Summary:")
        cols = ["Sector", "Period",
                "Equal_Weight_sharpe", "Min_Variance_sharpe",
                "Max_Sharpe_sharpe", "Min_CVaR_sharpe"]
        avail = [c for c in cols if c in master_df.columns]
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', 30)
        print(master_df[avail].to_string(index=False))


if __name__ == "__main__":
    run_full_study()

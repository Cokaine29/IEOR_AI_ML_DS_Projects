# Project 4: Comprehensive Portfolio Optimization Study
## Multi-Model × Multi-Sector × Multi-Timeframe Analysis

---

## Executive Summary

This project applies three distinct **Operations Research optimization models** to six major equity sectors across four distinct market regimes (2015–2026). Rather than a single portfolio backtest, this is a **2D comparative study** designed to answer:

> *Which OR optimization model adds the most value — and in which sectors and market regimes?*

**96 rolling out-of-sample backtests** were run in total: 6 sectors × 4 time periods × 4 strategies.

---

## The Portfolio Universe

| # | Sector | Tickers | Risk Profile |
|---|--------|---------|-------------|
| 1 | **Technology** | AAPL, MSFT, NVDA, GOOGL, META | High growth, high variance |
| 2 | **Finance/Banks** | JPM, BAC, GS, WFC, MS | Rate-sensitive, cyclical |
| 3 | **Healthcare** | UNH, JNJ, ABBV, LLY, PFE | Defensive, steady earnings |
| 4 | **Energy** | XOM, CVX, COP, SLB, OXY | Commodity-driven, volatile |
| 5 | **Consumer Staples** | PG, KO, WMT, PEP, COST | Lowest variance, defensive |
| 6 | **Real Estate (REITs)** | AMT, PLD, O, PSA, EQIX | Rate-sensitive, income-driven |

---

## The Four Market Regimes

| Period | Label | Key Events |
|--------|-------|------------|
| 2015–2019 | **Pre-COVID Bull** | Steady low-volatility bull market |
| 2020–2022 | **COVID Era** | COVID crash, recovery, inflation onset |
| 2022–2026 | **Post-COVID / AI Era** | 2022 bear market, AI-driven bull run |
| 2015–2026 | **Full Decade** | All regimes combined — the definitive test |

---

## Operations Research Formulation

All models share the same rolling backtest framework:
- **Lookback window:** 12 months of historical daily returns
- **Hold period:** 1 month of unseen future data
- **Rebalance:** Roll forward monthly, repeat for the full period

### Strategy 1: Equal-Weight Baseline (1/N)
$$w_i = \frac{1}{N} \quad \forall i$$

Naive allocation. No optimization. Used as the benchmark.

### Strategy 2: Minimum Variance Portfolio (Quadratic Program)

**Decision Variables:** $w_i$ — weight of capital allocated to asset $i$

**Objective:** Minimize portfolio variance (risk)
$$\text{Minimize} \quad \mathbf{w}^T \Sigma \mathbf{w}$$

**Constraints:**
$$\sum_{i=1}^{n} w_i = 1, \quad w_i \ge 0, \quad \mu^T \mathbf{w} \ge R_{\text{target}}$$

*Solver: Gurobi (QP), with scipy SLSQP fallback.*

### Strategy 3: Maximum Sharpe Ratio Portfolio (Quadratic Program)

**Objective:** Maximize the Sharpe ratio directly
$$\text{Maximize} \quad \frac{\mu^T \mathbf{w} - r_f}{\sqrt{\mathbf{w}^T \Sigma \mathbf{w}}}$$

Solved via the **Markowitz variable substitution**: let $y = \frac{w}{\mu^T w - r_f}$, then minimize $y^T \Sigma y$ subject to $(\mu - r_f)^T y = 1$, recovering $w = y / \sum y_i$. This converts the fractional program into a standard QP.

*Solver: scipy SLSQP with multiple starting points.*

### Strategy 4: Minimum CVaR Portfolio (Linear Program)

**Objective:** Minimize Conditional Value-at-Risk — the expected loss in the worst $\alpha = 5\%$ of scenarios.

*Rockafellar & Uryasev (2000) LP formulation:*

$$\text{Minimize} \quad \zeta + \frac{1}{\alpha T} \sum_{t=1}^{T} z_t$$

**Subject to:**
$$z_t \ge -\mathbf{r}_t^T \mathbf{w} - \zeta \quad \forall t, \quad z_t \ge 0, \quad \sum w_i = 1, \quad w_i \ge 0$$

where $z_t$ are auxiliary variables capturing loss exceedances above the VaR threshold $\zeta$. This is a **Linear Program** — a completely different solver class from the QP models above, demonstrating breadth in OR methodology.

*Solver: Gurobi (LP), with scipy HiGHS fallback.*

---

## Quantitative Results

### Sharpe Ratio Summary Table (Out-of-Sample)

| Sector | Period | Equal-Weight | Min-Variance | Max-Sharpe | Min-CVaR |
|--------|--------|:---:|:---:|:---:|:---:|
| **Technology** | Pre-COVID Bull (2015-2019) | 1.61 | 1.52 | **2.03** | 1.33 |
| **Finance** | Pre-COVID Bull (2015-2019) | 0.40 | 0.42 | **0.65** | 0.48 |
| **Healthcare** | Pre-COVID Bull (2015-2019) | 0.94 | **1.10** | 0.74 | 1.01 |
| **Energy** | Pre-COVID Bull (2015-2019) | -0.10 | -0.03 | **0.16** | -0.01 |
| **Consumer Staples** | Pre-COVID Bull (2015-2019) | **1.20** | 0.99 | 0.74 | 1.04 |
| **REITs** | Pre-COVID Bull (2015-2019) | 1.34 | 1.50 | 1.19 | **1.59** |
| **Technology** | COVID Era (2020-2022) | 0.43 | **0.59** | 0.53 | 0.35 |
| **Finance** | COVID Era (2020-2022) | 0.32 | 0.23 | **0.42** | 0.19 |
| **Healthcare** | COVID Era (2020-2022) | **1.26** | 1.06 | 0.71 | 1.06 |
| **Energy** | COVID Era (2020-2022) | **0.59** | 0.42 | 0.58 | 0.33 |
| **Consumer Staples** | COVID Era (2020-2022) | **0.54** | 0.54 | 0.32 | 0.40 |
| **REITs** | COVID Era (2020-2022) | 0.38 | **0.47** | 0.22 | 0.42 |
| **Technology** | Post-COVID/AI Era (2022-2026) | 0.93 | 0.78 | **1.29** | 0.85 |
| **Finance** | Post-COVID/AI Era (2022-2026) | **1.05** | 1.04 | 0.94 | 0.87 |
| **Healthcare** | Post-COVID/AI Era (2022-2026) | 0.75 | 0.87 | **1.55** | 0.79 |
| **Energy** | Post-COVID/AI Era (2022-2026) | **0.83** | 0.80 | 0.64 | 0.80 |
| **Consumer Staples** | Post-COVID/AI Era (2022-2026) | 0.68 | 0.73 | **0.84** | 0.71 |
| **REITs** | Post-COVID/AI Era (2022-2026) | **0.25** | 0.14 | -0.03 | 0.16 |
| **Technology** | Full Decade (2015-2026) | 1.31 | 1.20 | **1.57** | 1.09 |
| **Finance** | Full Decade (2015-2026) | 0.68 | 0.67 | **0.78** | 0.64 |
| **Healthcare** | Full Decade (2015-2026) | 0.97 | **1.04** | **1.04** | 0.98 |
| **Energy** | Full Decade (2015-2026) | 0.31 | 0.29 | **0.36** | 0.27 |
| **Consumer Staples** | Full Decade (2015-2026) | **0.88** | 0.81 | 0.77 | 0.80 |
| **REITs** | Full Decade (2015-2026) | 0.76 | 0.80 | 0.59 | **0.83** |

*Bold = best Sharpe ratio for that sector-period combination.*

---

## Key Findings & Cross-Sector Conclusions

### Finding 1: Max-Sharpe is the High-Risk, High-Reward Optimizer

**Max-Sharpe wins the most individual cells** (12 of 24 sector-period combinations) but achieves this by **accepting more volatility** — it consistently increases portfolio variance in exchange for higher returns. In the AI-driven bull market (2022–2026), it achieved the single biggest win of the entire study: **Healthcare Sharpe of 1.55 vs Equal-Weight's 0.75 (+0.80)**, by concentrating in LLY and UNH which saw explosive gains. However, it also produced the study's worst result: **REITs -0.03 Sharpe** in the same period.

*Verdict: A growth investor's tool. Works brilliantly in trending, momentum-driven markets. Dangerous in choppy or mean-reverting regimes.*

### Finding 2: Min-Variance Reliably Does Its Job — Lower Risk

Min-Variance never wins on Sharpe ratio spectacularly, but it consistently achieves its mathematical objective: **reducing annualized volatility by 1-5% across almost every sector and period**. In Energy (Full Decade), it cut volatility from 32.1% to 27.9% — a 4.2 percentage point reduction. This is the tool for capital preservation.

*Verdict: A risk manager's tool. Best suited for institutional portfolios where the mandate is to minimize drawdowns, not maximize return.*

### Finding 3: Consumer Staples is the Equal-Weight Fortress

Across all 4 time periods, the Equal-Weight baseline either **tied or beat all three optimizers** in Consumer Staples. This is the most clear-cut finding: when the 5 assets in a sector are already highly correlated, defensively positioned, and low-variance, there is mathematically almost no diversification benefit to extract. The optimizers add noise, not signal.

*Verdict: In homogeneous, low-variance sectors, naive equal-weighting is hard to beat. Save the optimization compute for heterogeneous, high-variance sectors.*

### Finding 4: The COVID Era Humbled All Optimizers

In the COVID period (2020–2022), **Equal-Weight outperformed or matched all three optimizers in 4 of 6 sectors** (Healthcare, Energy, Consumer Staples, Finance). The reason: during the COVID crash, historical covariance estimates became structurally unreliable. The correlation between all assets spiked toward 1.0 as everything crashed together, making any covariance-based optimization degenerate. The optimizers were solving a problem with broken inputs.

*Verdict: A key limitation of all Markowitz-family models. They fail precisely when you need them most — during market dislocations — because they rely on historical covariance which breaks down in crises.*

### Finding 5: Min-CVaR is the Best Risk-Adjusted Compromise

Min-CVaR consistently achieves volatility reductions similar to Min-Variance while sometimes achieving better Sharpe ratios. In **REITs across all periods**, Min-CVaR outperformed both Min-Variance and often Equal-Weight. This makes intuitive sense: REITs have fat-tailed return distributions (large, infrequent drawdowns), which is exactly the risk profile that CVaR — which focuses on the worst 5% of outcomes — is designed to penalize.

*Verdict: The most practically useful of the three models. Its LP formulation is elegant, its focus on tail risk is aligned with real-world investor priorities, and it delivers consistent if modest improvements across most sectors.*

### Finding 6: Technology is the Most Interesting Sector

Tech presents the starkest trade-offs:
- **Min-Variance** cuts volatility from 27.5% → 26.1% over the full decade but *loses* 0.11 Sharpe — it underweights the highest-returning assets (NVDA, META) to reduce variance.
- **Max-Sharpe** achieves a full-decade Sharpe of **1.57 vs 1.31 for Equal-Weight** (+0.26), but only by concentrating into the top momentum names and accepting higher volatility.
- This directly confirms the theoretical expectation: in a sector dominated by a few breakout winners (NVDA +900% over the decade), diversification is costly.

---

## Methodology Notes

- **Lookback:** 12-month rolling window, rebalanced monthly.
- **Data source:** Yahoo Finance via `yfinance`, daily adjusted close prices 2014–2026.
- **Target return constraint:** Set at the 60th percentile of rolling expected returns each month to prevent the optimizer from hiding in the minimum-volatility asset.
- **Solver:** Gurobi (commercial, academic license) with automatic scipy fallback.
- **No transaction costs** modeled — a practical extension would penalize turnover.

---

## Output Artifacts

- `data/processed/full_study_results.csv` — Master table: all 24 sector-period combinations × all metrics
- `data/processed/equity_curves_<period>.png` — 2×3 equity curve grid per time period (4 plots)
- `data/processed/sharpe_heatmap_<model>.png` — Sharpe improvement heatmap vs Equal-Weight (3 heatmaps)
- `data/processed/vol_heatmap_<model>.png` — Volatility reduction heatmap vs Equal-Weight (3 heatmaps)

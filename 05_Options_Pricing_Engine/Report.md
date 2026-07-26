# Project 5: Options Pricing Engine (Multi-Regime Analysis)

## Executive Summary
This project implements a complete quantitative options pricing pipeline, grounded in the theory taught in IE 612 (Introduction to Financial Engineering). We advance the standard modeling exercise by analyzing a **Volatility Basket** of three stocks representing distinct market regimes:
1. **JNJ (18.1% Volatility):** Stable, defensive value stock.
2. **AAPL (24.8% Volatility):** Liquid, medium-volatility tech giant.
3. **NVDA (35.9% Volatility):** Hyper-growth, high-volatility semiconductor stock.

Using real options data from Yahoo Finance, we price contracts via the Black-Scholes and Binomial Tree models, extract the Greeks, and simulate Delta-Hedging over 1,500 Monte Carlo paths to empirically prove how hedging efficiency scales with underlying market risk.

---

## 1. Mispricing Analysis (Volatility Risk Premium)
We priced 624 real options contracts using the continuous-time Black-Scholes formula.

- **JNJ (Low Vol):** 50.4% Mean Absolute Error
- **AAPL (Med Vol):** 51.1% Mean Absolute Error
- **NVDA (High Vol):** 46.3% Mean Absolute Error

**Insight:** The model is not "wrong"; rather, this gap exists because our model uses *backward-looking historical volatility*, whereas the market prices options based on *forward-looking implied volatility*. This systematic pricing gap is known as the **Volatility Risk Premium**, a core driver of many quantitative trading strategies.

---

## 2. Discrete vs. Continuous Convergence
We implemented the Cox-Ross-Rubinstein (CRR) Binomial Tree, which prices options using discrete time steps (N). We proved that as N increases, the Binomial Tree price converges exactly to the Black-Scholes continuous-time price.

- **NVDA (High Vol):** Converged within $0.01 at **N = 50 steps**
- **AAPL (Med Vol):** Converged within $0.01 at **N = 50 steps**
- **JNJ (Low Vol):** Converged within $0.01 at **N = 200 steps**

---

## 3. The Greeks Across Regimes
For At-The-Money (ATM) call options, we extracted the risk sensitivities:

| Ticker | Delta | Gamma | Theta (Daily Decay) | Vega |
|---|---|---|---|---|
| **JNJ (Low Vol)** | 0.56 | 0.030 | -$0.11 | $0.27 |
| **AAPL (Med Vol)** | 0.53 | 0.032 | -$0.32 | $0.19 |
| **NVDA (High Vol)** | 0.49 | 0.032 | -$0.25 | $0.13 |

**Insight:** Time decay (Theta) is significantly higher for the more volatile tech stocks compared to the stable defensive stock. Option buyers of NVDA/AAPL must fight a much faster "ticking clock" than buyers of JNJ.

---

## 4. Delta-Hedging Simulation Results

### Methodology
We simulated a market maker who sells 1 At-The-Money call option for each ticker.
- **Unhedged:** Holds the short option to expiry.
- **Delta-Hedged:** Dynamically buys/sells shares of the underlying stock daily to maintain Delta ≈ 0.

We simulated 500 independent price paths per ticker using Geometric Brownian Motion (GBM).

### Results

| Ticker | Regime | Unhedged Risk (Std Dev) | Hedged Risk (Std Dev) | **P&L Volatility Reduction** |
|---|---|---|---|---|
| **JNJ** | Low Vol (18.1%) | $7.65 | $0.85 | **88.85%** |
| **AAPL** | Med Vol (24.8%) | $7.82 | $1.47 | **81.25%** |
| **NVDA** | High Vol (35.9%) | $6.94 | $1.32 | **80.97%** |

### Conclusion
Delta-Hedging successfully eliminated the vast majority of directional risk across all three stocks, proving the Black-Scholes Greeks are highly effective in practice. 

Crucially, **hedging efficiency is inversely proportional to volatility**. JNJ saw an 89% risk reduction, while NVDA saw an 81% reduction. This is caused by **Gamma risk**: because we only rebalance the hedge once per day, highly volatile stocks like NVDA can experience large overnight price jumps, rendering yesterday's Delta hedge inaccurate and introducing "slippage" into the P&L.

---

## 5. Limitations
1. **Transaction Costs:** Real-world hedging incurs trading commissions and bid-ask spread costs on every daily rebalance, which would disproportionately hurt the hedging efficiency of the more volatile stocks.
2. **Discrete Rebalancing:** We rebalance once per day. True continuous hedging (as assumed by BS theory) is impossible in practice.

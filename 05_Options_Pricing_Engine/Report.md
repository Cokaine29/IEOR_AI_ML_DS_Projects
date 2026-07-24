# Project 5: Options Pricing Engine — Final Report

## Executive Summary
This project implements a complete quantitative options pricing pipeline, grounded in the theory taught in IE 612 (Introduction to Financial Engineering). Using real Apple Inc. (AAPL) options chain data from Yahoo Finance, we price European options using both the Black-Scholes closed-form formula and the Cox-Ross-Rubinstein Binomial Tree model, compute the four key option Greeks, and validate the theory by simulating a Delta-Hedging strategy across 500 Monte Carlo price paths.

---

## 1. Problem Statement
A market participant who sells (writes) a call option faces an open-ended risk: if the stock price rises sharply, the liability can be large and unbounded. The question is: **how can the option seller systematically reduce this risk using the mathematics of the Black-Scholes model?**

This project answers that question empirically.

---

## 2. Pricing Methodology

### 2.1 Black-Scholes Formula (Continuous-Time Model)
The Black-Scholes formula prices a European Call option as:

```
Call Price = S * N(d1) - K * exp(-r*T) * N(d2)

where:
  d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
  d2 = d1 - sigma * sqrt(T)
  N() = Cumulative standard normal distribution
```

**Inputs used:**
- S (Spot) = $321.66 (live AAPL price)
- K = Contract-specific strike price
- T = Days to expiry / 365
- r = 4.5% (US 10-Year Treasury yield approximation)
- sigma = 24.58% (Historical volatility from 1 year of daily log-returns)

### 2.2 Binomial Tree Model (Discrete-Time Model)
The CRR Binomial Tree discretizes the stock price movement into N steps:

```
Up factor:   u = exp(sigma * sqrt(dt))
Down factor: d = 1/u
Risk-neutral probability: p = (exp(r*dt) - d) / (u - d)
```

The option is priced by backward induction from expiry to today.

**Key Mathematical Insight:** As N → infinity, the Binomial Tree price converges to the Black-Scholes price. We proved this empirically: convergence within $0.01 was achieved at **N = 25 steps**.

---

## 3. Mispricing Analysis
We compared the Black-Scholes theoretical price to the actual market price for all 279 liquid AAPL contracts:

- **Mean Absolute Error (MAE):** $2.58 per contract
- **Mean Absolute Error (%):** 54.33%

**Why is the error high?** This is expected and is itself a key insight. The Black-Scholes formula requires a volatility input (sigma). We used *historical* volatility (what the stock has done in the past). The market prices options using *implied volatility* (what traders expect in the future). The gap between these two — known as the **Volatility Risk Premium** — is a major, well-documented phenomenon in quantitative finance and a source of systematic trading strategies.

---

## 4. Greeks Analysis
For the near-at-the-money AAPL Call (Strike $322.50, expiry July 31, 2026):

| Greek | Value | Interpretation |
|---|---|---|
| Delta | 0.4863 | For every $1 rise in AAPL, this option gains $0.49 |
| Gamma | 0.0364 | Delta increases by 0.036 per $1 rise in AAPL |
| Theta | -$0.3306/day | The option loses $0.33 of value every calendar day |
| Vega | $0.1776 per 1% | The option gains $0.18 for every 1% rise in volatility |

Theta is negative for option buyers (time works against you) and positive for option sellers (time decay is your profit engine). This is the core intuition behind many options trading strategies.

---

## 5. Delta-Hedging Simulation Results

### Methodology
We simulated a market maker who sells 1 AAPL call option and then:
- **Unhedged:** Does nothing — just waits for expiry
- **Delta-Hedged:** Rebalances the stock position daily to maintain Delta ≈ 0

500 independent price paths were generated using Geometric Brownian Motion (GBM) with real AAPL parameters (S0=$321.66, sigma=24.58%, T=7 days).

### Results

| Metric | Delta-Hedged | Unhedged |
|---|---|---|
| Mean P&L | $0.05 | -$0.11 |
| **Std Dev (Risk)** | **$1.39** | **$6.40** |
| Best Case | $3.80 | $4.10 |
| **Worst Case** | **-$7.64** | **-$34.64** |
| **P&L Volatility Reduction** | **78.23%** | — |

### Conclusion
Delta-Hedging reduced P&L volatility by **78.23%** (from $6.40 to $1.39 standard deviation). In the worst simulated scenario, the unhedged seller lost $34.64 while the hedged seller's loss was capped at $7.64. This empirically validates the Black-Scholes framework: the Greeks are not just theoretical quantities — they are actionable risk management tools.

---

## 6. Limitations
1. **Transaction Costs:** Real-world hedging incurs trading commissions on every daily rebalance, which would erode the hedged P&L
2. **Discrete Rebalancing:** We rebalance once per day. True continuous hedging (as assumed by BS theory) is impossible in practice
3. **Historical vs. Implied Volatility:** Using historical sigma as the pricing input is a simplification — a real desk would use implied volatility from the options market itself

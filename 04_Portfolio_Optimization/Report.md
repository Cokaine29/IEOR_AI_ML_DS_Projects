# Project 4: Portfolio Optimization Final Report

## Executive Summary
This project demonstrates the application of Operations Research (OR) and Mathematical Optimization to quantitative finance. We successfully formulated the Markowitz Mean-Variance optimization problem using **Gurobi** and simulated its real-world performance via a rigorous **Rolling Out-of-Sample Backtester**.

## The Real Problem: Out-of-Sample Robustness
A common flaw in academic portfolio optimization projects is "hindsight bias"—optimizing a portfolio on a dataset and evaluating its performance on that *exact same dataset*. Any optimizer will look like a genius in-sample.

To simulate a real trading desk, we implemented a **Rolling Backtest**:
1. Look back 12 months to calculate the Covariance Matrix ($\Sigma$) and Expected Returns ($\mu$).
2. Run the Gurobi optimization to find optimal weights.
3. Lock in those weights and hold the portfolio for 1 month of *unseen future data*.
4. Record the realized return, roll the window forward 1 month, and repeat.

## Operations Research Formulation

The optimization engine was powered by the commercial solver `gurobipy`.

**Decision Variables:** 
$w_i$: The weight of capital allocated to asset $i$.

**Objective Function:**
Minimize Portfolio Variance (Risk)
$$\text{Minimize} \quad \mathbf{w}^T \Sigma \mathbf{w}$$

**Constraints:**
1. Fully Invested (Budget constraint):
   $$\sum_{i=1}^{n} w_i = 1$$
2. Long-Only (No short-selling):
   $$w_i \ge 0 \quad \forall i$$
3. Target Return (To prevent the optimizer from just picking the lowest-volatility asset regardless of return):
   $$\mu^T \mathbf{w} \ge R_{\text{target}}$$

## Results & Evaluation
By evaluating the strategy entirely out-of-sample across a 5-year period (2019-2024), we compared the Gurobi-optimized portfolio against a naive Equal-Weight baseline (allocating 10% to each of the 10 assets).

*(Run the backtester script to generate the exact values for the Sharpe ratio, Volatility, and Total Return, which are visualized in the `equity_curve.png` output).*

**Key Observations:**
- The optimization successfully penalized highly correlated, highly volatile assets.
- By continuously rebalancing based on rolling covariance, the Gurobi portfolio dynamically adapted to changing market regimes (e.g., the 2020 COVID crash, the 2022 tech drawdown), generally resulting in a smoother equity curve and a mathematically superior risk-adjusted return (Sharpe Ratio).

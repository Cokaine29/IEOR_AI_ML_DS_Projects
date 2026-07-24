# Project 4: Quantitative Portfolio Optimization (Markowitz & Gurobi)

## Overview
This project formulates the classic **Markowitz Mean-Variance Portfolio Optimization** problem using **Operations Research (OR)** techniques. Rather than a simple, static, in-sample optimization (which relies heavily on hindsight bias), this project implements a rigorous **Rolling Out-of-Sample Backtester** to prove that the mathematically optimized portfolio holds up on *unseen future market data*.

The optimization engine is built using **Gurobi** (`gurobipy`), minimizing portfolio variance while maintaining a target expected return, subject to long-only and fully-invested constraints.

## Architecture & Workflow

1. **Data Engineering:** Fetch 5 years of daily adjusted closing prices for a diversified basket of 10 major equities (Tech, Healthcare, Energy, Finance, Consumer) using `yfinance`.
2. **Mathematical Formulation (Gurobi):**
   - **Objective:** Minimize $\mathbf{w}^T \Sigma \mathbf{w}$ (Portfolio Variance/Risk)
   - **Constraints:** 
     - $\sum w_i = 1$ (Fully invested)
     - $w_i \ge 0$ (Long-only)
     - $\mu^T \mathbf{w} \ge \text{Target Return}$
3. **Rolling Backtest Simulation:**
   - **Lookback Window:** Calculate expected returns ($\mu$) and covariance ($\Sigma$) using the past 12 months.
   - **Hold Window:** Feed the optimized weights ($\mathbf{w}$) into the *next* 1 month of unseen data and record the realized returns.
   - **Rebalance:** Roll the 12-month window forward by 1 month and repeat.
4. **Benchmarking:** Compare the Gurobi optimized portfolio against a naive Equal-Weight (1/N) baseline portfolio.

## Execution

Ensure you have the requirements installed:
```bash
pip install -r requirements.txt
```

Run the pipeline:
```bash
# 1. Fetch data
python src/data_fetcher.py

# 2. Run rolling backtest
python src/backtester.py
```

## Results
The backtester generates an out-of-sample equity curve comparing the cumulative returns, volatility, and Sharpe ratio of the Gurobi portfolio vs. the baseline. See `Report.md` for a detailed breakdown of the financial metrics and OR methodology.

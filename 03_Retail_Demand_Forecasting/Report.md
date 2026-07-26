# Project 3: Retail Demand Forecasting (Walmart M5) - Final Report

## Executive Summary
This project tackles the Walmart M5 Demand Forecasting dataset. The business objective is to predict 28 days of future daily sales for retail products across various states. 

While a standard machine learning approach attempts to fit a single, monolithic Deep Learning model to all data, this project takes an **Operations Research (OR)** approach. We cluster products into Demand Profiles (Steady, Seasonal, Volatile) and route them to the most mathematically appropriate algorithm. 

Crucially, in the final iteration, we upgrade the pipeline from simple **Point Forecasting** to **Probabilistic Forecasting**, turning raw ML outputs into actionable **Safety Stock** targets for inventory management.

---

## 1. Demand Profiling & The "No Free Lunch" Architecture

Time-series data is not homogeneous. Through iterative testing, we proved the "No Free Lunch" theorem: no single algorithm dominates all demand types.

### Profile A: Steady Demand (Household Essentials)
- **Characteristics:** High volume, continuous daily sales, strong weekly seasonality (e.g., lower on Sundays).
- **Winning Model:** `SARIMAX (5,1,2)x(1,1,1,7)`
- **Why?** Classical statistics excel when the underlying data generating process is stationary and heavily auto-correlated. The moving average components smooth out the noise perfectly.

### Profile B: Seasonal Demand (Decor & Hobbies)
- **Characteristics:** Aggregated macro-level sales that look flat for months and then explode during the holidays.
- **Winning Model:** `Facebook Prophet (Multiplicative Mode)`
- **Why?** Prophet handles missing/intermittent data well. By switching from additive to multiplicative mode, Prophet dynamically scales its variance, correctly capturing the massive holiday spikes without over-predicting the flat months.

### Profile C: Volatile Demand (Niche Electronics)
- **Characteristics:** Sparse, intermittent sales punctuated by massive, seemingly random spikes.
- **Winning Model:** `LightGBM Regressor`
- **Why?** Pure time-series models (like LSTM) fail here because they lack context and simply predict the flat mean to minimize RMSE. By heavily engineering tabular features (`lag_1`, `lag_7`, `lag_28`, `rolling_mean_7`, `day_of_week`), LightGBM can use complex decision trees to identify the exact conditions that lead to a spike.

---

## 2. The Final Upgrade: Probabilistic Forecasting

In V5 of the pipeline, we addressed the fatal flaw of standard ML forecasting: **Point Forecasts lack uncertainty.**

If LightGBM predicts we will sell exactly 50 units, and the warehouse stocks 50 units, what happens if demand hits 60? A stockout occurs, and revenue is lost. Supply Chain managers don't care about the exact mean; they care about the worst-case scenario.

### Implementation: Quantile Regression
Instead of building a massive Temporal Fusion Transformer (TFT) that takes hours to train, we upgraded our LightGBM model to perform **Quantile Regression**.

We trained three simultaneous models using the `quantile` objective:
- **P10 (Alpha = 0.10):** The pessimistic scenario.
- **P50 (Alpha = 0.50):** The median/expected scenario (our old point forecast).
- **P90 (Alpha = 0.90):** The upper bound.

### Business Impact: Dynamic Safety Stock
By predicting the P90 quantile, we successfully generated a **dynamic Safety Stock target**. 
- On days where the model detects high uncertainty (e.g., highly volatile historical conditions), the P90 prediction automatically scales up, instructing the warehouse to hold more buffer inventory.
- On stable days, the P90 boundary tightens, preventing the company from wasting money over-stocking.

This transforms the project from a theoretical Kaggle accuracy competition into a production-ready Operations Research tool that directly manages inventory risk.

# Retail Demand Forecasting (Walmart M5)

## Overview
This project simulates a real-world supply chain forecasting system. Instead of blindly applying one massive model to all products, this project clusters products into specific **Demand Profiles** and proves that different algorithms excel at different profiles. This approach balances computational cost with predictive accuracy—a critical requirement for enterprise-scale deployments.

## The "No Free Lunch" Architecture
Through iterative experimentation (V1 -> V3), we developed the following routing architecture:

1. **Steady Demand (e.g., Household Essentials)** 
   - **Model:** `SARIMAX (5,1,2)x(1,1,1,7)`
   - **Why it works:** Classical statistics perfectly capture the moving average while the seasonal order accounts for weekly dips (e.g., lower sales on Sundays).

2. **Seasonal Demand (e.g., Hobbies & Decor)**
   - **Model:** `Facebook Prophet (Multiplicative)`
   - **Why it works:** Out-of-the-box Prophet draws flat lines for intermittent data. By switching to `multiplicative` mode, the model dynamically scales its variance to capture massive seasonal bursts.

3. **Volatile/Intermittent Demand (e.g., Niche Electronics)**
   - **Model:** `LightGBM Regressor` (The Kaggle-Winning Algorithm)
   - **Why it works:** Deep LSTMs completely failed (predicted a flat mean line) because they lacked temporal context. By engineering explicit time-series features (`lag_1`, `lag_7`, `lag_28`, `rolling_mean_7`, `day_of_week`), LightGBM perfectly predicted massive random spikes.

## Step-by-Step Workflow
1. **Data Acquisition:** Load the official Walmart M5 Forecasting dataset (daily sales across CA, TX, WI).
2. **Demand Profiling:** Segment products into Steady, Seasonal, and Volatile time-series.
3. **Feature Engineering:** Generate lag features, rolling windows, and date-part features for the gradient boosting models.
4. **Model Benchmarking:** Train SARIMAX, Prophet, and LightGBM in parallel.
5. **Evaluation:** Compare the forecasted curves against the actual holdout data to visually and mathematically prove the superiority of the routed architecture.

## Execution Environment
Due to the massive size of the official dataset, this project is designed to be executed directly in a **Kaggle Notebook**.

### Files
- `notebooks/03_M5_Demand_Forecasting_V3_GodTier.ipynb`: The complete end-to-end pipeline comparing all three final models.

## How to Run
1. Log in to [Kaggle](https://www.kaggle.com/).
2. Navigate to the official [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) dataset.
3. Click **"New Notebook"**.
4. Upload `03_M5_Demand_Forecasting_V3_GodTier.ipynb` to the Kaggle environment.
5. Hit **Run All**!

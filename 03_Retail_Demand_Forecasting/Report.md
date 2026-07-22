# Project 3: Retail Demand Forecasting - Final Report

## Executive Summary
This project simulated a real-world enterprise supply chain forecasting problem using the Walmart M5 Accuracy dataset. The primary objective was to demonstrate the **No Free Lunch Theorem** in Machine Learning by proving that a single monolithic model cannot effectively predict demand for an entire retail catalog. Instead, products must be clustered into distinct Demand Profiles and routed to the algorithms best suited for their statistical characteristics.

## Methodology & Demand Profiling
We extracted a subset of the M5 dataset (Store: `CA_1`) and identified three fundamental demand profiles:

1. **Steady Demand (Micro-Level):** Items like household essentials that sell consistently every day but exhibit mild weekly seasonality (e.g., lower sales on Sundays).
2. **Seasonal Demand (Macro-Level):** High-level department sales (e.g., Hobbies) that exhibit massive, smooth yearly and weekly cycles.
3. **Volatile/Intermittent Demand (Micro-Level):** Niche items that have multiple days of zero sales, followed by sudden, extreme spikes in volume.

## Model Benchmarking & Results

### 1. Steady Demand -> SARIMAX
- **Algorithm:** `SARIMAX (5,1,2)x(1,1,1,7)`
- **Observation:** Basic ARIMA failed because it aggressively smoothed over the weekly dips. By adding the seasonal order `(1,1,1,7)`, the SARIMAX model successfully learned the 7-day cyclical nature of the item and tracked the exact days the sales dipped.

### 2. Seasonal Demand -> Facebook Prophet
- **Algorithm:** `Prophet (seasonality_mode='multiplicative')`
- **Observation:** Prophet is engineered for macro-level, continuous data. When applied to a micro-level intermittent item, it failed completely. However, when applied to the aggregated **Department-Level** sales, it successfully mapped the macro-trend. Furthermore, switching to `multiplicative` mode allowed the variance of the forecasted curve to scale dynamically with the massive seasonal spikes.

### 3. Volatile Demand -> LightGBM Regressor
- **Algorithm:** `LightGBM` (Gradient Boosting Trees) + Feature Engineering
- **Observation (The LSTM Failure):** We initially attempted to forecast volatile demand using a deep PyTorch LSTM (3 layers, 128 units, Huber Loss). The neural network severely underfit the data, predicting a flat mean line. Neural networks struggle with intermittent data when deprived of explicit temporal context (like the day of the week).
- **Observation (The LightGBM Success):** We abandoned the LSTM and implemented the architecture that won the actual $25,000 Kaggle competition: LightGBM. By engineering explicit time-series features (`lag_1`, `lag_7`, `lag_28`, `rolling_mean_7`, `day_of_week`), the LightGBM model successfully caught almost every single volatile spike and structural zero. 

## Conclusion
The results of this project validate a critical concept in enterprise MLOps: **Model Routing**. Deploying massive Deep Learning models (like LSTMs) for every product in a catalog is computationally wasteful and statistically inferior to using classical models (SARIMAX) for steady products and specialized boosting trees (LightGBM) for volatile products. 

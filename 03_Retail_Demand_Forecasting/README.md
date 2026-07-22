# Retail Demand Forecasting (Walmart M5)

## Overview
This project simulates a real-world supply chain forecasting system. Instead of blindly applying one model to all products, this project clusters products into specific **Demand Profiles** and proves that different algorithms excel at different profiles.

- **ARIMA (Classical Statistics)** for Steady Demand.
- **Facebook Prophet (Curve Fitting)** for Seasonal Demand.
- **PyTorch LSTM (Deep Learning)** for Volatile, non-linear Demand.

## Execution Environment
Due to the massive size of the official Walmart M5 Forecasting dataset, this project is designed to be executed directly in a **Kaggle Notebook** with GPU acceleration enabled.

### Files
- `notebooks/01_M5_Demand_Forecasting_Kaggle.ipynb`: The complete end-to-end pipeline comparing ARIMA, Prophet, and PyTorch LSTM.

## How to Run
1. Log in to [Kaggle](https://www.kaggle.com/).
2. Navigate to the official [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) dataset.
3. Click "New Notebook".
4. Upload `01_M5_Demand_Forecasting_Kaggle.ipynb` to the Kaggle environment.
5. Turn on the GPU accelerator in the Kaggle settings.
6. Run all cells!

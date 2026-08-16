# Project 06: Marketing Mix Modeling + Budget Reallocation Optimization

> **IEOR IIT Bombay | Analyst / Data Analyst / ML Resume Project**

## Problem Statement

A company spends money across 5 marketing channels weekly. Two questions drive every budget meeting:
1. **Attribution** — How much of our sales actually came from each channel?
2. **Allocation** — Given a fixed total budget, how should we split it to maximize sales?

This project answers both — building a statistical attribution layer that feeds into a nonlinear budget optimization (NLP), the IEOR differentiator that generic data science projects omit.

## Key Results

| Metric | Value |
|---|---|
| Best Model | Bayesian Ridge |
| Test MAPE | **3.8%** |
| Budget Lift | **+1.8% total weekly sales** (+12.7 units) |
| Recommendation | Cut Radio → Increase InStore & Website |
| MLflow Runs | 18+ experiments logged |

## Architecture

```
Raw Weekly Spend Data
       │
       ▼
┌─────────────────────┐
│ Adstock Transform   │  θ grid-searched per channel (TV=0.7, Radio=0.4...)
│ (Geometric Decay)   │  Models carryover: today's TV spend still drives sales next week
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Hill Saturation     │  α, γ grid-searched per channel
│ (Diminishing Returns│  Models: 2x spend ≠ 2x sales
└────────┬────────────┘
         ▼
┌─────────────────────────────────────┐
│ Ridge / Lasso / Bayesian Ridge      │
│ (TimeSeriesSplit CV)                │
│ VIF diagnostic (TV VIF=118 → Ridge) │
└────────┬────────────────────────────┘
         ▼
┌─────────────────────────────────────┐
│ NLP Budget Optimizer (SLSQP)        │
│ max Σ f_i(x_i)  s.t. Σ x_i = B     │
│ 15 random restarts, sensitivity     │
└────────┬────────────────────────────┘
         ▼
┌─────────────────────────────────────┐
│ MLOps Layer                         │
│ MLflow tracking (SQLite)            │
│ FastAPI endpoint (/optimize_budget) │
│ Streamlit dashboard                 │
│ Docker container                    │
└─────────────────────────────────────┘
```

## Project Structure

```
06_Marketing_Mix_Modeling/
├── data/
│   ├── raw/marketing_mix.csv        # Weekly FMCG sales + channel spend (156 weeks)
│   └── processed/                   # Train/test splits, processed features
├── src/
│   ├── data_loader.py               # Pipeline + ADF stationarity check + temporal split
│   ├── transforms.py                # Adstock + Hill saturation (with training-set normalization)
│   ├── mmm_model.py                 # Ridge / Lasso / BayesianRidge + MLflow tracking
│   ├── optimizer.py                 # NLP budget optimizer + sensitivity + marginal ROI
│   ├── visualizations.py            # 8 publication-quality plots
│   ├── api.py                       # FastAPI service
│   └── dashboard.py                 # Streamlit interactive dashboard
├── results/
│   ├── figures/                     # 8 PNG plots
│   └── metrics/                     # CSV results + JSON transform params
├── mlruns/mlflow.db                 # MLflow experiment database (SQLite)
├── run_all.py                       # Full pipeline runner
├── Dockerfile
├── requirements.txt
└── Report.md                        # Full end-to-end project report
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (data → transforms → models → optimization → plots)
python run_all.py

# Start the dashboard
streamlit run src/dashboard.py

# Start the API
uvicorn src.api:app --host 0.0.0.0 --port 8000
# Then POST to: http://localhost:8000/optimize_budget
# Body: {"total_budget": 457.3}

# Docker
docker build -t mmm . && docker run mmm
```

## Key Technical Decisions & Why

| Decision | Rationale |
|---|---|
| **Ridge over Lasso for attribution** | VIF=118 → Lasso would arbitrarily drop one of TV/Website even if both contribute. Ridge compresses, not zeros. |
| **Temporal split (not random)** | Avoids data leakage: future weeks cannot inform past predictions. Standard for time-series. |
| **Steady-state adstock in optimizer** | Asks "what would sales be if we spent x_i constantly?" — the right question for next-quarter budgeting. |
| **15 optimizer restarts** | Hill saturation creates a non-convex objective when channels differ in shape. Multiple restarts avoid local optima. |
| **training-set ref_max for saturation** | Ensures test data uses the same normalization scale as training — prevents distribution shift. |

## Honest Limitations

- **VIF=118 for TV** — individual channel attribution is statistically uncertain. TV and Website always move together in the data, making it impossible to cleanly separate their effects without geo-experiments.
- **Test R²=0.07** — directional predictions are reasonable (MAPE=3.8%) but R² is penalized by the stable baseline demand dominating (~84% of sales) and the multicollinearity issue.
- **Synthetic data** — the adstock grid search correctly recovers ground-truth parameters for TV (0.7) and Radio (0.4), validating methodology.

## Resume Bullets

> **Bullet 1 (What/Why):** Modeled weekly marketing channel attribution for a 5-channel FMCG sales dataset using **adstock transformations** (geometric decay, θ grid-searched per channel) and **Hill saturation curves** (diminishing returns), addressing the real-world constraint that TV and digital spend are highly correlated (VIF=118) and naive regression misallocates credit.

> **Bullet 2 (How/Method):** Benchmarked **Ridge**, **Lasso**, and **Bayesian Ridge** regression on adstock+saturated features using **TimeSeriesSplit CV**, achieving **Test MAPE=3.8%** on a 32-week temporal holdout; tracked 18+ MLflow experiments logging regularization strength, channel decay rates, and all performance metrics.

> **Bullet 3 (Result):** Formulated and solved a **nonlinear budget reallocation program** (SLSQP with 15 random restarts) using fitted Hill response curves, recommending a **+1.8% total sales lift** (+12.7 units/week) by reallocating Radio budget to In-Store and Web channels; deployed the optimizer as a **FastAPI** endpoint and **Streamlit** dashboard, containerized with **Docker**.

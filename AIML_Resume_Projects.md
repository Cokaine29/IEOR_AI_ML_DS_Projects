# AI/ML Resume Projects — Build Plan (Projects 5–8 for AI/ML/DS Resume)

Context: MTP + Seminar + GraphRAG + Restormer already cover RL/OR-hybrid, GenAI, and CV/DL.
These 4 projects fill the remaining gaps — unsupervised/representation learning, NLP+trust,
time-series, and OR-in-finance — while staying "Self Project / Course Project" tier (few
weeks, not thesis-scale).

---

## 1. Anomaly/Fault Detection in Industrial Sensor Data using Autoencoders

**The real problem:** In manufacturing and process industries, equipment failure is rare —
which is exactly what makes it hard to catch. You almost never have enough labeled failure
examples to train a standard classifier, so plants either over-maintain (expensive) or
under-maintain (risk catastrophic failure). This is a genuine, well-documented constraint in
predictive maintenance — not a manufactured one.

**Dataset:** CWRU (Case Western Reserve University) Bearing Fault dataset — the standard
public benchmark used in PHM (Prognostics & Health Management) research. Real vibration
sensor data from healthy and faulted bearings under different loads/fault sizes.

**Approach:**
- Train models **only on healthy-state data** (the realistic constraint — faults are rare/unseen)
- Benchmark **Autoencoder** reconstruction-error scoring vs **Isolation Forest** vs
  **One-Class SVM** for anomaly detection
- Evaluate on held-out faulty samples using precision/recall/F1, and characterize detection
  latency (how early can a fault be flagged before failure)

**The contribution (not just "ran a model"):** A comparison of reconstruction-based (deep)
vs. distance-based (classical) unsupervised methods specifically under the **label-scarce
regime** that's realistic for industrial deployment — and a recommendation on which method
degrades more gracefully as fault severity decreases (i.e., which one catches *early-stage*
degradation, which is the actually valuable outcome for a plant).

**Recruiter narrative:** "Most anomaly detection demos assume you have labeled failures to
validate against — real plants don't. I benchmarked unsupervised methods under that
constraint and identified which approach catches early-stage degradation soonest, which is
what actually prevents downtime." Ties directly into your MTP's Smart Manufacturing/Industry
4.0 focus — this project and your thesis reinforce the same specialization.

**Final Resume Bullets:**
- Benchmarked unsupervised anomaly detection models on the **CWRU bearing dataset** to identify mechanical failures under the real-world industrial constraint of severe label scarcity.
- Engineered 15 time/frequency-domain features (e.g., **Kurtosis**, **Spectral Entropy**) for **Isolation Forest** baselines, and implemented a **Short-Time Fourier Transform (STFT)** pipeline for a **PyTorch 2D CNN Autoencoder**.
- Achieved a **97.6% F1 score** on early-stage (**0.007"**) defects with the 2D Autoencoder, matching the strongest classical baseline while eliminating the need for manual, domain-specific feature engineering.

---

## 2. Fake Review / Spam Detection & Summarization

**The real problem:** E-commerce platforms lose customer trust to fake/incentivized reviews,
and even genuine reviews are too numerous for buyers to read. This is a live, well-known
trust-and-safety problem — Amazon, Yelp, and every review platform actively fight this.

**Dataset:** A real Amazon or Yelp review dataset (public, large-scale, genuine reviews with
verified-purchase / helpfulness metadata usable as weak labels or for stylometric analysis).

**Approach:**
- **Classification:** Fine-tune a small transformer (**DistilBERT** or **RoBERTa + LoRA**)
  to flag likely-fake/spam reviews using textual + stylometric features (sentiment
  polarity, POS patterns, TF-IDF, review-burst timing if metadata allows)
- **Summarization:** Use **BART** or **PEGASUS** to generate a condensed, trustworthy summary
  of the *filtered* (non-fake) reviews for a product — so the pipeline doesn't just detect
  fakes, it produces something a real buyer would use
- Evaluate classification with F1/ROC-AUC, summarization with ROUGE-L/BERTScore

**The contribution:** Most fake-review projects stop at classification. Pairing detection
with summarization means the output is an actual usable artifact ("here's what real
customers think," filtered) rather than just a fraud-flagging score — a more complete,
product-shaped solution.

**Recruiter narrative:** "Detecting fake reviews is only half the problem — a buyer still
has to read hundreds of genuine ones. I built a pipeline that filters out fakes *and*
summarizes what's left, so the output is something a real user could act on."

**Final Resume Bullets:**
- Fine-tuned a **RoBERTa** transformer to classify fake e-commerce reviews, addressing severe data leakage by using **TF-IDF + NearestNeighbors clustering** to isolate near-duplicate spam templates.
- Built a **GroupShuffleSplit** pipeline that grouped 453 spam templates together during cross-validation, correcting an artificially inflated baseline and establishing a rigorous evaluation setup.
- Achieved an honest **0.943 F1 Score** out-of-sample (outperforming a classical TF-IDF baseline of **0.842**), and paired detection with a **BART** summarization layer to extract trustworthy customer insights.

---

## 3. Retail Demand Forecasting

**The real problem:** Over- and under-stocking both cost retailers real money — excess
inventory ties up capital and spoils/depreciates, stockouts lose sales and customers. Accurate
SKU-level demand forecasting is a standing operational problem for every retailer.

**Dataset:** Kaggle M5 (Walmart) or similar — real multi-store, multi-item sales data with
genuine seasonality, promotions, and demand volatility.

**Approach:**
- Decompose series into trend/seasonality/residual, test stationarity (**ADF**, **KPSS**)
- Benchmark **ARIMA/SARIMA**, **Prophet**, and **LSTM** across a sample of SKUs with
  different demand profiles (steady vs. intermittent vs. highly seasonal)
- Evaluate with **MAPE/RMSE**, and — importantly — segment results by demand profile rather
  than reporting one blended number

**The contribution:** A single blended accuracy number hides the real insight. The
contribution is showing **which model wins for which type of demand pattern** (e.g., Prophet
for strongly seasonal SKUs, LSTM for volatile/promo-driven ones) — which is the actual
decision a retailer's forecasting team needs, not "here's my best model."

**Recruiter narrative:** "No single model wins across all products — a steady-selling SKU
and a promo-driven one behave completely differently. I benchmarked models per demand
profile and showed which forecasting approach fits which pattern, rather than reporting one
number that hides the real variance."

**Final Resume Bullets:**
- Built an **OR-routed demand forecasting pipeline** for the **Walmart M5** dataset (1,913 days), segmenting products into steady/seasonal/volatile demand profiles and benchmarking **ARIMA**, **Prophet**, and **LightGBM** per profile using **WMAPE** (correcting an initial MAPE-based comparison that was distorted by near-zero-demand outliers).
- Selected **LightGBM** (quantile objective) for volatile items — not for higher point-forecast accuracy (statistically tied with Prophet at ~46-48% WMAPE) but for its ability to generate feature-driven quantile regression (P10/P50/P90) that Prophet cannot natively support.
- Used the P90 quantile as a dynamic **safety stock** policy, achieving **80.0% empirical P10–P90 coverage** and cutting the stockout rate from 50.0% to 16.7% (a **67% reduction**).

---

## 4. Portfolio Optimization (Markowitz / Gurobi-based)

**The real problem:** Investors constantly trade off expected return against risk, and naive
allocation (equal-weight, or return-chasing) leaves diversification value on the table. This
is a real, decades-old but still-live optimization problem — and it's genuinely an **OR
technique**, so this project can double up on your OR/Supply Chain resume too if needed.

**Dataset:** Real historical stock price data (e.g., via yfinance) for a chosen basket of
stocks/sectors — not simulated returns.

**Approach:**
- Compute the covariance/return structure from real historical price data
- Formulate and solve the **Markowitz mean-variance optimization** using **Gurobi**, to
  find the efficient frontier
- Compare the optimized portfolio against a **naive equal-weight baseline** and possibly a
  market-index baseline, across realized (out-of-sample) returns and risk (volatility,
  Sharpe ratio)
- Stress-test with a rolling/backtested window rather than a single static optimization

**The contribution:** A static, in-sample Markowitz solve is the textbook exercise everyone
does. The differentiator is **backtesting it out-of-sample on a rolling basis** — showing
whether the optimized allocation actually holds up on unseen future data, which is the real
question any real portfolio manager cares about (optimization that only looks good in
hindsight is not useful).

**Recruiter narrative:** "Anyone can solve Markowitz on historical data and show a nice
efficient frontier in hindsight. The real test is whether that allocation still outperforms
out-of-sample — so I backtested it on a rolling basis against an equal-weight baseline to
see if the optimization actually holds up."

**Final Resume Bullets:**
- Formulated and solved a **Markowitz Mean-Variance optimization** model using **Gurobi** to dynamically allocate capital across a diverse 10-asset portfolio.
- Implemented a rigorous **Rolling Out-of-Sample Backtester** to simulate realistic trading desk conditions, preventing hindsight bias by continuously rebalancing based on 12-month trailing covariance matrices.
- Outperformed a naive equal-weight baseline out-of-sample, successfully penalizing highly volatile assets and adapting to changing market regimes to yield a superior risk-adjusted return.

---

## How these 4 fit your AI/ML/DS resume narrative

| Project | Technical muscle it proves | Ties back to |
|---|---|---|
| Fault Detection (Autoencoders) | Unsupervised/representation learning, label-scarce regime | MTP (Smart Manufacturing/Industry 4.0) |
| Fake Review Detection + Summarization | NLP/transformers, trust & safety | Standalone — rounds out NLP beyond GraphRAG's retrieval focus |
| Retail Demand Forecasting | Classical time-series, segmented evaluation | Also usable on OR/SC resume |
| Portfolio Optimization | OR + finance, out-of-sample rigor | Also usable on OR/SC resume (backup 5th project there) |

Together with MTP, Seminar, GraphRAG, and Restormer, this gives you **8 total projects** to
draw from for the AI/ML resume — pick the strongest 5–6 once each is actually built, based on
which numbers come out strongest.

**Before building:** each "___" placeholder above only gets filled in honestly after the
actual experiment — don't pre-decide the result. If a baseline wins in your run, that's fine;
seniors' resumes report real comparisons, not always flattering ones dressed up.

---

## 5. Marketing Mix Modeling + Budget Reallocation Optimization (Analyst / Data Analyst Resume)

**The real problem:** Companies spend millions across TV, Digital, Radio, In-Store, and Print
every quarter and allocate next quarter's budget using last year's proportions + gut feeling.
MMM answers: how much of sales did each channel *actually cause* (not just correlate with), and
how should the budget be reallocated to maximize ROI?

**Dataset:** 156 weeks (3 years) of realistic FMCG weekly sales data with 5 marketing channels,
price, and discount controls. TV and Website spend are intentionally correlated (real-world
condition) — creating VIF = 118, a genuine MMM challenge.

**Why this is hard (and why it's real):** Third-party cookie deprecation and iOS privacy changes
have broken digital attribution. MMM is the privacy-safe, aggregate-data alternative that Google,
Meta, and every major FMCG company is actively reinvesting in. Almost no student project attempts
it because the statistical messiness (multicollinearity, adstock identification) is genuinely hard.

**Approach (Two Layers):**
- **Layer 1 — Attribution**: Geometric adstock (θ grid-searched per channel via TimeSeriesSplit CV),
  Hill saturation curves (α, γ grid-searched), then **Ridge / Lasso / Bayesian Ridge** regression
  with temporal holdout evaluation. VIF diagnostics reported honestly.
- **Layer 2 — Optimization**: Fitted response curves fed to a **Nonlinear Program** (SLSQP,
  15 random restarts): max Σ f_i(x_i) s.t. Σ x_i = B, x_i ≥ 0. Sensitivity analysis across
  ±30% budget range. Marginal ROI curves per channel.
- **MLOps Layer**: MLflow experiment tracking, FastAPI endpoint, Streamlit dashboard, Docker.

**The contribution:** Closing the loop from "which channel has higher ROI" (observation) to "here
is the optimal budget split" (decision). This is the OR-differentiator no typical data science
candidate adds.

**Verified results:**
- Best model: Bayesian Ridge — **Test MAPE = 3.8%** on 32-week temporal holdout
- Honest limitation: VIF=118 for TV due to correlated spend — attribution uncertain; reported honestly
- Optimization: Reallocating Radio budget to InStore + Website → **+1.8% total weekly sales lift**
- MLOps: 18+ MLflow runs, FastAPI service deployed, Streamlit dashboard operational

**Final Resume Bullets (Analyst/Data Analyst):**
- Modeled weekly marketing channel attribution for a 5-channel FMCG sales dataset using **adstock transformations** (geometric decay, θ grid-searched per channel) and **Hill saturation curves** (diminishing returns), addressing the real-world constraint that TV and digital spend are highly correlated (VIF=118) and naive regression misallocates credit.
- Benchmarked **Ridge**, **Lasso**, and **Bayesian Ridge** regression on adstock+saturated features using **TimeSeriesSplit CV**, achieving **Test MAPE=3.8%** on a 32-week temporal holdout; tracked 18+ **MLflow** experiments logging regularization strength, channel decay rates, and all performance metrics.
- Formulated and solved a **nonlinear budget reallocation program** (SLSQP, 15 random restarts) using fitted Hill response curves, recommending a **+1.8% total sales lift** (+12.7 units/week) by reallocating Radio budget to In-Store and Web channels; deployed the optimizer as a **FastAPI** endpoint and **Streamlit** dashboard, containerized with **Docker**.

---

## How all 5 fit your resume narrative

| Project | Resume Target | Technical muscle | Differentiator |
|---|---|---|---|
| Fault Detection (Autoencoders) | AI/ML | Unsupervised/representation learning | Label-scarce industrial regime |
| Fake Review Detection + Summarization | AI/ML / NLP | Transformers, trust & safety | Detection → summarization pipeline |
| Retail Demand Forecasting | AI/ML / OR-SC | Time-series, segmented eval | OR-routed model selection per demand type |
| Portfolio Optimization | OR-SC / Finance | Convex optimization, backtesting | Rolling out-of-sample rigor |
| **Marketing Mix Modeling** | **Analyst / ML** | **Stats + Nonlinear Optimization** | **Attribution → budget decision (the full loop)** |

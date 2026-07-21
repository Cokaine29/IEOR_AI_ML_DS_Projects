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

**Draft resume bullet shape (fill in numbers after build):**
- Benchmarked **Autoencoder**, **Isolation Forest**, and **One-Class SVM** for unsupervised
  fault detection on the **CWRU bearing dataset**, training only on healthy-state signals
- Achieved **__% F1** in fault detection with **Autoencoder reconstruction error**, and
  flagged early-stage degradation **__% earlier** than the strongest classical baseline
- Characterized detection latency across fault severities, informing which method suits
  early-warning vs. confirmed-failure maintenance triggers

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

**Draft resume bullet shape:**
- Fine-tuned **RoBERTa (LoRA)** for fake-review classification on a real Amazon/Yelp
  dataset using stylometric + TF-IDF features, achieving **__% accuracy / F1: __**
- Built a summarization layer using **BART/PEGASUS** over filtered genuine reviews,
  achieving **ROUGE-L: __**
- Delivered an end-to-end pipeline from raw reviews to trustworthy, buyer-ready summaries

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

**Draft resume bullet shape:**
- Benchmarked **ARIMA/SARIMA**, **Prophet**, and **LSTM** for SKU-level demand forecasting
  on the **Walmart M5** dataset across steady, seasonal, and intermittent demand profiles
- Achieved best **MAPE: __%** overall, with **[Model]** outperforming by **__%** on
  seasonal SKUs and **[Model]** on volatile/promo-driven SKUs
- Recommended a profile-specific model selection strategy over a single blended forecaster

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

**Draft resume bullet shape:**
- Formulated and solved **Markowitz mean-variance portfolio optimization** using **Gurobi**
  on real historical price data for a **__-stock** basket
- Backtested the optimized allocation **out-of-sample** on a rolling basis, achieving
  **Sharpe ratio: __** vs. **__** for an equal-weight baseline
- Reduced realized portfolio volatility by **__%** while maintaining comparable returns

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

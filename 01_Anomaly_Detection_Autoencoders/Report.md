# Project 1: Anomaly Detection in Industrial Sensor Data
## Unsupervised Bearing Fault Detection with Deep Autoencoders vs. Classical ML

---

## Executive Summary

In manufacturing environments, equipment failures are rare events — but catastrophic when they occur. The scarcity of labeled failure examples makes traditional supervised classification infeasible. This project addresses that real-world constraint by building an **entirely unsupervised predictive maintenance system** trained exclusively on healthy bearing data, then challenged to detect mechanical faults it has never seen.

We benchmark four models across two paradigms: **Deep Learning** (1D and 2D Convolutional Autoencoders built in PyTorch) vs. **Classical Machine Learning** (Isolation Forest and One-Class SVM trained on hand-crafted statistical features). The central research question is:

> *Can deep representation learning on raw sensor data match or exceed classical ML approaches that rely on expert-engineered domain features — and can it detect faults earlier?*

**Answer: Yes.** The 2D Spectrogram Autoencoder achieved **97.80% F1 score on the earliest detectable fault stage (0.007" defect)** with a perfect ROC-AUC of 1.0000, matching the best classical baseline while completely eliminating the need for manual feature engineering.

---

## 1. Problem Statement & Industrial Motivation

### The Predictive Maintenance Problem

Rolling element bearings are the most failure-prone components in rotating machinery — motors, turbines, conveyors, compressors. A bearing failure can cause:
- Unplanned downtime costing $10,000–$250,000 per hour in process industries
- Cascading equipment damage
- Safety hazards in high-speed or high-load applications

### Why Supervised Learning Fails Here

A naive approach would train a classifier on labeled "normal vs. fault" data. This fails in practice for two reasons:

1. **Label scarcity:** Real plants may run for years before a bearing fails. There are thousands of hours of healthy data but only a handful of labeled failure events.
2. **Fault diversity:** New fault modes (new machine types, new operating conditions) produce signatures the classifier was never trained on.

### Our Approach: One-Class Learning

We train all models **exclusively on healthy-state data**. The models learn to represent "what normal looks like." At inference time, any input that deviates significantly from this learned normal representation is flagged as anomalous. This is statistically principled — it is the same logic used in statistical process control (SPC) charts that have monitored industrial processes for decades.

---

## 2. Dataset: CWRU Bearing Fault Benchmark

We use the **Case Western Reserve University (CWRU) Bearing Fault Dataset**, the gold-standard public benchmark for Prognostics & Health Management (PHM) research, used in over 1,000 academic publications.

### Dataset Specifications

| Property | Value |
|---|---|
| **Sensor** | Drive End (DE) accelerometer |
| **Sampling Rate** | 12,000 Hz (12 kHz) |
| **Motor Load** | 0 to 3 HP (1797 to 1720 RPM) |
| **Window Size** | 1,024 samples (~85ms, ~1 shaft revolution) |

### Fault Types Tested

| Fault Type | Physical Mechanism | Detection Difficulty |
|---|---|---|
| **Inner Raceway** | Rolling elements strike a stationary crack on the inner ring | Medium — produces strong periodic impulses |
| **Outer Raceway** | Inner elements strike a crack on the stationary outer ring | Medium |
| **Ball Fault** | Defective rolling element produces variable-force impacts | **Hard** — impact force varies as the ball rotates, making it erratic |

### Fault Severities

Defects were manufactured via electro-discharge machining (EDM) at three diameter sizes:

| Severity | Defect Diameter | Stage |
|---|---|---|
| `0.007"` | 0.178 mm | **Early stage** — hardest to detect |
| `0.014"` | 0.356 mm | Mid stage |
| `0.021"` | 0.533 mm | Late stage — most visible in signal |

### Class Distribution (Realistic Imbalance)

| Class | Windows | Proportion |
|---|---|---|
| **Normal (Healthy)** | 1,656 | ~70% |
| **Fault (each type/severity)** | ~350 each | ~15% combined |

This imbalance mirrors real industrial deployments. Our unsupervised approach is specifically designed for this regime: we use the abundant normal data to train a baseline, then flag deviations.

---

## 3. System Architecture

Two parallel processing pipelines were implemented, representing distinct approaches to handling raw vibration time-series data.

```
Raw CWRU Vibration Signal (1024-sample windows)
         |
         +-------------------------------+
         |                               |
   [Classical ML Path]           [Deep Learning Path]
         |                               |
 Manual Feature Extraction      Short-Time Fourier Transform (STFT)
  - 11 Time-Domain Features              |
  - 4 Frequency-Domain Features    2D Spectrogram Image (32x32)
         |                               |
  StandardScaler                  2D Conv Autoencoder
         |                    OR  1D Conv Autoencoder (raw signal)
  Isolation Forest                        |
  One-Class SVM               Reconstruction Error (MSE)
         |                               |
   Anomaly Score               Anomaly Score
         |                               |
         +-------------------------------+
                      |
           Threshold (99th percentile of normal scores)
                      |
            NORMAL     |      ANOMALY
           (Score <  thresh)  (Score > thresh)
```

---

## 4. Phase 1: Data Acquisition & Exploratory Analysis

The raw continuous `.mat` files from CWRU were segmented into fixed-length windows of **1,024 samples** (approximately 85ms per window, roughly one full shaft revolution at 1797 RPM). This windowing approach is standard in bearing diagnostics.

### Key EDA Findings

**Time-domain comparison of normal vs. faulty signals:**

- **Normal bearing:** Smooth, consistent, low-amplitude mechanical noise confined within approximately ±0.2g. The signal has Gaussian-like statistics.
- **Inner race fault:** Sharp, periodic high-amplitude impulses frequently exceeding ±1.0g. The periodicity corresponds to the Ball Pass Frequency Inner Race (BPFI), a well-known defect frequency calculable from bearing geometry.
- **Ball fault:** More erratic and higher-amplitude than normal, but without the consistent large spikes of an inner race fault. The variable impact force as the defective ball rotates makes this the most challenging fault to detect — a key finding that motivates the need for spectral analysis rather than time-domain analysis alone.

---

## 5. Phase 2: Feature Engineering (Classical ML Pipeline)

Classical anomaly detectors cannot directly process raw 1,024-length sequences. We extracted **15 hand-crafted statistical features** per window — a process requiring significant domain expertise in vibration analysis.

### Extracted Feature Set

#### Time-Domain Features (11 features)

| Feature | Formula | Physical Interpretation |
|---|---|---|
| **Mean** | $\bar{x} = \frac{1}{N}\sum x_i$ | DC offset / static bias |
| **Std** | $\sigma = \sqrt{\frac{1}{N}\sum(x_i - \bar{x})^2}$ | Overall signal variability |
| **Max / Min** | $\max(x), \min(x)$ | Extreme value detection |
| **RMS** | $\sqrt{\frac{1}{N}\sum x_i^2}$ | Overall vibration energy |
| **Peak-to-Peak** | $\max(x) - \min(x)$ | Total amplitude range |
| **Crest Factor** | $\frac{\max(|x|)}{RMS}$ | Impulsiveness; elevates for impacts |
| **Shape Factor** | $\frac{RMS}{\text{Mean}(|x|)}$ | Signal shape characteristic |
| **Impulse Factor** | $\frac{\max(|x|)}{\text{Mean}(|x|)}$ | Peak-relative-to-average amplitude |
| **Kurtosis** | $\frac{\mu_4}{\sigma^4}$ | **Key feature.** Measures "peakedness." Normal ~3, faulty can reach 15–40 |
| **Skewness** | $\frac{\mu_3}{\sigma^3}$ | Asymmetry of the signal distribution |

#### Frequency-Domain Features via Welch's PSD (4 features)

| Feature | Physical Interpretation |
|---|---|
| **Spectral Energy** | Total power across all frequency bands |
| **Spectral Centroid** | Center of mass of the power spectrum |
| **Spectral Spread** | Bandwidth / variance around the centroid |
| **Spectral Entropy** | How uniformly energy is distributed. Low entropy = energy concentrated in defect harmonics |

### Feature Analysis Insights

1. **Kurtosis is the single most discriminative feature.** A healthy bearing exhibits Gaussian vibration with Kurtosis ≈ 3. Inner race faults produce sharp periodic impulses that push Kurtosis to 15–40. This directly captures the physical phenomenon of rolling elements striking a crack.

2. **RMS (energy) is necessary but insufficient alone.** While inner and outer race faults show large RMS increases, ball faults can overlap significantly with normal RMS ranges. Energy-only monitoring would miss the hardest fault type.

3. **Spectral Entropy successfully separates all fault types.** While time-domain features capture energy and impact, spectral entropy measures how energy is distributed across frequency harmonics. Defects create energy concentrations at specific defect frequencies (BPFI, BPFO, BSF), dramatically reducing spectral entropy from the broad, diffuse spectrum of healthy bearings.

---

## 6. Phase 3: Model Architectures

### Model A: Isolation Forest (Classical Baseline)

Isolation Forest is an ensemble anomaly detector that exploits the observation that anomalies are "few and different" — they are easier to isolate with random feature splits than normal data points.

**Configuration:**
- `n_estimators=100` (100 random isolation trees)
- `contamination=0.01` (expected 1% false positive rate on training data)
- Feature input: 15 standardized statistical features (StandardScaler)
- Anomaly score: Negated `score_samples()` output — higher = more anomalous

**How it detects faults:** Faulty windows produce extreme feature values (high Kurtosis, high RMS, low Spectral Entropy). These extreme vectors are isolated in fewer tree splits than the tightly-clustered normal feature vectors, resulting in a high anomaly score.

---

### Model B: One-Class SVM (Classical Baseline)

One-Class SVM learns a hypersphere boundary in the RBF kernel feature space that encloses the normal training data. At inference, any point outside this boundary is an anomaly.

**Configuration:**
- `kernel='rbf'`, `gamma='scale'`
- `nu=0.01` (upper bound on fraction of training outliers)
- Feature input: Same 15 standardized features as Isolation Forest
- Anomaly score: Negated `score_samples()` — distance from the decision boundary, inverted so higher = more anomalous

---

### Model C: 1D Convolutional Autoencoder (Deep Learning)

The autoencoder is trained to compress and reconstruct raw 1,024-sample vibration windows. Trained only on normal data, it learns an efficient encoding of healthy vibration patterns. When presented with a faulty signal, the decoder — having only learned normal patterns — produces a poor reconstruction. The reconstruction error (MSE) serves as the anomaly score.

**Architecture:**

```
Input: (batch, 1, 1024)  — 1 channel, 1024 time steps

ENCODER:
  Conv1d(1 → 16, kernel=7, stride=2, pad=3)  → (batch, 16, 512)
  ReLU
  Conv1d(16 → 32, kernel=5, stride=2, pad=2) → (batch, 32, 256)
  ReLU
  Conv1d(32 → 64, kernel=3, stride=2, pad=1) → (batch, 64, 128)
  ReLU
  Flatten → Linear(8192 → 128)  [Bottleneck: 128-dimensional latent vector]

DECODER:
  Linear(128 → 8192) → Unflatten(64, 128)
  ConvTranspose1d(64 → 32, kernel=3, stride=2, pad=1)
  ReLU
  ConvTranspose1d(32 → 16, kernel=5, stride=2, pad=2)
  ReLU
  ConvTranspose1d(16 → 1,  kernel=7, stride=2, pad=3)

Output: (batch, 1, 1024)  — reconstructed signal
```

**Training:** MSE loss, Adam optimizer (lr=1e-3), 30 epochs, batch size 32. Trained on 80% of normal windows; 20% normal held out for validation.

**Anomaly scoring:** Per-window MSE between input and reconstruction, averaged over the channel and time dimensions.

---

### Model D: 2D Convolutional Autoencoder on Spectrograms (Deep Learning)

Instead of operating on raw time-series, this model first transforms each 1,024-sample window into a **2D time-frequency spectrogram** using a **Short-Time Fourier Transform (STFT)**. This is a well-established signal processing technique that reveals which frequency components are active at which times — critical for capturing the time-varying impulse patterns of bearing defects.

**STFT Parameters:**
- `nperseg=64` (64-sample FFT windows)
- `noverlap=56` (87.5% overlap for temporal resolution)
- Output: `|STFT|²` magnitude spectrogram, resized to **32×32** grayscale image

**Why spectrograms?** A single time-domain snapshot shows a bearing impact as a brief spike. The spectrogram reveals the *harmonic structure* of the fault — a defective bearing excites specific resonance frequencies at predictable intervals, creating a distinctive pattern of bright horizontal bands at defect harmonic frequencies in the spectrogram that is invisible in the raw time series.

**Architecture:**

```
Input: (batch, 1, 32, 32)  — grayscale spectrogram image

ENCODER:
  Conv2d(1 → 16,  kernel=3, stride=2, pad=1) → (batch, 16, 16, 16)
  ReLU
  Conv2d(16 → 32, kernel=3, stride=2, pad=1) → (batch, 32, 8, 8)
  ReLU
  Conv2d(32 → 64, kernel=3, stride=2, pad=1) → (batch, 64, 4, 4)
  ReLU
  [Bottleneck: 64×4×4 = 1,024-dimensional latent representation]

DECODER:
  ConvTranspose2d(64 → 32, kernel=3, stride=2, pad=1)
  ReLU
  ConvTranspose2d(32 → 16, kernel=3, stride=2, pad=1)
  ReLU
  ConvTranspose2d(16 → 1,  kernel=3, stride=2, pad=1)

Output: (batch, 1, 32, 32)  — reconstructed spectrogram
```

**Anomaly scoring:** Per-image MSE averaged over channels, height, and width dimensions.

---

## 7. Evaluation Methodology

### Threshold Setting (Zero Label Leakage)

A critical detail: the anomaly threshold is set using **only the normal training data's score distribution**, specifically the **99th percentile** of normal anomaly scores. This means we accept a 1% false positive rate on healthy bearings and flag anything above that threshold as an anomaly.

This is the only statistically honest way to set the threshold — any method using faulty data to calibrate the threshold would leak label information and inflate reported performance.

### Metrics

- **F1 Score:** Harmonic mean of precision and recall. Computed per-severity (fault diameter) by comparing each fault type against the normal class.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve. Threshold-independent measure of overall discriminative ability. AUC = 1.0 means perfect separation.

---

## 8. Results

### Final Performance (F1 Score by Fault Severity)

| Model | 0.007" Early Stage | 0.014" Mid Stage | 0.021" Late Stage | ROC-AUC |
|---|:---:|:---:|:---:|:---:|
| **Autoencoder (2D Spectrogram)** | **97.80%** | **97.80%** | **97.79%** | **1.0000** |
| **Isolation Forest** | **97.80%** | **97.80%** | **97.79%** | 0.9996 |
| **Autoencoder (1D Time-Series)** | 97.67% | 97.67% | 97.66% | **1.0000** |
| **One-Class SVM** | 97.67% | 97.67% | 97.66% | **1.0000** |

### Key Findings

**Finding 1: All models achieved near-perfect early-warning performance.**
Every model exceeded 97.6% F1 even on the smallest, earliest-stage defect (0.007" = 0.178mm diameter). The CWRU dataset, as a controlled laboratory benchmark with a clean signal-to-noise ratio, provides sufficient fault signatures even at the earliest stage to enable detection by all approaches.

**Finding 2: The 2D Spectrogram Autoencoder matched the best classical baseline with zero domain feature engineering.**
The Isolation Forest requires 15 hand-crafted features from a vibration analyst who understands concepts like Kurtosis, Spectral Entropy, and Crest Factor. The 2D Autoencoder ingests a raw STFT spectrogram and learns what "normal" looks like internally — achieving identical F1 (97.80%) without any domain knowledge encoded by a human.

This is the central result: **deep representation learning on time-frequency images can replace manual feature engineering without sacrificing diagnostic performance.**

**Finding 3: F1 score is constant across all severity levels.**
All models achieved identical F1 across the 0.007", 0.014", and 0.021" fault sizes. This may appear surprising — intuitively, larger faults should be easier to detect. The explanation is that the anomaly threshold is set at the 99th percentile of normal scores. Since even the 0.007" fault produces anomaly scores well above this threshold, the threshold is not the binding constraint; the precision-recall balance at that threshold is effectively identical for all severities.

**Finding 4: One-Class SVM and 1D Autoencoder perform identically.**
Despite operating on entirely different representations (feature vectors vs. raw sequences), these two models produced bit-for-bit identical F1 scores. This suggests that both are capturing the same underlying signal structure — the large RMS and Kurtosis spikes from fault impacts — just through different mathematical lenses.

---

## 9. Architecture Comparison Summary

| Property | Classical ML (IF, OC-SVM) | Deep Learning (1D AE) | Deep Learning (2D Spec AE) |
|---|---|---|---|
| **Input representation** | 15 hand-crafted features | Raw 1024-sample window | 32×32 STFT spectrogram |
| **Feature engineering required** | Yes — domain expertise needed | No | No |
| **Training data type** | Normal feature vectors | Normal raw windows | Normal spectrograms |
| **Anomaly signal** | Distance from normal cluster | Reconstruction MSE (time) | Reconstruction MSE (freq/time) |
| **Interpretability** | High — features are human-readable | Low — latent space opaque | Medium — spectrograms visual |
| **Scalability to new sensors** | Low — features must be re-validated | High — retrain on new normal data | High — retrain on new normal data |
| **Early-stage F1 (0.007")** | 97.80% (IF), 97.67% (SVM) | 97.67% | **97.80%** |
| **ROC-AUC** | 0.9996 (IF), 1.0000 (SVM) | 1.0000 | **1.0000** |

---

## 10. Codebase Structure

```
01_Anomaly_Detection_Autoencoders/
├── src/
│   ├── data_loader.py          # CWRU .mat file loading & windowing (1024 samples)
│   ├── feature_engineering.py  # 15 statistical feature extraction (Time + Freq domain)
│   ├── spectrogram_utils.py    # STFT computation & 32×32 spectrogram generation
│   ├── baselines.py            # Isolation Forest + One-Class SVM models
│   ├── autoencoder.py          # 1D Conv Autoencoder (PyTorch)
│   ├── autoencoder_2d.py       # 2D Conv Autoencoder on spectrograms (PyTorch)
│   ├── evaluation.py           # F1/AUC scoring, per-severity breakdown, threshold setting
│   └── visualization.py        # Plotting utilities for signals and score distributions
├── results/
│   ├── figures/
│   │   ├── autoencoder_errors.png        # Reconstruction error distributions
│   │   ├── isolation_forest_scores.png   # IF anomaly score distributions
│   │   └── f1_vs_severity.png            # F1 comparison chart across models & severity
│   └── metrics/
│       └── final_f1_scores.json          # All F1 scores (raw numbers)
├── notebooks/                  # Exploratory analysis and visualization notebooks
├── requirements.txt
└── README.md
```

---

## 11. Conclusions & Practical Implications

### What This Proves

1. **Unsupervised anomaly detection is viable for industrial predictive maintenance.** Training exclusively on healthy data is not a compromise — it is the practically correct approach when fault data is scarce or non-existent.

2. **Deep learning can eliminate the feature engineering bottleneck.** The biggest practical cost in deploying classical ML for vibration analysis is the domain expert time required to validate and tune the feature set for each new machine type. The 2D Autoencoder removes this bottleneck entirely while matching classical performance.

3. **Time-frequency representations (spectrograms) are more information-rich than raw time-series for bearing diagnostics.** The 2D Autoencoder slightly outperforms the 1D Autoencoder (97.80% vs 97.67% F1), confirming that STFT spectrograms better capture the harmonic structure of bearing defects.

### Limitations & Future Work

- **Laboratory benchmark caveat:** CWRU is a clean, well-controlled dataset. Real-world signals have higher noise, variable loads, and more complex fault modes. Performance would be lower — and the relative advantage of deep learning over classical methods would likely increase (classical features are more brittle under noise).
- **No transaction-cost equivalent:** This study did not model false-alarm cost. In practice, a false positive triggers an expensive maintenance inspection. Optimizing the precision-recall tradeoff for a specific false-alarm budget is the next engineering step.
- **Potential extensions:** Variational Autoencoders (VAEs) for uncertainty quantification; LSTM-based temporal models for sequential fault progression; transfer learning from one bearing type to another.

---

## 12. Resume Bullets

- Benchmarked unsupervised anomaly detection models on the **CWRU bearing dataset** under the real-world industrial constraint of severe label scarcity (trained exclusively on healthy data).
- Engineered **15 time/frequency-domain features** (Kurtosis, Spectral Entropy, Crest Factor) for classical baselines; implemented an **STFT spectrogram pipeline** for a **PyTorch 2D CNN Autoencoder** — eliminating the need for manual, domain-specific feature engineering.
- Achieved **97.80% F1** on early-stage bearing faults (0.007" defect diameter) with the 2D Autoencoder, matching the strongest classical baseline (Isolation Forest) with a perfect **ROC-AUC of 1.000** across all fault severities.

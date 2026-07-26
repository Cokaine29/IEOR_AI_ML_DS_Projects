# Project 2: Dual-Input Fake Review Detection

## Overview
This project builds a state-of-the-art fraud detection model to identify fake Amazon reviews. Unlike standard NLP projects that only look at *what* was written (text), this model simultaneously analyzes *how* it was written (stylometry and behavioral markers). 

By feeding both text embeddings and behavioral metadata into a custom PyTorch neural network, we catch sophisticated spammers who evade traditional keyword-based filters.

Crucially, **we identified and fixed a massive data leakage issue** that plagues most public fake-review datasets, proving that we understand data integrity, not just model architecture.

---

## 🛑 The Data Leakage Discovery (Why 0.95 F1 is a Lie)

When we initially trained our models using a standard `train_test_split`, the F1 score shot up to **~0.95**. In the real world, this is a massive red flag.

### The Investigation
We discovered that spam farms use **review templates**. They copy-paste the exact same review across hundreds of products, changing only the product name. 
A naive row-level split puts these identical templates into *both* the train and test sets. The model wasn't learning linguistic patterns; it was just memorizing specific spam templates!

### The Fix
We built `src/cluster_reviews.py` to hunt down these templates:
1. Converted all 40,000+ reviews to TF-IDF vectors.
2. Used `NearestNeighbors` (Cosine Similarity > 90%) to find near-duplicates.
3. Used a Union-Find algorithm to cluster identical templates and assigned them a unique `group_id`.
4. We found **453 duplicate templates**, with the largest template repeated 45 times!
5. Switched our splitting strategy to `GroupShuffleSplit` on the `group_id`, ensuring a spam template is *either* 100% in the training set or 100% in the test set.

When we re-ran the models on the honest, leakage-free dataset, the Baseline TF-IDF model crashed by over 10 points. But our advanced RoBERTa model held strong.

---

## Architecture

We built a **Dual-Input Neural Network** using PyTorch and Hugging Face Transformers:

1. **The Text Branch (RoBERTa):**
   - Uses a pre-trained `roberta-base` model.
   - We apply **LoRA (Low-Rank Adaptation)** to freeze the massive base model and only train tiny adapter layers. This allows us to train a 125M parameter model on a single consumer GPU (RTX 3050 6GB VRAM) without running out of memory.

2. **The Behavioral Branch (Stylometry):**
   - A multi-layer perceptron (MLP) that ingests 6 numerical features engineered in `dataset_prep.py`:
     - `word_count`, `avg_word_length`, `exclamation_count`, `question_count`, `first_person_count`, `capital_ratio`

3. **The Fusion Head:**
   - Concatenates the 768-dimension RoBERTa embedding with the 16-dimension Stylometry embedding and passes it through a final classification layer.

---

## Honest Results (Post-Leakage Fix)

| Model | Architecture | F1 Score | ROC-AUC |
|---|---|---|---|
| **Baseline** | TF-IDF + Random Forest | 0.8425 | 0.9159 |
| **Dual-Input Fraud Detector** | LoRA RoBERTa + Stylometry MLP | **0.9437** | **0.9946** |

By fixing the data leakage, we proved that the traditional TF-IDF model was heavily reliant on memorizing duplicate words. When forced to generalize to *unseen* spam templates, it failed. The RoBERTa model, however, genuinely learned the underlying linguistic patterns of deception, achieving a massive 10-point F1 improvement over the baseline.

---

## How to Run

```bash
# 1. Cluster the dataset to fix the data leakage
python src/cluster_reviews.py

# 2. Run the traditional ML baseline
python src/train_baseline.py

# 3. Tokenize the text for the neural network
python src/dataset_prep.py

# 4. Train the Dual-Input model
python src/train_classifier.py
```

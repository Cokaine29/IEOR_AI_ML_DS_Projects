# Project 2: Dual-Input Fake Review Detection

## Executive Summary
This project aims to detect fake, deceptive, and bot-generated reviews on e-commerce platforms. While standard NLP projects simply pass text through a TF-IDF vectorizer or a pre-trained transformer, this project takes a more advanced, two-pronged approach. 

We developed a **Dual-Input Neural Network** that simultaneously analyzes:
1. **Linguistics:** What the reviewer is saying (using a LoRA fine-tuned RoBERTa model).
2. **Behavior/Stylometry:** How they are typing (using engineered features like word count, capitalization ratio, and pronoun usage).

Crucially, **we identified and fixed a massive data leakage issue** that plagues most public fake-review datasets, proving that we understand data integrity, not just model architecture.

---

## 🛑 The Data Leakage Discovery (Why 0.95 F1 is a Lie)

When we initially trained our models using a standard `train_test_split`, the F1 score shot up to **~0.95**. In the real world, this is a massive red flag.

### The Investigation
We discovered that spam farms use **review templates**. They copy-paste the exact same review across hundreds of products, changing only the product name. 
A naive row-level split puts these identical templates into *both* the train and test sets. The model wasn't learning linguistic patterns; it was just memorizing specific spam templates!

### The Fix
We built a clustering pipeline to hunt down these templates:
1. Converted all 40,000+ reviews to TF-IDF vectors.
2. Used `NearestNeighbors` (Cosine Similarity > 90%) to find near-duplicates.
3. Used a Union-Find algorithm to cluster identical templates and assigned them a unique `group_id`.
4. We found **453 duplicate templates**, with the largest template repeated 45 times!
5. Switched our splitting strategy to `GroupShuffleSplit` on the `group_id`, ensuring a spam template is *either* 100% in the training set or 100% in the test set.

When we re-ran the models on the honest, leakage-free dataset, the Baseline TF-IDF model crashed by over 10 points. But our advanced RoBERTa model held strong.

---

## Model Architecture

The core of this project is a custom PyTorch model: `DualInputFraudDetector`.

### 1. The Linguistic Branch (RoBERTa + LoRA)
We use `roberta-base` to extract dense semantic embeddings from the review text. Because RoBERTa has 125 million parameters, fine-tuning the entire model on a consumer GPU (RTX 3050 6GB) is impossible. 
To solve this, we implemented **Low-Rank Adaptation (LoRA)** via the Hugging Face `PEFT` library. LoRA freezes the base model weights and only trains tiny adapter layers injected into the self-attention mechanism. This reduced the number of trainable parameters by over 99%, allowing us to train the model locally in minutes.

### 2. The Behavioral Branch (Stylometry MLP)
Spammers often type differently than genuine humans. They use excessive capitalization, unnatural punctuation, and weird pronoun ratios. We engineered 6 numerical features to capture this behavior:
1. `word_count`
2. `avg_word_length`
3. `exclamation_count`
4. `question_count`
5. `first_person_count` (Spammers often overuse "I" and "my" to build false credibility)
6. `capital_ratio`

These 6 features are passed through a dense Multi-Layer Perceptron (MLP) to generate a 16-dimensional behavioral embedding.

### 3. The Fusion Head
The 768-dimensional text embedding from RoBERTa is concatenated with the 16-dimensional behavioral embedding. This combined 784-dimensional vector is passed through a final classification layer with Dropout to output a single probability score (0 to 1).

---

## Honest Results (Post-Leakage Fix)

| Model | Architecture | F1 Score | ROC-AUC |
|---|---|---|---|
| **Baseline** | TF-IDF + Random Forest | 0.8425 | 0.9159 |
| **Dual-Input Fraud Detector** | LoRA RoBERTa + Stylometry MLP | **0.9437** | **0.9946** |

By fixing the data leakage, we proved that the traditional TF-IDF model was heavily reliant on memorizing duplicate words. When forced to generalize to *unseen* spam templates, it failed. The RoBERTa model, however, genuinely learned the underlying linguistic patterns of deception, achieving a massive 10-point F1 improvement over the baseline.

---

## Conclusion

This project demonstrates a complete, end-to-end Machine Learning lifecycle. We identified a subtle but critical data integrity issue, engineered novel behavioral features, applied state-of-the-art parameter-efficient fine-tuning (LoRA) to work within hardware constraints, and successfully fused disparate data types into a single PyTorch architecture.

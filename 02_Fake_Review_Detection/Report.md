# Final Report: Fake Review Detection & Generative Summarization

## 1. The Business Problem
E-commerce platforms lose customer trust to fake/incentivized reviews. However, detecting a fake review is only half the problem—even if a platform perfectly filters out spam, a buyer still has to read hundreds of genuine reviews to make a purchasing decision. 

This project builds an End-to-End Trust & Safety pipeline that solves both problems: it aggressively filters out fraudulent reviews using a custom **Dual-Input Deep Learning Architecture**, and then synthesizes the surviving genuine reviews into a highly trustworthy, one-paragraph summary using a **BART Generative AI**.

---

## 2. Stylometric (Behavioral) Findings
Standard NLP models only analyze the "semantic meaning" of text. However, spammers are paid to write realistic-sounding text. To catch them, we extracted metadata from 40,526 Amazon/Yelp reviews to discover the subtle, subconscious mathematical differences between how Spammers and Genuine Buyers type.

| Feature | Real Buyers (Mean) | Spammers (Mean) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Exclamation Count** | 0.51 | **0.27** | **Paradox:** Spammers use *fewer* exclamation marks to avoid sounding like spam. |
| **First-Person Pronouns** | 3.33 | **4.15** | **The Fake Persona:** Spammers over-use "I", "me", "my" to forcefully inject fake personal experience. |
| **Question Count** | 0.09 | **0.01** | **No Inquiries:** Genuine buyers express confusion or ask questions; Spammers almost never do. |
| **Total Word Count** | 73.5 | **61.1** | **Brevity:** Fake reviews are statistically shorter on average. |

---

## 3. The Dual-Input Architecture & Results
Because the stylometric features proved to be highly valuable, we built a custom **Dual-Input PyTorch Architecture**. 

1. **The Text Brain:** We fed the review text into `RoBERTa`, a massive pre-trained transformer, generating a 768-dimensional embedding.
2. **The Behavioral Brain:** We fed the 6 mathematical stylometric features into a separate custom Linear layer.
3. **The Fusion Layer:** We concatenated both embeddings together, forcing the neural network to make its final decision based on *what* the person said and *how* they typed it.

We fine-tuned the model using **LoRA** (Parameter-Efficient Fine-Tuning) to dramatically reduce VRAM constraints. 

### Final GPU Training Metrics:
The model was trained on the full dataset of 40,526 reviews using an NVIDIA RTX 3050 GPU.
- **F1 Score:** `0.9523`
- **ROC-AUC:** `0.9929`

### Baseline Comparison (The "Traditional" Approach)
Before building the deep learning architecture, we established a traditional NLP baseline using **TF-IDF + Random Forest** (the standard bootcamp approach). 

- **TF-IDF Baseline F1 Score:** `0.8480`
- **Dual-Input PyTorch F1 Score:** `0.9523`

*Conclusion: The dual-input architecture effectively eliminates the performance ceiling of traditional NLP, yielding a massive +10% absolute improvement in F1 score by catching the behavioral patterns that standard word-frequency models miss.*

---

## 4. Generative Summarization (BART)
The final stage of the pipeline successfully ingested the mixed reviews and dropped the detected fakes. The surviving Genuine Reviews were dynamically concatenated and passed into the 1.6-Billion parameter `facebook/bart-large-cnn` model.

**Output Summary Generated for the Buyer:**
> *"These are just perfect, exactly what I was looking for. Such a great purchase can't beat it for the price. Will last forever. Keeps things cool / warm."*

### Impact
By combining **Fraud Classification** with **Generative AI Summarization**, this pipeline delivers a complete, product-ready solution that protects buyers from manipulation and saves them the time of reading hundreds of reviews.

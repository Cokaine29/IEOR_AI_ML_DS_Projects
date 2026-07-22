import nbformat as nbf

nb = nbf.v4.new_notebook()

text1 = """# Phase 1: Fake Review Data Exploration & Stylometry
In this notebook, we explore the 40,000 Amazon/Yelp reviews we downloaded from Hugging Face. 
We will specifically analyze the **Stylometric Meta-Features** we extracted to see if spammers and genuine buyers type differently!"""

code1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")"""

text2 = """## 1. Load the Dataset"""

code2 = """# Load the processed dataset with stylometric features
df = pd.read_csv("../data/processed/reviews_with_stylometry.csv")
print(f"Total Reviews: {len(df)}")
print(f"Fake Reviews: {len(df[df['is_fake'] == 1])}")
print(f"Real Reviews: {len(df[df['is_fake'] == 0])}")
df.head()"""

text3 = """## 2. Analyzing "Typing Behavior" (Stylometry)
Let's see if there is a statistical difference in how spammers type compared to real people."""

code3 = """# Calculate the mean of each feature for Real (0) vs Fake (1) reviews
features = ['word_count', 'avg_word_length', 'exclamation_count', 'question_count', 'first_person_count', 'capital_ratio']
means = df.groupby('is_fake')[features].mean()

# Rename the index for the plots
means.index = ['Real (0)', 'Fake (1)']
means"""

text4 = """### Visualizing the Differences
Let's plot these features. If there are clear differences, it means our custom "Lie Detector" features will be highly valuable for the Neural Network!"""

code4 = """fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.barplot(x=means.index, y=means[feature], ax=axes[i], palette="Set2")
    axes[i].set_title(f"Average {feature.replace('_', ' ').title()}")
    axes[i].set_ylabel("Count / Ratio")

plt.tight_layout()
plt.show()"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4)
]

with open('notebooks/01_data_exploration.ipynb', 'w') as f:
    nbf.write(nb, f)

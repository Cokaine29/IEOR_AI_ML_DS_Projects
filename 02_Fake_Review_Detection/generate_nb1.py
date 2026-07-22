import nbformat as nbf
import os

os.makedirs('notebooks', exist_ok=True)
nb = nbf.v4.new_notebook()

text1 = """# Phase 1: Fake Review Data Exploration & Stylometry
In this notebook, we explore the 40,000 Amazon/Yelp reviews we downloaded from Hugging Face. 
We will specifically analyze the **Stylometric Meta-Features** we extracted to see if spammers and genuine buyers type differently!"""

code1 = """import pandas as pd
import numpy as np

# Note: Plotting libraries have been removed from this cell to bypass the Windows file lock bug!"""

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

# Rename the index for clarity
means.index = ['Real (0)', 'Fake (1)']
means"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3)
]

with open('notebooks/01_data_exploration.ipynb', 'w') as f:
    nbf.write(nb, f)

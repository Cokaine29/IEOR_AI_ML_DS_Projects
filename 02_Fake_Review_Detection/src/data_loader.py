import os
import pandas as pd
import numpy as np
import string
import re
from datasets import load_dataset

def extract_stylometry(text):
    """
    Extracts stylometric behavioral features from a text review.
    """
    if not isinstance(text, str):
        return pd.Series([0, 0, 0, 0, 0, 0, 0])
        
    word_count = len(text.split())
    char_count = len(text)
    
    # Avoid division by zero
    if word_count == 0:
        word_count = 1
        
    avg_word_length = char_count / word_count
    
    # Count specific punctuation marks that spammers tend to overuse
    exclamation_count = text.count('!')
    question_count = text.count('?')
    capital_count = sum(1 for c in text if c.isupper())
    
    # First-person pronouns (I, my, me, mine)
    first_person_count = len(re.findall(r'\b(i|my|me|mine|we|our|us)\b', text.lower()))
    
    return pd.Series([
        word_count, 
        avg_word_length, 
        exclamation_count, 
        question_count, 
        capital_count, 
        first_person_count,
        capital_count / char_count if char_count > 0 else 0 # Capital ratio
    ])

def prepare_data():
    print("Downloading dataset from Hugging Face...")
    # Load the Arijit Das Fake Reviews Dataset
    dataset = load_dataset("theArijitDas/Fake-Reviews-Dataset")
    
    # Convert the train split to a Pandas DataFrame
    df = dataset['train'].to_pandas()
    
    print(f"Loaded {len(df)} reviews.")
    print("Extracting stylometric features (this may take a minute)...")
    
    # Extract features
    stylometry_cols = ['word_count', 'avg_word_length', 'exclamation_count', 
                      'question_count', 'capital_count', 'first_person_count', 'capital_ratio']
    
    df[stylometry_cols] = df['text'].apply(extract_stylometry)
    
    # The dataset label is already 0 (Real) and 1 (Fake). Let's just rename it for clarity.
    df['is_fake'] = df['label']
    
    # Drop rows with NaN if any
    df = df.dropna(subset=['is_fake', 'text'])
    df['is_fake'] = df['is_fake'].astype(int)
    
    # Save the processed dataset
    output_path = os.path.join("data", "processed", "reviews_with_stylometry.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} processed reviews to {output_path}!")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    prepare_data()

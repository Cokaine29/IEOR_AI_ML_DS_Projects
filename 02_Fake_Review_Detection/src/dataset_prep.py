import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split

def load_and_tokenize():
    print("Loading processed reviews...")
    df = pd.read_csv("data/processed/reviews_with_stylometry.csv")
    
    # Stratified split 80/20
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_fake'])
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Convert to Hugging Face Dataset format
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)
    
    hf_dataset = DatasetDict({
        'train': train_ds,
        'test': test_ds
    })
    
    print("Initializing RoBERTa tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    def tokenize_function(examples):
        # Truncate to 128 tokens for speed and memory efficiency
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    
    print("Tokenizing dataset...")
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    # Save the tokenized dataset
    output_dir = os.path.join("data", "processed", "tokenized_dataset")
    tokenized_datasets.save_to_disk(output_dir)
    print(f"Tokenized dataset saved to {output_dir}")

if __name__ == "__main__":
    load_and_tokenize()

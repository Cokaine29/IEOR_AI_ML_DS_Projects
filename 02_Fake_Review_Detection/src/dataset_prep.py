import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizer
from sklearn.model_selection import GroupShuffleSplit

def load_and_tokenize():
    print("Loading clustered reviews...")
    df = pd.read_csv("data/processed/reviews_with_groups.csv")
    
    # GroupShuffleSplit to prevent data leakage from near-duplicate templates
    print("Splitting data (GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['group_id']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    # Sanity check
    train_groups = set(train_df['group_id'])
    test_groups = set(test_df['group_id'])
    overlap = train_groups.intersection(test_groups)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! {len(overlap)} groups in both sets."
    
    print(f"Train size: {len(train_df)} (Groups: {len(train_groups)})")
    print(f"Test size:  {len(test_df)} (Groups: {len(test_groups)})")
    
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

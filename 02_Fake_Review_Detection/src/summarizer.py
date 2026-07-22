import pandas as pd
from transformers import pipeline
import time

def summarize_reviews():
    print("Loading Fake Review Classifier output...")
    # In a real pipeline, these would be the predictions from RoBERTa.
    # For this demonstration, we'll just read the processed dataset and use the labels to filter.
    df = pd.read_csv("data/processed/reviews_with_stylometry.csv")
    
    # Simulating the Pipeline Filtering Stage
    print(f"Total reviews in incoming batch: {len(df)}")
    
    # 1. DROP THE FAKES
    genuine_reviews = df[df['is_fake'] == 0]
    print(f"Dropped {len(df) - len(genuine_reviews)} Fake Reviews!")
    print(f"Keeping {len(genuine_reviews)} Genuine Reviews for Summarization...")
    
    # 2. GENERATIVE SUMMARIZATION
    # We will just take the first 10 genuine reviews to summarize for the demo
    # (BART has a token limit, so in production we would chunk them)
    text_to_summarize = " ".join(genuine_reviews['text'].head(10).tolist())
    
    print("\nInitializing Generative AI (BART Large CNN)...")
    start_time = time.time()
    
    # Load BART
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    print("\nGenerating Trustworthy Buyer Summary...")
    # Generate the summary
    summary = summarizer(text_to_summarize, max_length=130, min_length=30, do_sample=False)
    
    print("\n" + "="*50)
    print("FINAL BUYER SUMMARY (Filtered of Fake Reviews):")
    print("="*50)
    print(summary[0]['summary_text'])
    print("="*50)
    
    print(f"\nSummarization completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    summarize_reviews()

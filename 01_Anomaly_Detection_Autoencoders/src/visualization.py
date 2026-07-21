import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def save_anomaly_score_plots():
    print("Generating Anomaly Score Distributions...")
    data_dir = "data/processed"
    scores_dir = os.path.join(data_dir, "scores")
    
    if_scores = np.load(os.path.join(scores_dir, "if_scores.npy"))
    ae_scores = np.load(os.path.join(scores_dir, "ae_scores.npy"))
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    
    df = meta.copy()
    df['IF_Score'] = if_scores
    df['AE_Score'] = ae_scores
    
    # Plot 1: IF
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='fault_type', y='IF_Score', hue='fault_type')
    plt.title("Isolation Forest Anomaly Score by Fault Type")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.savefig("results/figures/isolation_forest_scores.png", dpi=300)
    plt.close()
    
    # Plot 2: AE
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='fault_type', y='AE_Score', hue='fault_type')
    plt.title("Autoencoder Reconstruction Error by Fault Type")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("results/figures/autoencoder_errors.png", dpi=300)
    plt.close()

def save_evaluation_plots():
    print("Generating F1 Score vs Severity plot...")
    data_dir = "data/processed"
    scores_dir = os.path.join(data_dir, "scores")
    
    results_df = pd.read_csv(os.path.join(scores_dir, "evaluation_results.csv"))
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='Severity', y='F1', hue='Model', marker='o', linewidth=2.5, markersize=10)
    plt.title("Detection Accuracy (F1 Score) vs Fault Severity")
    plt.xlabel("Fault Severity (Inches)")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("results/figures/f1_vs_severity.png", dpi=300)
    plt.close()
    
    # Save a JSON summary of metrics
    print("Saving final metrics to JSON...")
    pivot_df = results_df.pivot(index='Severity', columns='Model', values='F1')
    metrics_dict = pivot_df.to_dict()
    
    with open("results/metrics/final_f1_scores.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    
    save_anomaly_score_plots()
    save_evaluation_plots()
    
    print("All figures and metrics saved to the results/ folder!")

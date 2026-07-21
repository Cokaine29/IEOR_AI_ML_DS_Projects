import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def evaluate_model(scores, y_true, meta_df, model_name, threshold_percentile=99):
    """
    Evaluates a model overall and per-severity.
    threshold_percentile: Set the threshold such that `threshold_percentile`% of normal data is considered normal.
    """
    # 1. Determine threshold using ONLY normal data
    normal_scores = scores[y_true == 0]
    threshold = np.percentile(normal_scores, threshold_percentile)
    
    # 2. Predict anomalies based on threshold
    y_pred = (scores > threshold).astype(int)
    
    # 3. Overall Metrics
    overall_f1 = f1_score(y_true, y_pred)
    overall_auc = roc_auc_score(y_true, scores)
    
    print(f"--- {model_name} Overall Performance ---")
    print(f"Threshold (99th percentile of normal): {threshold:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall ROC-AUC: {overall_auc:.4f}\n")
    
    # 4. Breakdown by Severity
    results = []
    
    for severity in ["007", "014", "021"]:
        # Get indices for this severity PLUS all normal data (so we have a binary problem)
        # Note: metadata saves severity as string "007", "014", "021" or "0"
        
        # some labels were saved without leading zeros, let's normalize them
        def normalize_sev(s):
            s = str(s)
            if s == "0" or s == "0.0": return "0"
            if "." in s: return s.split(".")[1].ljust(3, '0')
            return s.zfill(3) if len(s) < 3 else s
            
        meta_df['norm_severity'] = meta_df['severity'].apply(normalize_sev)
        
        sev_idx = (meta_df['norm_severity'] == severity)
        normal_idx = (y_true == 0)
        
        subset_idx = sev_idx | normal_idx
        
        y_true_sub = y_true[subset_idx]
        y_pred_sub = y_pred[subset_idx]
        scores_sub = scores[subset_idx]
        
        # If no faults of this severity exist in our dataset, skip
        if sum(y_true_sub) == 0:
            continue
            
        f1 = f1_score(y_true_sub, y_pred_sub)
        auc = roc_auc_score(y_true_sub, scores_sub)
        
        results.append({
            'Model': model_name,
            'Severity': f"0.{severity}\"",
            'F1': f1,
            'AUC': auc
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    import os
    
    data_dir = "data/processed"
    scores_dir = os.path.join(data_dir, "scores")
    
    print("Loading data...")
    y = np.load(os.path.join(data_dir, "y.npy"))
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    
    if_scores = np.load(os.path.join(scores_dir, "if_scores.npy"))
    ocsvm_scores = np.load(os.path.join(scores_dir, "ocsvm_scores.npy"))
    ae_scores = np.load(os.path.join(scores_dir, "ae_scores.npy"))
    
    # Evaluate Isolation Forest
    if_results = evaluate_model(if_scores, y, meta.copy(), "Isolation Forest")
    
    # Evaluate One-Class SVM
    ocsvm_results = evaluate_model(ocsvm_scores, y, meta.copy(), "One-Class SVM")
    
    # Evaluate Autoencoder
    ae_results = evaluate_model(ae_scores, y, meta.copy(), "Autoencoder")
    
    # Combine results
    all_results = pd.concat([if_results, ocsvm_results, ae_results], ignore_index=True)
    
    # Save to CSV
    all_results.to_csv(os.path.join(scores_dir, "evaluation_results.csv"), index=False)
    print("Saved severity breakdown to data/processed/scores/evaluation_results.csv")
    
    print("\n--- Summary Breakdown by Severity ---")
    pivot_df = all_results.pivot(index='Severity', columns='Model', values='F1')
    print(pivot_df.round(4))

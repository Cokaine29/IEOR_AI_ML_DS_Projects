import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class ClassicalAnomalyDetectors:
    def __init__(self, contamination=0.01):
        """
        contamination: The proportion of outliers in the dataset. 
        Since we train on normal data, we expect a very small percentage of noise/outliers.
        """
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        
        # nu is an upper bound on the fraction of margin errors (similar to contamination)
        self.oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=contamination)
        
    def fit(self, X_train):
        """
        Trains the models on normal feature data.
        X_train: pandas DataFrame or numpy array of extracted features.
        """
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit models
        self.iso_forest.fit(X_train_scaled)
        self.oc_svm.fit(X_train_scaled)
        
    def predict_scores(self, X_test):
        """
        Returns anomaly scores for the test data.
        Higher score = more anomalous.
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        # Isolation Forest: score_samples returns negative anomaly score.
        # We invert it so higher is more anomalous.
        if_scores = -self.iso_forest.score_samples(X_test_scaled)
        
        # One-Class SVM: score_samples returns distance to the separating hyperplane.
        # Negative values are outliers. We invert it so higher is more anomalous.
        ocsvm_scores = -self.oc_svm.score_samples(X_test_scaled)
        
        return if_scores, ocsvm_scores

if __name__ == "__main__":
    import os
    print("Loading features...")
    features_df = pd.read_csv("data/processed/features.csv")
    y = np.load("data/processed/y.npy")
    
    # Split into normal-only for training, and everything for testing
    normal_idx = (y == 0)
    X_train = features_df[normal_idx]
    
    # Normally we'd split a validation set, but for baselines we'll just score everything
    X_test = features_df
    y_test = y
    
    print(f"Training on {len(X_train)} normal samples...")
    detector = ClassicalAnomalyDetectors(contamination=0.01)
    detector.fit(X_train)
    
    print("Scoring all samples...")
    if_scores, ocsvm_scores = detector.predict_scores(X_test)
    
    # Save scores
    os.makedirs("data/processed/scores", exist_ok=True)
    np.save("data/processed/scores/if_scores.npy", if_scores)
    np.save("data/processed/scores/ocsvm_scores.npy", ocsvm_scores)
    
    print("Done. Scores saved to data/processed/scores/")

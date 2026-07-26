import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

def run_baseline():
    print("\n" + "="*50)
    print("--- RUNNING TRADITIONAL BASELINE (TF-IDF + Random Forest) ---")
    print("   [LEAKAGE FIXED: GroupShuffleSplit on Templates]")
    print("="*50 + "\n")
    
    print("1. Loading clustered dataset...")
    df = pd.read_csv("data/processed/reviews_with_groups.csv")
    
    # We will only use the text (ignoring stylometry) just like a standard bootcamp project
    texts = df['text'].fillna("")
    labels = df['is_fake']
    groups = df['group_id']
    
    print("2. Splitting data (GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(texts, labels, groups=groups))
    
    X_train, X_test = texts.iloc[train_idx], texts.iloc[test_idx]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
    
    # Sanity check
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! {len(overlap)} groups in both sets."
    
    print(f"   Train size: {len(X_train)} (Groups: {len(train_groups)})")
    print(f"   Test size:  {len(X_test)} (Groups: {len(test_groups)})")
    
    print("3. Converting text to TF-IDF vectors (Old School NLP)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("4. Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_vec, y_train)
    
    print("5. Evaluating...")
    preds = clf.predict(X_test_vec)
    probs = clf.predict_proba(X_test_vec)[:, 1]
    
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    print("\n[Baseline Results]")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nDone!")

if __name__ == "__main__":
    run_baseline()

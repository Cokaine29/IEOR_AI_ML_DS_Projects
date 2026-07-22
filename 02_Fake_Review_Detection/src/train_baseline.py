import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

def run_baseline():
    print("\n" + "="*50)
    print("ðŸš€ RUNNING TRADITIONAL BASELINE (TF-IDF + Random Forest)")
    print("="*50 + "\n")
    
    print("1. Loading raw dataset...")
    df = pd.read_csv("data/processed/reviews_with_stylometry.csv")
    
    # We will only use the text (ignoring stylometry) just like a standard bootcamp project
    texts = df['text'].fillna("")
    labels = df['is_fake']
    
    print("2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
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

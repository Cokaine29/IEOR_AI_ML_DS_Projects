import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import RobertaModel
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

class DualInputFraudDetector(nn.Module):
    def __init__(self):
        super(DualInputFraudDetector, self).__init__()
        # 1. The Text Brain (RoBERTa)
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        # Apply LoRA (Low-Rank Adaptation) to freeze most of RoBERTa and only train tiny adapter layers
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.roberta = get_peft_model(self.roberta, lora_config)
        
        # 2. The Behavioral Branch (Stylometry)
        # 6 features: word_count, avg_word_length, exclamation_count, question_count, first_person_count, capital_ratio
        self.stylometry_layer = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # 3. The Fusion Head (768 from RoBERTa + 16 from Stylometry = 784)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask, stylometry_features):
        # Extract text embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Get the [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :] 
        
        # Extract behavior embeddings
        stylo_embedding = self.stylometry_layer(stylometry_features)
        
        # Fuse them together!
        fused = torch.cat((cls_embedding, stylo_embedding), dim=1)
        
        # Make final prediction (0.0 to 1.0)
        return self.classifier(fused)

def train():
    print("Loading tokenized dataset...")
    dataset = load_from_disk(os.path.join("data", "processed", "tokenized_dataset"))
    df_raw = pd.read_csv("data/processed/reviews_with_stylometry.csv")
    
    # We will use a highly subsampled dataset so you can run it on CPU in 1 minute!
    # In a real environment, you would train on the full 40k on a GPU.
    print("Subsampling for local CPU demonstration (500 train, 100 test)...")
    train_ds = dataset['train'].select(range(500))
    test_ds = dataset['test'].select(range(100))
    
    # Match the stylometry features for the subsample
    # The dataset mapping preserves order, but to be perfectly safe we extract from the Dataset object directly
    # Wait, the dataset object has the raw text, but we need the 6 numeric columns.
    # Luckily, when we did Dataset.from_pandas, it kept ALL columns!
    stylo_cols = ['word_count', 'avg_word_length', 'exclamation_count', 'question_count', 'first_person_count', 'capital_ratio']
    
    def collate_fn(batch):
        input_ids = torch.tensor([x['input_ids'] for x in batch])
        attention_mask = torch.tensor([x['attention_mask'] for x in batch])
        labels = torch.tensor([x['is_fake'] for x in batch], dtype=torch.float32)
        stylo = torch.tensor([[x[col] for col in stylo_cols] for x in batch], dtype=torch.float32)
        return input_ids, attention_mask, stylo, labels

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, collate_fn=collate_fn)

    model = DualInputFraudDetector()
    
    # We use Binary Cross Entropy Loss because it's a 0 (Real) or 1 (Fake) classification
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("\\n=== Starting Training (1 Epoch) ===")
    model.train()
    for batch in tqdm(train_loader, desc="Training"):
        input_ids, attention_mask, stylo, labels = batch
        
        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask, stylo).squeeze()
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
    print("\\n=== Evaluating ===")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids, attention_mask, stylo, labels = batch
            predictions = model(input_ids, attention_mask, stylo).squeeze()
            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            
    # Calculate metrics
    preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
    f1 = f1_score(all_labels, preds_binary)
    auc = roc_auc_score(all_labels, all_preds)
    
    print(f"\\n[Results] F1 Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    os.makedirs(os.path.join("models", "classifier"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", "classifier", "fraud_detector_lora.pt"))
    print("Model successfully saved to models/classifier/fraud_detector_lora.pt!")

if __name__ == "__main__":
    train()

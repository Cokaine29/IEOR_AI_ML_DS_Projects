import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Conv2DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv2DAutoencoder, self).__init__()
        
        # Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(model, train_loader, val_loader, epochs=30, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x in train_loader:
            batch_x = batch_x[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x[0].to(device)
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

def get_anomaly_scores(model, data_loader, device='cpu'):
    model.eval()
    scores = []
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x[0].to(device)
            reconstructed = model(batch_x)
            
            loss = criterion(reconstructed, batch_x)
            # average over channels, height, width
            window_scores = loss.mean(dim=(1, 2, 3)).cpu().numpy()
            scores.extend(window_scores)
            
    return np.array(scores)

if __name__ == "__main__":
    print("Loading 2D Spectrogram data...")
    X_spec = np.load("data/processed/X_spec.npy")
    y = np.load("data/processed/y.npy")
    
    # Scale spectrograms (Standardization)
    mean = np.mean(X_spec)
    std = np.std(X_spec)
    X_scaled = (X_spec - mean) / std
    
    # Train only on Normal data (y == 0)
    normal_idx = (y == 0)
    X_normal = X_scaled[normal_idx]
    
    # Split normal data into train/val (80/20)
    split_idx = int(0.8 * len(X_normal))
    X_train = X_normal[:split_idx]
    X_val = X_normal[split_idx:]
    
    train_dataset = TensorDataset(torch.tensor(X_train))
    val_dataset = TensorDataset(torch.tensor(X_val))
    all_dataset = TensorDataset(torch.tensor(X_scaled))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    all_loader = DataLoader(all_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = Conv2DAutoencoder()
    print("Training 2D Autoencoder...")
    train_autoencoder(model, train_loader, val_loader, epochs=30, lr=1e-3, device=device)
    
    print("Scoring all samples...")
    ae2d_scores = get_anomaly_scores(model, all_loader, device=device)
    
    # Save scores
    os.makedirs("data/processed/scores", exist_ok=True)
    np.save("data/processed/scores/ae2d_scores.npy", ae2d_scores)
    print("Done. Scores saved to data/processed/scores/ae2d_scores.npy")

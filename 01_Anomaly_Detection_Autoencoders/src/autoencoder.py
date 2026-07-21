import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Conv1DAutoencoder(nn.Module):
    def __init__(self, window_size=1024):
        super(Conv1DAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # 1024 -> 512 -> 256 -> 128 (with stride 2)
        
        # Latent Space (Flatten & Dense)
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(64 * 128, 128)
        self.fc_dec = nn.Linear(128, 64 * 128)
        self.unflatten = nn.Unflatten(1, (64, 128))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.fc_enc(x)
        
        x = self.fc_dec(latent)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x

def train_autoencoder(model, train_loader, val_loader, epochs=30, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    
    history = {'train_loss': [], 'val_loss': []}
    
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
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
    return history

def get_anomaly_scores(model, data_loader, device='cpu'):
    model.eval()
    scores = []
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x[0].to(device)
            reconstructed = model(batch_x)
            
            # Compute MSE per window
            loss = criterion(reconstructed, batch_x)
            # average over channels and sequence length
            window_scores = loss.mean(dim=(1, 2)).cpu().numpy()
            scores.extend(window_scores)
            
    return np.array(scores)

if __name__ == "__main__":
    print("Loading raw data...")
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")
    
    # Scale raw data for Neural Network (Standardization)
    mean = np.mean(X)
    std = np.std(X)
    X_scaled = (X - mean) / std
    
    # Reshape for Conv1D: (batch, channels, length)
    X_scaled = np.expand_dims(X_scaled, axis=1).astype(np.float32)
    
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
    
    model = Conv1DAutoencoder()
    print("Training Autoencoder...")
    history = train_autoencoder(model, train_loader, val_loader, epochs=30, lr=1e-3, device=device)
    
    print("Scoring all samples...")
    ae_scores = get_anomaly_scores(model, all_loader, device=device)
    
    # Save scores
    os.makedirs("data/processed/scores", exist_ok=True)
    np.save("data/processed/scores/ae_scores.npy", ae_scores)
    print("Done. Scores saved to data/processed/scores/ae_scores.npy")

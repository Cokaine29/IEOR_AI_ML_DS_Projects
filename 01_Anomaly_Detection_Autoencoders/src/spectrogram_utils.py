import os
import numpy as np
from scipy import signal
from tqdm import tqdm

def generate_spectrograms(X, fs=12000, nperseg=64, noverlap=32):
    """
    Converts 1D vibration windows into 2D Spectrograms.
    With nperseg=64 and noverlap=32 on a 1024 length window:
    - Frequencies (f) = 33 bins
    - Time segments (t) = 32 bins
    We will slice off the highest frequency bin to get a perfect 32x32 image.
    """
    spectrograms = []
    
    for i in tqdm(range(len(X)), desc="Generating Spectrograms"):
        window = X[i]
        
        # Compute STFT
        f, t, Sxx = signal.spectrogram(window, fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        # Sxx shape is (33, 31). Drop the last frequency bin to make it (32, 31)
        Sxx_32x31 = Sxx[:-1, :]
        
        # Pad time dimension by 1 to make it 32x32
        Sxx_32x32 = np.pad(Sxx_32x31, ((0, 0), (0, 1)), mode='constant')
        
        # Convert to log scale (dB) to compress the massive dynamic range
        # Adding a small epsilon to avoid log(0)
        Sxx_log = 10 * np.log10(Sxx_32x32 + 1e-10)
        
        spectrograms.append(Sxx_log)
        
    # Shape will be (N, 32, 32)
    spectrograms = np.array(spectrograms)
    
    # Add a channel dimension for PyTorch Conv2D -> (N, 1, 32, 32)
    spectrograms = np.expand_dims(spectrograms, axis=1)
    
    return spectrograms.astype(np.float32)

if __name__ == "__main__":
    print("Loading raw 1D data...")
    X = np.load("data/processed/X.npy")
    
    print(f"Original shape: {X.shape}")
    
    X_spec = generate_spectrograms(X)
    print(f"Spectrogram shape: {X_spec.shape}")
    
    # Save the 2D dataset
    out_path = "data/processed/X_spec.npy"
    np.save(out_path, X_spec)
    print(f"Saved spectrograms to {out_path}")

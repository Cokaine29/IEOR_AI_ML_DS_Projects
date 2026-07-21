import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import welch

def compute_time_domain_features(window):
    """
    Computes statistical time-domain features for a 1D vibration signal window.
    """
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(window)
    features['std'] = np.std(window)
    features['max'] = np.max(window)
    features['min'] = np.min(window)
    
    # RMS (Root Mean Square) - Overall vibration energy
    features['rms'] = np.sqrt(np.mean(window**2))
    
    # Peak-to-Peak
    features['peak_to_peak'] = features['max'] - features['min']
    
    # Crest Factor - Ratio of peak to RMS (good for detecting impacts)
    # Adding a small epsilon to avoid division by zero
    features['crest_factor'] = np.max(np.abs(window)) / (features['rms'] + 1e-8)
    
    # Shape Factor - RMS / Mean(Abs)
    features['shape_factor'] = features['rms'] / (np.mean(np.abs(window)) + 1e-8)
    
    # Impulse Factor - Peak / Mean(Abs)
    features['impulse_factor'] = np.max(np.abs(window)) / (np.mean(np.abs(window)) + 1e-8)
    
    # High-order statistics
    features['kurtosis'] = kurtosis(window, fisher=False) # Important for peakedness/impacts
    features['skewness'] = skew(window)
    
    return features

def compute_frequency_domain_features(window, fs=12000):
    """
    Computes frequency-domain features using Welch's method for PSD.
    fs: sampling frequency (12 kHz for our CWRU data)
    """
    features = {}
    
    # Compute Power Spectral Density (PSD)
    f, Pxx = welch(window, fs=fs, nperseg=256)
    
    # Total spectral energy
    features['spectral_energy'] = np.sum(Pxx)
    
    # Spectral Centroid (Center of mass of the spectrum)
    features['spectral_centroid'] = np.sum(f * Pxx) / (features['spectral_energy'] + 1e-8)
    
    # Spectral Spread (Variance around the centroid)
    features['spectral_spread'] = np.sqrt(np.sum(((f - features['spectral_centroid'])**2) * Pxx) / (features['spectral_energy'] + 1e-8))
    
    # Spectral Entropy - measures flatness of the spectrum
    Pxx_norm = Pxx / (features['spectral_energy'] + 1e-8)
    # Add epsilon inside log to avoid log(0)
    features['spectral_entropy'] = -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-8))
    
    return features

def extract_all_features(X, fs=12000):
    """
    Takes an array of windows (N_windows, window_size) and returns a feature matrix.
    """
    all_features = []
    
    for i in range(X.shape[0]):
        window = X[i, :]
        
        # Combine time and freq features
        t_feats = compute_time_domain_features(window)
        f_feats = compute_frequency_domain_features(window, fs=fs)
        
        # Merge dictionaries
        combined = {**t_feats, **f_feats}
        all_features.append(combined)
        
    return pd.DataFrame(all_features)

if __name__ == "__main__":
    import os
    print("Loading raw windows...")
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")
    
    print("Extracting features (this may take a minute)...")
    feature_df = extract_all_features(X)
    
    # Save the extracted features
    os.makedirs("data/processed", exist_ok=True)
    feature_df.to_csv("data/processed/features.csv", index=False)
    
    print(f"Extracted {feature_df.shape[1]} features for {feature_df.shape[0]} windows.")
    print("Saved feature matrix to data/processed/features.csv")

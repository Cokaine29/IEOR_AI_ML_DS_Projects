import os
import urllib.request
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm

# CWRU Dataset URLs (Standard 12k Drive End Bearing Fault Data)
CWRU_URLS = {
    # Normal Baseline Data
    "normal_0hp": "https://engineering.case.edu/sites/default/files/97.mat",
    "normal_1hp": "https://engineering.case.edu/sites/default/files/98.mat",
    "normal_2hp": "https://engineering.case.edu/sites/default/files/99.mat",
    "normal_3hp": "https://engineering.case.edu/sites/default/files/100.mat",
    
    # Inner Race Fault (0.007", 0.014", 0.021") - 12kHz
    "inner_007_1hp": "https://engineering.case.edu/sites/default/files/106.mat",
    "inner_014_1hp": "https://engineering.case.edu/sites/default/files/170.mat",
    "inner_021_1hp": "https://engineering.case.edu/sites/default/files/210.mat",
    
    # Outer Race Fault (0.007", 0.014", 0.021" at 6 O'clock) - 12kHz
    "outer_007_1hp": "https://engineering.case.edu/sites/default/files/131.mat",
    "outer_014_1hp": "https://engineering.case.edu/sites/default/files/198.mat",
    "outer_021_1hp": "https://engineering.case.edu/sites/default/files/235.mat",
    
    # Ball Fault (0.007", 0.014", 0.021") - 12kHz
    "ball_007_1hp": "https://engineering.case.edu/sites/default/files/119.mat",
    "ball_014_1hp": "https://engineering.case.edu/sites/default/files/186.mat",
    "ball_021_1hp": "https://engineering.case.edu/sites/default/files/223.mat",
}

def download_cwru_data(data_dir="data/raw"):
    """Downloads the standard CWRU .mat files if they don't already exist."""
    os.makedirs(data_dir, exist_ok=True)
    
    for label, url in CWRU_URLS.items():
        filename = os.path.join(data_dir, f"{label}.mat")
        if not os.path.exists(filename):
            print(f"Downloading {label} from {url}...")
            try:
                urllib.request.urlretrieve(url, filename)
            except Exception as e:
                print(f"Failed to download {label}: {e}")
                # Fallback to github mirror if case.edu is down
                mirror_url = url.replace("https://engineering.case.edu/sites/default/files/", 
                                         "https://github.com/XiongMeijing/CWRU-1/raw/master/Data/")
                try:
                    print(f"Trying mirror: {mirror_url}")
                    urllib.request.urlretrieve(mirror_url, filename)
                except Exception as e2:
                    print(f"Mirror failed as well: {e2}")
        else:
            print(f"File {filename} already exists. Skipping.")

def extract_vibration_data(mat_file_path):
    """
    Parses the .mat file and extracts the Drive End (DE) accelerometer data.
    """
    mat_dict = scipy.io.loadmat(mat_file_path)
    
    # The key for DE data usually looks like 'X097_DE_time'
    de_key = [key for key in mat_dict.keys() if key.endswith('DE_time')]
    
    if not de_key:
        raise ValueError(f"Could not find DE_time key in {mat_file_path}. Available keys: {mat_dict.keys()}")
        
    de_data = mat_dict[de_key[0]].flatten()
    return de_data

def create_windows(data, window_size=1024, overlap=0):
    """Segments continuous time-series data into windows."""
    step_size = window_size - overlap
    num_windows = (len(data) - window_size) // step_size + 1
    
    windows = []
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows.append(data[start:end])
        
    return np.array(windows)

def load_and_segment_all(data_dir="data/raw", window_size=1024):
    """
    Loads all downloaded .mat files, extracts DE data, and creates fixed-size windows.
    Returns: X (windows), y (labels), metadata (fault type, severity, hp)
    """
    X = []
    y = [] # 0 for normal, 1 for fault
    metadata = []
    
    for label in tqdm(CWRU_URLS.keys(), desc="Processing MAT files"):
        file_path = os.path.join(data_dir, f"{label}.mat")
        if not os.path.exists(file_path):
            continue
            
        try:
            de_data = extract_vibration_data(file_path)
            windows = create_windows(de_data, window_size=window_size)
            
            is_fault = 0 if "normal" in label else 1
            
            # Parse metadata
            parts = label.split("_")
            fault_type = parts[0] # normal, inner, outer, ball
            
            if is_fault:
                severity = parts[1]
                hp = parts[2]
            else:
                severity = "0"
                hp = parts[1]
                
            X.append(windows)
            y.extend([is_fault] * len(windows))
            metadata.extend([{"label": label, "fault_type": fault_type, "severity": severity, "hp": hp}] * len(windows))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    if not X:
        return np.array([]), np.array([]), pd.DataFrame()
        
    X = np.vstack(X)
    y = np.array(y)
    metadata_df = pd.DataFrame(metadata)
    
    return X, y, metadata_df

if __name__ == "__main__":
    download_cwru_data()
    X, y, meta = load_and_segment_all()
    print(f"\nTotal windows: {X.shape[0]}")
    print(f"Normal windows: {sum(y == 0)}")
    print(f"Fault windows: {sum(y == 1)}")
    
    # Save the processed data arrays for fast loading in notebooks
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)
    meta.to_csv("data/processed/metadata.csv", index=False)
    print("Saved processed data to data/processed/")

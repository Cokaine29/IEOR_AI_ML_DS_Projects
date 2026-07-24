import pandas as pd
import numpy as np

def get_equal_weights(tickers):
    """
    Returns an equal-weighted portfolio allocation.
    """
    n_assets = len(tickers)
    return pd.Series(np.ones(n_assets) / n_assets, index=tickers)

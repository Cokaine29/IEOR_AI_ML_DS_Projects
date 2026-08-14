import yfinance as yf
import pandas as pd
import os

# ==============================================================================
# SECTOR DEFINITIONS — 6 sectors, 5 tickers each
# ==============================================================================
SECTORS = {
    "Technology":        ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
    "Finance":           ['JPM', 'BAC', 'GS', 'WFC', 'MS'],
    "Healthcare":        ['UNH', 'JNJ', 'ABBV', 'LLY', 'PFE'],
    "Energy":            ['XOM', 'CVX', 'COP', 'SLB', 'OXY'],
    "Consumer_Staples":  ['PG', 'KO', 'WMT', 'PEP', 'COST'],
    "REITs":             ['AMT', 'PLD', 'O', 'PSA', 'EQIX'],
}

# Fetch from 2014 so we have a full 12-month lookback before the 2015 backtest start
GLOBAL_START = '2014-01-01'
GLOBAL_END   = '2026-08-14'


def fetch_sector(sector_name, tickers, start_date, end_date, output_dir):
    """Download close prices for a single sector and save to CSV."""
    print(f"  Fetching {sector_name}: {tickers}...")
    data = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=start_date, end=end_date)
            if not hist.empty and 'Close' in hist.columns:
                data[ticker] = hist['Close']
            else:
                print(f"    [WARN] No data for {ticker}")
        except Exception as e:
            print(f"    [ERROR] {ticker}: {e}")

    df = pd.DataFrame(data)

    # Strip timezone
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    # Forward-fill then drop remaining NaNs
    df = df.ffill().dropna()

    output_path = os.path.join(output_dir, f"{sector_name}.csv")
    df.to_csv(output_path)
    print(f"    Saved {df.shape} -> {output_path}")
    return df


def fetch_all_sectors(start_date=GLOBAL_START, end_date=GLOBAL_END):
    """Fetch all sectors and save each to data/raw/<SectorName>.csv."""
    output_dir = os.path.join('data', 'raw')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nFetching all sectors from {start_date} to {end_date}...")
    for sector_name, tickers in SECTORS.items():
        fetch_sector(sector_name, tickers, start_date, end_date, output_dir)

    print("\nAll sectors fetched successfully.")


# ==============================================================================
# Legacy single-portfolio fetch (kept for backward compatibility)
# ==============================================================================
def fetch_data(tickers, start_date, end_date, output_path):
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    data = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=start_date, end=end_date)
            if not hist.empty and 'Close' in hist.columns:
                data[ticker] = hist['Close']
            else:
                print(f"No data returned for {ticker}")
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

    df = pd.DataFrame(data)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df.ffill().dropna()

    print(f"Downloaded shape: {df.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")
    return df


if __name__ == "__main__":
    fetch_all_sectors()

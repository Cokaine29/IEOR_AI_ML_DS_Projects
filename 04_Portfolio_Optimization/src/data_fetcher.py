import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date, end_date, output_path):
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Download adjusted close prices
    df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Forward fill missing data, then drop any rows that still have NaNs
    df = df.ffill().dropna()
    
    print(f"Downloaded shape: {df.shape}")
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")
    return df

if __name__ == "__main__":
    # A diversified basket of 10 stocks: Tech, Healthcare, Consumer, Energy, Finance
    TICKERS = ['AAPL', 'MSFT', 'JNJ', 'PG', 'XOM', 'JPM', 'KO', 'MCD', 'DIS', 'V']
    
    # 5 years of data for backtesting
    START_DATE = '2019-01-01'
    END_DATE = '2024-01-01'
    
    OUTPUT_FILE = os.path.join('data', 'raw', 'historical_prices.csv')
    
    fetch_data(TICKERS, START_DATE, END_DATE, OUTPUT_FILE)

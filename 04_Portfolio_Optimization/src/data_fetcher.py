import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date, end_date, output_path):
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    data = {}
    for ticker in tickers:
        try:
            # Using yf.Ticker().history() avoids all MultiIndex issues
            t = yf.Ticker(ticker)
            hist = t.history(start=start_date, end=end_date)
            
            # The history dataframe always has a 'Close' column
            if not hist.empty and 'Close' in hist.columns:
                data[ticker] = hist['Close']
            else:
                print(f"No data returned for {ticker}")
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            
    df = pd.DataFrame(data)
    
    # Strip timezone info from index to make it naive datetime
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
        
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

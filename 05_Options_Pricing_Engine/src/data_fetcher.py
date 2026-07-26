import yfinance as yf
import pandas as pd
import numpy as np
import os

def fetch_options_data(tickers=['AAPL', 'NVDA', 'JNJ']):
    """
    Fetches the real options chain and historical price data for multiple stocks.
    """
    print(f"--- Fetching data for basket: {tickers} ---")
    
    all_options = []
    all_hist_prices = []
    
    for ticker_symbol in tickers:
        print(f"\nProcessing {ticker_symbol}...")
        ticker = yf.Ticker(ticker_symbol)

        # --- 1. Get current stock price ---
        hist_recent = ticker.history(period='1d')
        spot_price = hist_recent['Close'].iloc[-1]
        print(f"Current Spot Price: ${spot_price:.2f}")

        # --- 2. Get 1 year of historical data to calculate volatility (sigma) ---
        hist_1yr = ticker.history(period='1y')
        
        # Strip timezone info
        if hist_1yr.index.tz is not None:
            hist_1yr.index = hist_1yr.index.tz_convert(None)
        
        hist_prices = hist_1yr[['Close']].copy()
        hist_prices['ticker'] = ticker_symbol
        all_hist_prices.append(hist_prices)
        
        # Calculate daily log returns and annualize the standard deviation to get sigma
        log_returns = np.log(hist_1yr['Close'] / hist_1yr['Close'].shift(1)).dropna()
        sigma = log_returns.std() * np.sqrt(252)  # Annualized volatility
        print(f"Calculated Historical Volatility (sigma): {sigma:.4f} ({sigma*100:.2f}%)")

        # --- 3. Fetch options chain ---
        expiry_dates = ticker.options
        
        if len(expiry_dates) < 3:
            print(f"Warning: Not enough expiries for {ticker_symbol}. Skipping.")
            continue

        # Skip very near-term (today/this week) expiries
        # Pick expiry dates 3–6 months out for realistic liquid contracts
        selected_expiries = expiry_dates[3:6] if len(expiry_dates) >= 6 else expiry_dates
        
        all_calls = []
        all_puts = []
        
        for expiry in selected_expiries:
            chain = ticker.option_chain(expiry)
            
            calls = chain.calls.copy()
            calls['expiry'] = expiry
            calls['option_type'] = 'call'
            
            puts = chain.puts.copy()
            puts['expiry'] = expiry
            puts['option_type'] = 'put'
            
            all_calls.append(calls)
            all_puts.append(puts)

        # Combine all expiry dates into one dataframe for this ticker
        ticker_options_df = pd.concat(all_calls + all_puts, ignore_index=True)

        cols_to_keep = ['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask',
                        'impliedVolatility', 'expiry', 'option_type']
        ticker_options_df = ticker_options_df[cols_to_keep]

        # Use mid-price (average of bid and ask) as the market price
        ticker_options_df['market_price'] = ticker_options_df.apply(
            lambda r: (r['bid'] + r['ask']) / 2 if (r['bid'] > 0 or r['ask'] > 0) else r['lastPrice'],
            axis=1
        )
        
        ticker_options_df = ticker_options_df[ticker_options_df['market_price'] > 0]
        ticker_options_df['spot_price'] = spot_price
        ticker_options_df['sigma'] = sigma
        ticker_options_df['ticker'] = ticker_symbol
        
        all_options.append(ticker_options_df)

    # --- Combine all tickers ---
    final_options_df = pd.concat(all_options, ignore_index=True)
    final_hist_prices = pd.concat(all_hist_prices)
    
    print(f"\nTotal liquid options contracts fetched across basket: {len(final_options_df)}")

    # --- 4. Save to CSV ---
    os.makedirs('data', exist_ok=True)
    final_options_df.to_csv('data/options_chain.csv', index=False)
    final_hist_prices.to_csv('data/historical_prices.csv')
    
    print(f"Options chain saved to data/options_chain.csv")
    print(f"Historical prices saved to data/historical_prices.csv")
    
    return final_options_df


if __name__ == '__main__':
    df = fetch_options_data(['AAPL', 'NVDA', 'JNJ'])
    print("\n--- Sample of fetched data ---")
    print(df[['ticker', 'strike', 'option_type', 'expiry', 'market_price', 'spot_price', 'sigma']].head(5).to_string())


import yfinance as yf
import pandas as pd
import numpy as np
import os

def fetch_options_data(ticker_symbol='AAPL'):
    """
    Fetches the real options chain and historical price data for a given stock.
    
    Steps:
    1. Get current stock price
    2. Fetch historical prices (needed to calculate sigma - historical volatility)
    3. Fetch live options chain (the list of all available contracts)
    4. Save everything to CSV for offline use
    """
    print(f"--- Fetching data for {ticker_symbol} ---")
    ticker = yf.Ticker(ticker_symbol)

    # --- 1. Get current stock price ---
    hist_recent = ticker.history(period='1d')
    spot_price = hist_recent['Close'].iloc[-1]
    print(f"Current {ticker_symbol} Spot Price: ${spot_price:.2f}")

    # --- 2. Get 1 year of historical data to calculate volatility (sigma) ---
    print("Fetching 1 year of historical price data for volatility calculation...")
    hist_1yr = ticker.history(period='1y')
    
    # Strip timezone info
    if hist_1yr.index.tz is not None:
        hist_1yr.index = hist_1yr.index.tz_convert(None)
    
    # Calculate daily log returns and annualize the standard deviation to get sigma
    log_returns = np.log(hist_1yr['Close'] / hist_1yr['Close'].shift(1)).dropna()
    sigma = log_returns.std() * np.sqrt(252)  # Annualized volatility
    print(f"Calculated Historical Volatility (sigma): {sigma:.4f} ({sigma*100:.2f}%)")

    # --- 3. Fetch options chain ---
    # Get available expiry dates
    expiry_dates = ticker.options
    print(f"\nAvailable expiry dates: {expiry_dates[:5]}...")  # Show first 5

    # Skip very near-term (today/this week) expiries — they often have no bid/ask
    # Pick expiry dates 3–6 months out for realistic liquid contracts
    selected_expiries = expiry_dates[3:6] if len(expiry_dates) >= 6 else expiry_dates
    
    all_calls = []
    all_puts = []
    
    for expiry in selected_expiries:
        print(f"  Fetching options chain for expiry: {expiry}...")
        chain = ticker.option_chain(expiry)
        
        # Tag each row with its expiry date
        calls = chain.calls.copy()
        calls['expiry'] = expiry
        calls['option_type'] = 'call'
        
        puts = chain.puts.copy()
        puts['expiry'] = expiry
        puts['option_type'] = 'put'
        
        all_calls.append(calls)
        all_puts.append(puts)

    # Combine all expiry dates into one dataframe
    options_df = pd.concat(all_calls + all_puts, ignore_index=True)

    # Keep only the columns we need for pricing
    cols_to_keep = ['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask',
                    'impliedVolatility', 'expiry', 'option_type']
    options_df = options_df[cols_to_keep]

    # Use mid-price (average of bid and ask) as the market price
    # If both bid and ask are zero (market closed or illiquid), fall back to lastPrice
    options_df['market_price'] = options_df.apply(
        lambda r: (r['bid'] + r['ask']) / 2 if (r['bid'] > 0 or r['ask'] > 0) else r['lastPrice'],
        axis=1
    )
    
    # Drop rows where we have no price at all
    options_df = options_df[options_df['market_price'] > 0]

    # Add the spot price and sigma as columns for easy access downstream
    options_df['spot_price'] = spot_price
    options_df['sigma'] = sigma

    print(f"\nTotal liquid options contracts fetched: {len(options_df)}")

    # --- 4. Save to CSV ---
    os.makedirs('data', exist_ok=True)
    options_df.to_csv('data/options_chain.csv', index=False)
    hist_1yr[['Close']].to_csv('data/historical_prices.csv')
    
    print(f"Options chain saved to data/options_chain.csv")
    print(f"Historical prices saved to data/historical_prices.csv")
    
    return options_df, sigma, spot_price


if __name__ == '__main__':
    df, sigma, spot = fetch_options_data('AAPL')
    print("\n--- Sample of fetched data ---")
    print(df[['strike', 'option_type', 'expiry', 'market_price', 'spot_price', 'sigma']].head(10).to_string())

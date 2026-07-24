import numpy as np
from scipy.stats import norm
import pandas as pd
import os


def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the four key Greeks for an option using Black-Scholes.

    What are the Greeks?
    - Delta: Sensitivity of option price to a $1 move in the stock price
    - Gamma: Sensitivity of Delta itself to a $1 move in the stock price
    - Theta: How much value the option loses per day (time decay)
    - Vega:  Sensitivity of option price to a 1% move in volatility

    Parameters:
    -----------
    S     : float - Current stock price
    K     : float - Strike price
    T     : float - Time to expiry in years
    r     : float - Annual risk-free rate
    sigma : float - Annualized volatility

    Returns:
    --------
    dict with keys: delta, gamma, theta, vega
    """
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # --- Delta ---
    # Call Delta is always between 0 and 1 (how much the call moves per $1 move in stock)
    # Put Delta is always between -1 and 0
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # --- Gamma ---
    # Same for calls and puts. Measures the curvature of the option price.
    # High Gamma = Delta changes rapidly (options near expiry, near the strike)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # --- Theta ---
    # Time decay per CALENDAR day (divided by 365)
    # Almost always negative — options lose value as time passes
    theta_per_year = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
    )
    if option_type == 'put':
        theta_per_year += r * K * np.exp(-r * T)
    theta = theta_per_year / 365  # Convert to daily

    # --- Vega ---
    # Sensitivity to a 1% change in volatility (divided by 100 for 1% unit)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        'delta': round(delta, 6),
        'gamma': round(gamma, 6),
        'theta': round(theta, 6),   # $ per day
        'vega':  round(vega,  6),   # $ per 1% vol change
    }


def greeks_for_chain(options_df, r=0.045):
    """Computes all four Greeks for every contract in the options chain."""
    today = pd.Timestamp.today().normalize()

    records = []
    for _, row in options_df.iterrows():
        T = (pd.Timestamp(row['expiry']) - today).days / 365.0
        greeks = calculate_greeks(
            S=row['spot_price'],
            K=row['strike'],
            T=T,
            r=r,
            sigma=row['sigma'],
            option_type=row['option_type']
        )
        records.append({**row.to_dict(), **greeks})

    return pd.DataFrame(records)


def print_greeks_summary(df):
    """Prints a clean breakdown of Greeks for calls and puts."""
    print("\n" + "="*60)
    print("GREEKS SUMMARY (Sample — Near-The-Money Contracts)")
    print("="*60)

    spot = df['spot_price'].iloc[0]
    # Filter to contracts within 10% of spot price (near-the-money)
    ntm = df[(df['strike'] >= spot * 0.92) & (df['strike'] <= spot * 1.08)]

    for opt_type in ['call', 'put']:
        subset = ntm[ntm['option_type'] == opt_type].sort_values('strike')
        print(f"\n--- {opt_type.upper()}S (Near-the-Money) ---")
        print(subset[['strike', 'expiry', 'market_price', 'delta', 'gamma', 'theta', 'vega']].to_string(index=False))

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    calls_ntm = ntm[ntm['option_type'] == 'call'].copy()
    calls_ntm['dist_to_spot'] = (calls_ntm['strike'] - spot).abs()
    atm_call = calls_ntm.sort_values('dist_to_spot').iloc[0]
    print(f"\nFor the ~At-The-Money Call (Strike ${atm_call['strike']}):")
    print(f"  Delta = {atm_call['delta']:.4f}  => Option moves ${atm_call['delta']:.2f} for every $1 move in AAPL")
    print(f"  Gamma = {atm_call['gamma']:.4f}  => Delta changes by {atm_call['gamma']:.4f} for every $1 move in AAPL")
    print(f"  Theta = {atm_call['theta']:.4f}  => Option loses ${abs(atm_call['theta']):.4f} in value per day")
    print(f"  Vega  = {atm_call['vega']:.4f}   => Option gains ${atm_call['vega']:.4f} for every 1% rise in volatility")
    print("="*60)


if __name__ == '__main__':
    df = pd.read_csv(os.path.join('data', 'options_chain.csv'))
    results = greeks_for_chain(df, r=0.045)
    results.to_csv(os.path.join('data', 'greeks_results.csv'), index=False)
    print("Greeks saved to data/greeks_results.csv")
    print_greeks_summary(results)

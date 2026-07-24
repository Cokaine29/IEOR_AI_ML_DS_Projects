import numpy as np
from scipy.stats import norm
import pandas as pd
import os


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the theoretical fair price of a European option
    using the Black-Scholes closed-form formula.

    Parameters:
    -----------
    S     : float  - Current stock price (Spot price)
    K     : float  - Strike price of the option contract
    T     : float  - Time to expiry in years (e.g., 30 days = 30/365)
    r     : float  - Annual risk-free interest rate (e.g., 0.045 for 4.5%)
    sigma : float  - Annualized historical volatility of the stock
    option_type : str - 'call' or 'put'

    Returns:
    --------
    float : The Black-Scholes theoretical price of the option
    """
    if T <= 0:
        # Option has expired — return intrinsic value only
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    # The two intermediate values (d1 and d2) used in the formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def price_options_chain(options_df, r=0.045):
    """
    Applies Black-Scholes pricing to every row in the options DataFrame.

    r = 0.045 approximates the current US 10-Year Treasury yield (~4.5%)
    """
    print(f"--- Running Black-Scholes Pricing on {len(options_df)} contracts ---")
    print(f"Risk-Free Rate (r): {r*100:.2f}%")

    today = pd.Timestamp.today().normalize()

    def price_row(row):
        expiry_date = pd.Timestamp(row['expiry'])
        T = (expiry_date - today).days / 365.0  # Time to expiry in years
        return black_scholes_price(
            S=row['spot_price'],
            K=row['strike'],
            T=T,
            r=r,
            sigma=row['sigma'],
            option_type=row['option_type']
        )

    options_df = options_df.copy()
    options_df['bs_price'] = options_df.apply(price_row, axis=1)

    # --- The Core Analysis: Mispricing ---
    # How far is our theoretical price from what the market is actually charging?
    options_df['mispricing'] = options_df['bs_price'] - options_df['market_price']
    options_df['mispricing_pct'] = (options_df['mispricing'] / options_df['market_price']) * 100

    return options_df


def print_summary(options_df):
    """Prints a clean summary of the mispricing analysis."""
    print("\n" + "="*55)
    print("BLACK-SCHOLES MISPRICING ANALYSIS")
    print("="*55)

    mae = options_df['mispricing'].abs().mean()
    mae_pct = options_df['mispricing_pct'].abs().mean()

    print(f"\nTotal contracts analyzed: {len(options_df)}")
    print(f"Mean Absolute Error (MAE):     ${mae:.4f}")
    print(f"Mean Absolute Error (%):       {mae_pct:.2f}%")

    print("\n--- By Option Type ---")
    for opt_type in ['call', 'put']:
        subset = options_df[options_df['option_type'] == opt_type]
        print(f"  {opt_type.capitalize()}s  |  MAE: ${subset['mispricing'].abs().mean():.4f}  |  MAE%: {subset['mispricing_pct'].abs().mean():.2f}%")

    print("\n--- Most Underpriced by Market (BS says BUY) ---")
    underpriced = options_df.nsmallest(5, 'mispricing')[['strike', 'option_type', 'expiry', 'market_price', 'bs_price', 'mispricing_pct']]
    print(underpriced.to_string(index=False))

    print("\n--- Most Overpriced by Market (BS says SELL) ---")
    overpriced = options_df.nlargest(5, 'mispricing')[['strike', 'option_type', 'expiry', 'market_price', 'bs_price', 'mispricing_pct']]
    print(overpriced.to_string(index=False))
    print("="*55)


if __name__ == '__main__':
    # Load the data fetched in Step 1
    options_df = pd.read_csv(os.path.join('data', 'options_chain.csv'))

    # Run Black-Scholes on all 279 contracts
    results_df = price_options_chain(options_df, r=0.045)

    # Save enriched results
    results_df.to_csv(os.path.join('data', 'bs_pricing_results.csv'), index=False)
    print("Results saved to data/bs_pricing_results.csv")

    # Print the mispricing summary
    print_summary(results_df)

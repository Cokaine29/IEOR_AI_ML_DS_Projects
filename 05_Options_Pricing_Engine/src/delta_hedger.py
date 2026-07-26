import numpy as np
import pandas as pd
import os

# Reuse our existing modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from black_scholes import black_scholes_price
from greeks import calculate_greeks


def simulate_delta_hedging(S0, K, T_days, r, sigma, n_paths=500, seed=42):
    """
    Simulates the P&L of two strategies over the life of an option:

    1. UNHEDGED: Simply sell a call option and do nothing — your P&L is just
       the premium received minus whatever the option is worth at expiry.

    2. DELTA-HEDGED: Sell a call option AND dynamically buy/sell shares of
       the stock every day to maintain a "Delta-neutral" position
       (i.e., your portfolio barely moves even if the stock price moves).

    The goal is to show that the hedged P&L is FAR less volatile than
    the unhedged one — proving that Delta-Hedging reduces risk.

    Parameters:
    -----------
    S0      : float - Initial stock price
    K       : float - Strike price of the option we're selling
    T_days  : int   - Number of days until expiry
    r       : float - Annual risk-free rate
    sigma   : float - Annualized volatility
    n_paths : int   - How many different future price paths to simulate
    seed    : int   - Random seed for reproducibility

    Returns:
    --------
    tuple: (hedged_pnl_series, unhedged_pnl_series, summary_stats)
    """
    np.random.seed(seed)
    T = T_days / 365.0
    dt = 1 / 365.0  # One day at a time

    # The option premium we collect upfront for selling the call
    option_premium = black_scholes_price(S0, K, T, r, sigma, option_type='call')
    print(f"\nOption Premium Collected (Upfront): ${option_premium:.4f}")

    hedged_final_pnl = []
    unhedged_final_pnl = []

    for path_idx in range(n_paths):
        # --- Simulate a stock price path using Geometric Brownian Motion ---
        daily_returns = np.exp(
            (r - 0.5 * sigma**2) * dt +
            sigma * np.sqrt(dt) * np.random.randn(T_days)
        )
        # Build the price path day by day
        stock_path = np.zeros(T_days + 1)
        stock_path[0] = S0
        for t in range(1, T_days + 1):
            stock_path[t] = stock_path[t-1] * daily_returns[t-1]

        S_final = stock_path[-1]

        # --- UNHEDGED P&L ---
        # We collected the premium, and now we must pay the option's final value
        call_payoff = max(S_final - K, 0)
        unhedged_pnl = option_premium - call_payoff
        unhedged_final_pnl.append(unhedged_pnl)

        # --- DELTA-HEDGED P&L ---
        # Start by collecting the premium
        cash = option_premium
        shares_held = 0  # Initially hold no shares

        for t in range(T_days):
            S_t = stock_path[t]
            T_remaining = (T_days - t) / 365.0

            if T_remaining <= 0:
                break

            # What is Delta right now? (How many shares do I need to hold?)
            greeks = calculate_greeks(S_t, K, T_remaining, r, sigma, option_type='call')
            target_delta = greeks['delta']

            # Rebalance: buy or sell shares to match target Delta
            shares_to_trade = target_delta - shares_held
            cash -= shares_to_trade * S_t  # Buying costs cash; selling gains cash
            shares_held = target_delta

            # Accrue interest on cash position overnight
            cash *= np.exp(r * dt)

        # At expiry: close the stock position and settle the option
        cash += shares_held * S_final          # Sell all shares
        cash -= max(S_final - K, 0)            # Pay the option buyer if they exercise
        hedged_final_pnl.append(cash)

    return np.array(hedged_final_pnl), np.array(unhedged_final_pnl)


def print_summary(results_dict):
    """Prints a comparative summary across all simulated tickers."""
    print("\n" + "="*80)
    print("MULTI-REGIME DELTA-HEDGING SIMULATION RESULTS")
    print("="*80)
    
    print(f"{'Ticker':<10} {'Unhedged Risk':>15} {'Hedged Risk':>15} {'Vol Reduction %':>20}")
    print("-" * 80)
    
    for ticker, stats in results_dict.items():
        u_std = stats['unhedged_std']
        h_std = stats['hedged_std']
        reduction = (1 - h_std / u_std) * 100
        print(f"{ticker:<10} ${u_std:>14.4f} ${h_std:>14.4f} {reduction:>19.2f}%")
    
    print("="*80)
    print("\nConclusion: Delta-Hedging significantly reduces P&L risk across ALL regimes.")
    print("Notice how high-volatility names (NVDA) may show different hedging efficiencies")
    print("than low-volatility names (JNJ) due to overnight jump risks (Gamma risk).")


def run_all_simulations():
    try:
        options_df = pd.read_csv(os.path.join('data', 'options_chain.csv'))
    except FileNotFoundError:
        print("Run data_fetcher.py first to get the options chain.")
        return

    print("="*80)
    print("DELTA-HEDGING SIMULATION (MULTI-REGIME)")
    print(f"Simulating 500 paths of Geometric Brownian Motion per ticker...")
    print("="*80)

    tickers = options_df['ticker'].unique()
    all_results_df = []
    summary_dict = {}

    for ticker in tickers:
        subset = options_df[(options_df['ticker'] == ticker) & (options_df['option_type'] == 'call')].copy()
        if subset.empty: continue
        
        # Find ATM option
        spot = subset['spot_price'].iloc[0]
        sigma = subset['sigma'].iloc[0]
        subset['dist'] = (subset['strike'] - spot).abs()
        atm_opt = subset.sort_values('dist').iloc[0]
        
        S0 = spot
        K = atm_opt['strike']
        
        # Rough T approx from first expiry
        today = pd.Timestamp.today().normalize()
        expiry = pd.Timestamp(atm_opt['expiry'])
        T_days = (expiry - today).days
        if T_days <= 0: T_days = 7
        
        r = 0.045
        
        print(f"\n[{ticker}] S0=${S0:.2f} | K=${K} | T={T_days} days | sigma={sigma*100:.1f}%")
        
        hedged, unhedged = simulate_delta_hedging(S0, K, T_days, r, sigma, n_paths=500)
        
        df = pd.DataFrame({'ticker': ticker, 'hedged_pnl': hedged, 'unhedged_pnl': unhedged})
        all_results_df.append(df)
        
        summary_dict[ticker] = {
            'hedged_std': hedged.std(),
            'unhedged_std': unhedged.std()
        }

    if all_results_df:
        final_df = pd.concat(all_results_df, ignore_index=True)
        final_df.to_csv(os.path.join('data', 'hedging_simulation.csv'), index=False)
        print("\nSimulation results saved to data/hedging_simulation.csv")
        
        print_summary(summary_dict)

if __name__ == '__main__':
    run_all_simulations()

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


def print_summary(hedged_pnl, unhedged_pnl):
    """Prints the key comparison statistics."""
    print("\n" + "="*55)
    print("DELTA-HEDGING SIMULATION RESULTS")
    print("="*55)

    metrics = {
        'Mean P&L ($)':          (hedged_pnl.mean(),    unhedged_pnl.mean()),
        'Std Dev / Risk ($)':    (hedged_pnl.std(),     unhedged_pnl.std()),
        'Best Case P&L ($)':     (hedged_pnl.max(),     unhedged_pnl.max()),
        'Worst Case P&L ($)':    (hedged_pnl.min(),     unhedged_pnl.min()),
    }

    print(f"\n{'Metric':<25} {'Delta-Hedged':>15} {'Unhedged':>15}")
    print("-" * 55)
    for label, (h, u) in metrics.items():
        print(f"{label:<25} {h:>15.4f} {u:>15.4f}")

    # The key resume number
    variance_reduction = (1 - hedged_pnl.std() / unhedged_pnl.std()) * 100
    print(f"\n{'P&L Volatility Reduction:':<25} {variance_reduction:>14.2f}%")
    print("="*55)
    print("\nConclusion: Delta-Hedging significantly reduces P&L risk,")
    print("proving that the Black-Scholes Greeks are actionable and effective.")

    return variance_reduction


if __name__ == '__main__':
    # Use real AAPL values from our data fetcher
    S0    = 321.66   # Spot price
    K     = 322.50   # Near-ATM strike
    T_days = 7       # Days to expiry (July 31 option)
    r     = 0.045
    sigma = 0.2458

    print("="*55)
    print("DELTA-HEDGING SIMULATION")
    print(f"AAPL | Strike=${K} | T={T_days} days | sigma={sigma*100:.1f}%")
    print(f"Simulating {500} paths of Geometric Brownian Motion...")
    print("="*55)

    hedged, unhedged = simulate_delta_hedging(S0, K, T_days, r, sigma, n_paths=500)

    # Save results
    results_df = pd.DataFrame({'hedged_pnl': hedged, 'unhedged_pnl': unhedged})
    results_df.to_csv(os.path.join('data', 'hedging_simulation.csv'), index=False)
    print("Simulation results saved to data/hedging_simulation.csv")

    variance_reduction = print_summary(hedged, unhedged)

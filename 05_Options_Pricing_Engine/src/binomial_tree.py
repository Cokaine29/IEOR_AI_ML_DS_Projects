import numpy as np
import pandas as pd
import os


def binomial_tree_price(S, K, T, r, sigma, N, option_type='call', american=False):
    """
    Prices a European or American option using the Cox-Ross-Rubinstein (CRR)
    Binomial Tree model.

    The core idea:
    - Divide the time to expiry T into N small steps
    - At each step, the stock price can go UP by factor u, or DOWN by factor d
    - These up/down factors are derived from the volatility (sigma)
    - Work BACKWARDS from expiry to today to find the fair price today

    Parameters:
    -----------
    S       : float - Current stock price
    K       : float - Strike price
    T       : float - Time to expiry in years
    r       : float - Annual risk-free rate
    sigma   : float - Annualized historical volatility
    N       : int   - Number of time steps in the tree
    option_type : str - 'call' or 'put'
    american    : bool - If True, allow early exercise (American option)

    Returns:
    --------
    float : The Binomial Tree theoretical price of the option
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    dt = T / N  # Length of each time step in years

    # CRR up and down factors — derived from volatility
    u = np.exp(sigma * np.sqrt(dt))   # Up factor
    d = 1 / u                          # Down factor (symmetric)

    # Risk-neutral probability of going up
    # This is the key insight from the course: we don't use real-world probabilities
    # We use risk-neutral probabilities to price under no-arbitrage
    p = (np.exp(r * dt) - d) / (u - d)

    # Discount factor per step
    discount = np.exp(-r * dt)

    # --- Build the stock price tree at expiry (leaf nodes only) ---
    # At step N, stock has gone up j times and down (N-j) times
    # Stock price at leaf j = S * u^j * d^(N-j)
    j = np.arange(N + 1)
    stock_at_expiry = S * (u ** j) * (d ** (N - j))

    # --- Calculate option payoff at expiry ---
    if option_type == 'call':
        option_values = np.maximum(stock_at_expiry - K, 0)
    else:
        option_values = np.maximum(K - stock_at_expiry, 0)

    # --- Work backwards through the tree ---
    for i in range(N - 1, -1, -1):
        # The option value at each node = discounted expected value
        option_values = discount * (p * option_values[1:] + (1 - p) * option_values[:-1])

        if american:
            # For American options, check if early exercise is better
            stock_prices = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            if option_type == 'call':
                intrinsic = np.maximum(stock_prices - K, 0)
            else:
                intrinsic = np.maximum(K - stock_prices, 0)
            option_values = np.maximum(option_values, intrinsic)

    return option_values[0]


def convergence_analysis(S, K, T, r, sigma, option_type='call', bs_price=None):
    """
    Demonstrates the key mathematical insight:
    As N (number of steps) increases, the Binomial Tree price converges
    to the Black-Scholes closed-form price.

    This is a direct proof from IE 612 theory.
    """
    step_sizes = [1, 5, 10, 25, 50, 100, 200, 500]
    prices = []

    for N in step_sizes:
        price = binomial_tree_price(S, K, T, r, sigma, N, option_type=option_type)
        prices.append({'N_steps': N, 'binomial_price': round(price, 6)})

    results_df = pd.DataFrame(prices)
    if bs_price is not None:
        results_df['bs_price'] = round(bs_price, 6)
        results_df['diff_from_bs'] = (results_df['binomial_price'] - bs_price).abs().round(6)

    return results_df


def run_convergence_demo():
    """
    Run the convergence demo on a sample at-the-money AAPL call option.
    """
    from black_scholes import black_scholes_price

    # Use real AAPL values from our data fetcher
    S = 321.66      # Spot price
    K = 320.0       # Strike (near at-the-money)
    T = 7 / 365    # 7 days to expiry (approximately July 31 from today)
    r = 0.045       # Risk-free rate
    sigma = 0.2458  # Historical volatility

    bs = black_scholes_price(S, K, T, r, sigma, option_type='call')

    print("="*60)
    print("BINOMIAL TREE CONVERGENCE TO BLACK-SCHOLES")
    print(f"AAPL At-The-Money Call | Strike=${K} | T={T*365:.0f} days")
    print(f"Black-Scholes Price: ${bs:.4f}")
    print("="*60)

    results = convergence_analysis(S, K, T, r, sigma, option_type='call', bs_price=bs)
    print(results.to_string(index=False))

    # Find the N where convergence is within $0.01 of Black-Scholes
    converged = results[results['diff_from_bs'] < 0.01]
    if not converged.empty:
        convergence_N = converged.iloc[0]['N_steps']
        print(f"\nConvergence within $0.01 of Black-Scholes achieved at N = {int(convergence_N)} steps")

    # Save results
    results.to_csv(os.path.join('data', 'binomial_convergence.csv'), index=False)
    print("Convergence results saved to data/binomial_convergence.csv")
    print("="*60)

    return results, bs


if __name__ == '__main__':
    run_convergence_demo()

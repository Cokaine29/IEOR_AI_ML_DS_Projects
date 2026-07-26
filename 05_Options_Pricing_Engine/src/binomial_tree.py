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
    Run the convergence demo on a sample at-the-money call option for each ticker.
    """
    from black_scholes import black_scholes_price

    try:
        options_df = pd.read_csv(os.path.join('data', 'options_chain.csv'))
    except FileNotFoundError:
        print("Run data_fetcher.py first to get the options chain.")
        return

    print("="*70)
    print("BINOMIAL TREE CONVERGENCE TO BLACK-SCHOLES (MULTI-REGIME)")
    print("="*70)

    all_results = []
    
    tickers = options_df['ticker'].unique()
    for ticker in tickers:
        subset = options_df[(options_df['ticker'] == ticker) & (options_df['option_type'] == 'call')].copy()
        if subset.empty: continue
        
        # Find ATM option
        spot = subset['spot_price'].iloc[0]
        sigma = subset['sigma'].iloc[0]
        subset['dist'] = (subset['strike'] - spot).abs()
        atm_opt = subset.sort_values('dist').iloc[0]
        
        S = spot
        K = atm_opt['strike']
        # Rough T approx from first expiry
        today = pd.Timestamp.today().normalize()
        expiry = pd.Timestamp(atm_opt['expiry'])
        T_days = (expiry - today).days
        if T_days <= 0: T_days = 7
        T = T_days / 365.0
        r = 0.045
        
        bs = black_scholes_price(S, K, T, r, sigma, option_type='call')
        
        print(f"\n--- {ticker} ATM Call | Strike=${K} | Spot=${S:.2f} | sigma={sigma*100:.1f}% ---")
        print(f"Black-Scholes Price: ${bs:.4f}")
        
        results = convergence_analysis(S, K, T, r, sigma, option_type='call', bs_price=bs)
        results['ticker'] = ticker
        all_results.append(results)
        
        print(results[['N_steps', 'binomial_price', 'diff_from_bs']].to_string(index=False))
        
        # Find convergence
        converged = results[results['diff_from_bs'] < 0.01]
        if not converged.empty:
            convergence_N = converged.iloc[0]['N_steps']
            print(f"-> Convergence within $0.01 achieved at N = {int(convergence_N)} steps")
        else:
            print(f"-> Did not converge within $0.01 in max steps.")

    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv(os.path.join('data', 'binomial_convergence.csv'), index=False)
        print("\nConvergence results saved to data/binomial_convergence.csv")
        print("="*70)
        return final_results
    
    return None


if __name__ == '__main__':
    run_convergence_demo()

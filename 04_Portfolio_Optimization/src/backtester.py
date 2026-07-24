import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from markowitz_gurobi import optimize_portfolio
from baseline import get_equal_weights

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib could not be imported due to local environment corruption. Plotting will be skipped, but CSV results will be saved.")
    MATPLOTLIB_AVAILABLE = False

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates annualized Sharpe Ratio"""
    if returns.std() == 0:
        return 0
    return np.sqrt(252) * (returns.mean() - risk_free_rate) / returns.std()

def run_backtest():
    print("Loading historical data...")
    prices = pd.read_csv(os.path.join('data', 'raw', 'historical_prices.csv'), index_col=0, parse_dates=True)
    daily_returns = prices.pct_change().dropna()
    
    # Backtest parameters
    lookback_months = 12
    hold_months = 1
    
    start_date = daily_returns.index[0] + relativedelta(months=lookback_months)
    end_date = daily_returns.index[-1]
    
    current_date = start_date
    
    gurobi_portfolio_returns = []
    baseline_portfolio_returns = []
    dates = []
    
    print(f"Running backtest from {start_date.date()} to {end_date.date()}...")
    
    while current_date < end_date:
        # Define lookback window
        window_start = current_date - relativedelta(months=lookback_months)
        window_data = daily_returns.loc[window_start:current_date]
        
        if len(window_data) < 20:
            current_date += relativedelta(months=hold_months)
            continue
            
        # 1. Calculate Expected Returns and Covariance Matrix (Annualized)
        exp_returns_annual = window_data.mean() * 252
        cov_matrix_annual = window_data.cov() * 252
        
        # We set a target return (e.g., slightly above average of all stocks)
        target_ret = np.percentile(exp_returns_annual, 60) 
        
        # 2. Get optimal weights (Gurobi) and baseline weights
        gurobi_weights = optimize_portfolio(exp_returns_annual, cov_matrix_annual, target_return=target_ret)
        baseline_weights = get_equal_weights(prices.columns)
        
        # 3. Simulate hold period (1 month)
        hold_end = current_date + relativedelta(months=hold_months)
        hold_data = daily_returns.loc[current_date:hold_end].iloc[1:] # Exclude current_date itself to prevent overlap
        
        if not hold_data.empty:
            # Calculate daily returns of the portfolios
            gurobi_daily_rets = hold_data.dot(gurobi_weights)
            baseline_daily_rets = hold_data.dot(baseline_weights)
            
            gurobi_portfolio_returns.extend(gurobi_daily_rets)
            baseline_portfolio_returns.extend(baseline_daily_rets)
            dates.extend(hold_data.index)
            
        current_date = hold_end
        
    print("Backtest Complete!")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Gurobi_Optimized': gurobi_portfolio_returns,
        'Equal_Weight': baseline_portfolio_returns
    }, index=dates)
    
    # Calculate Cumulative Returns
    cumulative_returns = (1 + results_df).cumprod()
    
    # Metrics
    gurobi_sharpe = calculate_sharpe_ratio(results_df['Gurobi_Optimized'])
    baseline_sharpe = calculate_sharpe_ratio(results_df['Equal_Weight'])
    
    gurobi_total_return = cumulative_returns['Gurobi_Optimized'].iloc[-1] - 1
    baseline_total_return = cumulative_returns['Equal_Weight'].iloc[-1] - 1
    
    gurobi_vol = results_df['Gurobi_Optimized'].std() * np.sqrt(252)
    baseline_vol = results_df['Equal_Weight'].std() * np.sqrt(252)
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS (Out-of-Sample)")
    print("="*40)
    print(f"Gurobi Portfolio:")
    print(f"  Total Return: {gurobi_total_return:.2%}")
    print(f"  Volatility:   {gurobi_vol:.2%}")
    print(f"  Sharpe Ratio: {gurobi_sharpe:.2f}")
    print(f"\nEqual-Weight Baseline:")
    print(f"  Total Return: {baseline_total_return:.2%}")
    print(f"  Volatility:   {baseline_vol:.2%}")
    print(f"  Sharpe Ratio: {baseline_sharpe:.2f}")
    print("="*40)
    
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    
    # Always save raw results to CSV
    csv_path = os.path.join('data', 'processed', 'backtest_results.csv')
    cumulative_returns.to_csv(csv_path)
    print(f"\nResults saved to {csv_path}")
    
    # Plotting
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns['Gurobi_Optimized'], label=f'Gurobi Markowitz (Sharpe: {gurobi_sharpe:.2f})', linewidth=2)
        plt.plot(cumulative_returns['Equal_Weight'], label=f'Equal Weight (Sharpe: {baseline_sharpe:.2f})', linewidth=2, linestyle='--')
        plt.title('Out-of-Sample Cumulative Returns: Gurobi vs Baseline')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join('data', 'processed', 'equity_curve.png')
        plt.savefig(plot_path)
        print(f"Equity curve saved to {plot_path}")

if __name__ == "__main__":
    run_backtest()

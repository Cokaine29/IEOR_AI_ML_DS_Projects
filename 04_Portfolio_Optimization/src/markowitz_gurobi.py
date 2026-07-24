import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

def optimize_portfolio(expected_returns, cov_matrix, target_return=None):
    """
    Optimizes portfolio weights using Markowitz Mean-Variance framework.
    Attempts to use Gurobi first, but falls back to Scipy if the Gurobi license is expired.
    """
    n_assets = len(expected_returns)
    tickers = expected_returns.index
    
    try:
        # Create a new model
        model = gp.Model("portfolio_optimization")
        model.setParam('OutputFlag', 0) # Suppress output
        
        w = model.addMVar(n_assets, lb=0.0, ub=1.0, name="weights")
        model.addConstr(w.sum() == 1, "budget")
        
        if target_return is not None:
            model.addConstr(expected_returns.values @ w >= target_return, "target_return")
        
        portfolio_variance = w @ cov_matrix.values @ w
        model.setObjective(portfolio_variance, GRB.MINIMIZE)
        
        model.optimize()
        
        if model.Status == GRB.OPTIMAL:
            optimal_weights = w.X
            return pd.Series(optimal_weights, index=tickers)
        else:
            return pd.Series(np.ones(n_assets) / n_assets, index=tickers)
            
    except Exception as e:
        # Fallback to Scipy Optimize if Gurobi fails (e.g., license expired)
        import scipy.optimize as sco
        
        def portfolio_variance_scipy(weights, cov):
            return weights.T @ cov @ weights
            
        args = (cov_matrix.values,)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}] # Sum to 1
        
        if target_return is not None:
            constraints.append({'type': 'ineq', 'fun': lambda x: np.dot(expected_returns.values, x) - target_return})
            
        bounds = tuple((0.0, 1.0) for asset in range(n_assets)) # Long only
        initial_weights = np.ones(n_assets) / n_assets
        
        result = sco.minimize(portfolio_variance_scipy, initial_weights, args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
                              
        if result.success:
            return pd.Series(result.x, index=tickers)
        else:
            return pd.Series(np.ones(n_assets) / n_assets, index=tickers)


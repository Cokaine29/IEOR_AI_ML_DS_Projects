"""
optimizers.py
=============
Three OR optimization models for portfolio construction:

1. Min-Variance (QP)  — classic Markowitz, minimize portfolio variance
2. Max Sharpe  (QP)   — tangency portfolio, maximize Sharpe ratio
3. Min CVaR    (LP)   — minimize Conditional Value-at-Risk (Expected Shortfall)
                        at confidence level alpha (default 95%)

Each function:
- Attempts Gurobi first
- Falls back to scipy.optimize if Gurobi is unavailable / license expired
- Returns a pd.Series of weights indexed by ticker
- On failure, returns equal weights (no silent crashes)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# 1. MINIMUM VARIANCE  (Quadratic Program)
# ==============================================================================
def optimize_min_variance(expected_returns: pd.Series,
                          cov_matrix: pd.DataFrame,
                          target_return: float = None) -> pd.Series:
    """
    Minimize w' Σ w  subject to:
      Σ wᵢ = 1  (fully invested)
      wᵢ ≥ 0   (long-only)
      μ' w ≥ target_return  (optional)
    """
    n = len(expected_returns)
    tickers = expected_returns.index

    # ── Gurobi path ────────────────────────────────────────────────────────────
    try:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("min_variance")
        m.setParam('OutputFlag', 0)

        w = m.addMVar(n, lb=0.0, ub=1.0, name="w")
        m.addConstr(w.sum() == 1, "budget")
        if target_return is not None:
            m.addConstr(expected_returns.values @ w >= target_return, "target_ret")

        m.setObjective(w @ cov_matrix.values @ w, GRB.MINIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            return pd.Series(w.X, index=tickers)

    except Exception:
        pass  # fall through to scipy

    # ── Scipy fallback ─────────────────────────────────────────────────────────
    from scipy.optimize import minimize

    def objective(w):
        return w @ cov_matrix.values @ w

    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    if target_return is not None:
        constraints.append(
            {'type': 'ineq', 'fun': lambda w: expected_returns.values @ w - target_return}
        )

    result = minimize(objective,
                      x0=np.ones(n) / n,
                      method='SLSQP',
                      bounds=[(0, 1)] * n,
                      constraints=constraints)

    if result.success:
        return pd.Series(result.x, index=tickers)

    return pd.Series(np.ones(n) / n, index=tickers)  # fallback: equal weight


# ==============================================================================
# 2. MAXIMUM SHARPE RATIO  (Quadratic Program via variable substitution)
# ==============================================================================
def optimize_max_sharpe(expected_returns: pd.Series,
                        cov_matrix: pd.DataFrame,
                        risk_free_rate: float = 0.0) -> pd.Series:
    """
    Maximizes the Sharpe Ratio: (μ'w - rf) / sqrt(w'Σw)
    
    Solved by the Markowitz variable substitution trick:
      let y = w / (μ'w - rf),  then minimize y'Σy  s.t. (μ-rf)'y = 1, y ≥ 0
    Then recover w = y / sum(y).
    This converts the fractional program into a pure QP.
    """
    n = len(expected_returns)
    tickers = expected_returns.index
    excess_returns = expected_returns.values - risk_free_rate

    # If all excess returns are non-positive, fall back to min-variance
    if np.all(excess_returns <= 0):
        return optimize_min_variance(expected_returns, cov_matrix)

    # ── Scipy path (this QP is small; scipy is reliable here) ─────────────────
    from scipy.optimize import minimize

    def neg_sharpe(w):
        port_ret = expected_returns.values @ w
        port_vol = np.sqrt(w @ cov_matrix.values @ w + 1e-12)
        return -(port_ret - risk_free_rate) / port_vol

    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]

    # Try multiple starting points to avoid local minima
    best_result = None
    best_val = np.inf

    starts = [
        np.ones(n) / n,
        np.array([1.0 if i == np.argmax(excess_returns) else 0.0 for i in range(n)]),
    ]

    for x0 in starts:
        result = minimize(neg_sharpe,
                          x0=x0,
                          method='SLSQP',
                          bounds=[(0, 1)] * n,
                          constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-9})
        if result.success and result.fun < best_val:
            best_val = result.fun
            best_result = result

    if best_result is not None and best_result.success:
        w = np.clip(best_result.x, 0, 1)
        w /= w.sum()
        return pd.Series(w, index=tickers)

    return pd.Series(np.ones(n) / n, index=tickers)


# ==============================================================================
# 3. MINIMUM CVaR  (Linear Program)
# ==============================================================================
def optimize_min_cvar(historical_returns: pd.DataFrame,
                      expected_returns: pd.Series,
                      target_return: float = None,
                      alpha: float = 0.05) -> pd.Series:
    """
    Minimizes Conditional Value-at-Risk (CVaR / Expected Shortfall) at
    confidence level (1 - alpha). Default alpha=0.05 → minimize expected loss
    in the worst 5% of scenarios.

    LP Formulation (Rockafellar & Uryasev, 2000):
    
    Variables: w (weights), ζ (VaR threshold), z_t (auxiliary loss variables)

    Minimize:  ζ + (1/αT) Σ z_t
    Subject to:
        z_t ≥ -r_t'w - ζ   ∀ t   (loss exceedance)
        z_t ≥ 0              ∀ t
        Σ wᵢ = 1
        wᵢ ≥ 0
        μ'w ≥ target_return  (optional)
    """
    n = len(expected_returns)
    tickers = expected_returns.index
    T = len(historical_returns)  # number of scenarios (historical days)
    R = historical_returns[tickers].values  # T × n return matrix

    # ── Gurobi path ────────────────────────────────────────────────────────────
    try:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("min_cvar")
        m.setParam('OutputFlag', 0)

        w   = m.addMVar(n, lb=0.0, ub=1.0, name="w")
        zeta = m.addVar(lb=-GRB.INFINITY, name="zeta")   # VaR estimate
        z   = m.addMVar(T, lb=0.0, name="z")              # loss exceedances

        # Budget + long-only
        m.addConstr(w.sum() == 1, "budget")

        # Optional target return
        if target_return is not None:
            m.addConstr(expected_returns.values @ w >= target_return, "target_ret")

        # CVaR loss constraints: z_t ≥ -R[t]'w - ζ
        for t in range(T):
            m.addConstr(z[t] >= -R[t] @ w - zeta)

        # Objective: minimize VaR + (1/αT) Σ z_t
        m.setObjective(zeta + (1.0 / (alpha * T)) * z.sum(), GRB.MINIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            return pd.Series(w.X, index=tickers)

    except Exception:
        pass  # fall through to scipy LP

    # ── Scipy linprog fallback ─────────────────────────────────────────────────
    from scipy.optimize import linprog

    # Variable vector: [w (n), zeta (1), z (T)]
    # Objective: min  0*w + 1*zeta + (1/αT)*z
    c_obj = np.concatenate([np.zeros(n), [1.0], np.ones(T) / (alpha * T)])

    # Inequality constraints: -R[t]'w - zeta - z[t] ≤ 0  →  z_t ≥ -R[t]'w - ζ
    # Row t:  -R[t,:]  (for w)  | -1  (for zeta)  | -e_t  (for z)
    A_ub_list = []
    for t in range(T):
        row = np.zeros(n + 1 + T)
        row[:n]       = -R[t]   # -R[t]'w
        row[n]        = -1.0    # -zeta
        row[n + 1 + t] = -1.0  # -z_t
        A_ub_list.append(row)
    A_ub = np.array(A_ub_list)
    b_ub = np.zeros(T)

    # Equality: Σ wᵢ = 1
    A_eq = np.zeros((1, n + 1 + T))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # Bounds
    bounds = [(0, 1)] * n + [(None, None)] + [(0, None)] * T

    # Optional target return as additional inequality
    if target_return is not None:
        ret_row = np.zeros((1, n + 1 + T))
        ret_row[0, :n] = -expected_returns.values  # negate for ≤ form
        A_ub = np.vstack([A_ub, ret_row])
        b_ub = np.append(b_ub, -target_return)

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                     A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        w_opt = np.clip(result.x[:n], 0, 1)
        if w_opt.sum() > 0:
            w_opt /= w_opt.sum()
            return pd.Series(w_opt, index=tickers)

    return pd.Series(np.ones(n) / n, index=tickers)


# ==============================================================================
# Convenience wrapper (backward compatibility with old backtester.py)
# ==============================================================================
def optimize_portfolio(expected_returns, cov_matrix, target_return=None):
    """Backward-compatible wrapper — calls Min-Variance."""
    return optimize_min_variance(expected_returns, cov_matrix, target_return)

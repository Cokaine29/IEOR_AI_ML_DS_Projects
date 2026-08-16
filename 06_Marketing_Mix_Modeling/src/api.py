"""
api.py
======
FastAPI service exposing the MMM optimizer as a REST endpoint.

Endpoints:
  GET  /                       - health check
  GET  /channel_contributions  - current channel attribution breakdown
  POST /optimize_budget        - given total budget, return optimal allocation
  GET  /sensitivity            - budget sensitivity analysis (50%-150%)

Usage:
  uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))
from optimizer import channel_response, optimize_budget, sensitivity_analysis, compute_marginal_roi

BASE_DIR    = Path(__file__).parent.parent
METRICS_DIR = BASE_DIR / "results" / "metrics"

app = FastAPI(
    title="Marketing Mix Modeling API",
    description=(
        "Optimal budget reallocation across marketing channels "
        "using fitted MMM response curves."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pre-computed results on startup ──────────────────────────────────────

_STATE: dict = {}

@app.on_event("startup")
def load_state():
    """Load pre-computed response params and contributions from disk."""
    global _STATE
    try:
        # Load transform params
        tp_path = METRICS_DIR / "transform_params.json"
        with open(tp_path) as f:
            _STATE["transform_params"] = json.load(f)

        # Load optimization result
        opt_path = METRICS_DIR / "optimization_result.csv"
        opt_df   = pd.read_csv(opt_path)
        _STATE["opt_df"] = opt_df

        # Build resp_params from optimization CSV
        resp_params = {}
        for _, row in opt_df.iterrows():
            ch = row["Channel"]
            resp_params[ch] = {
                "coef":          float(row["coef"]),
                "theta":         float(row["theta"]),
                "alpha":         float(row["alpha"]),
                "gamma":         float(row["gamma"]),
                "mean_spend":    float(row["mean_spend"]),
                "max_spend_ref": float(opt_df["Current_Spend"].max() * 1.5),
            }
        _STATE["resp_params"] = resp_params
        _STATE["channels"]    = list(resp_params.keys())
        _STATE["base_budget"] = float(opt_df["Current_Spend"].sum())

        # Load contributions
        contrib_path = METRICS_DIR / "channel_contributions.csv"
        if contrib_path.exists():
            _STATE["contributions"] = pd.read_csv(contrib_path).to_dict(orient="records")
        else:
            _STATE["contributions"] = []

        print("[api] State loaded successfully.")
    except FileNotFoundError as e:
        print(f"[api] WARNING: Could not load state: {e}")
        print("[api] Run mmm_model.py and optimizer.py first to generate results.")
        _STATE["resp_params"]   = {}
        _STATE["channels"]      = []
        _STATE["base_budget"]   = 1000.0
        _STATE["contributions"] = []


# ── Request / Response schemas ─────────────────────────────────────────────────

class BudgetRequest(BaseModel):
    total_budget: float = Field(..., gt=0, description="Total weekly marketing budget")
    n_restarts:   int   = Field(10, ge=1, le=50, description="Optimizer restarts")


class AllocationResult(BaseModel):
    channel: str
    current_spend:  float
    optimal_spend:  float
    change:         float
    change_pct:     float


class OptimizeResponse(BaseModel):
    total_budget:                float
    predicted_sales_current:     float
    predicted_sales_optimal:     float
    predicted_sales_lift:        float
    predicted_sales_lift_pct:    float
    convergence:                 bool
    allocation:                  list[AllocationResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status":   "ok",
        "channels": _STATE.get("channels", []),
        "base_budget": _STATE.get("base_budget", 0),
    }


@app.get("/channel_contributions")
def get_contributions():
    """Return channel attribution breakdown from the fitted model."""
    if not _STATE.get("contributions"):
        raise HTTPException(status_code=404, detail="Contributions not found. Run pipeline first.")
    return {"contributions": _STATE["contributions"]}


@app.post("/optimize_budget", response_model=OptimizeResponse)
def post_optimize_budget(req: BudgetRequest):
    """
    Given a total weekly marketing budget, return the optimal channel allocation
    that maximizes predicted sales.
    """
    resp_params = _STATE.get("resp_params", {})
    channels    = _STATE.get("channels", [])
    if not resp_params:
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")

    result = optimize_budget(resp_params, req.total_budget, channels, req.n_restarts)

    allocation = []
    for ch in channels:
        cur = result["current_allocation"][ch]
        opt = result["optimal_allocation"][ch]
        allocation.append(AllocationResult(
            channel=ch,
            current_spend=round(cur, 2),
            optimal_spend=round(opt, 2),
            change=round(opt - cur, 2),
            change_pct=round((opt - cur) / (cur + 1e-8) * 100, 1),
        ))

    return OptimizeResponse(
        total_budget=round(req.total_budget, 2),
        predicted_sales_current=round(result["predicted_sales_current"], 2),
        predicted_sales_optimal=round(result["predicted_sales_optimal"], 2),
        predicted_sales_lift=round(result["predicted_sales_lift"], 2),
        predicted_sales_lift_pct=round(result["predicted_sales_lift_pct"], 2),
        convergence=result["convergence"],
        allocation=allocation,
    )


@app.get("/sensitivity")
def get_sensitivity(budget_min_pct: float = 0.5, budget_max_pct: float = 1.5, steps: int = 10):
    """Return sensitivity analysis: how predicted sales change with budget level."""
    resp_params = _STATE.get("resp_params", {})
    channels    = _STATE.get("channels", [])
    base_budget = _STATE.get("base_budget", 1000.0)
    if not resp_params:
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")

    budget_range = list(np.linspace(budget_min_pct, budget_max_pct, steps))
    records      = []
    for pct in budget_range:
        budget = base_budget * pct
        res    = optimize_budget(resp_params, budget, channels, n_restarts=5)
        records.append({
            "budget_pct":       round(pct * 100, 1),
            "total_budget":     round(budget, 1),
            "predicted_sales":  round(res["predicted_sales_optimal"], 1),
            "allocation":       {ch: round(res["optimal_allocation"][ch], 1) for ch in channels},
        })
    return {"sensitivity": records}

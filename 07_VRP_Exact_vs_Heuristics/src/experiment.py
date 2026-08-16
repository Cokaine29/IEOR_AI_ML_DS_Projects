import json
import os
import time
from pathlib import Path

from src.data_loader import load_vrp_instance
from src.cost_model import build_distance_matrix
from src.solvers.exact_pulp import solve_exact
from src.solvers.sa import solve_sa
from src.solvers.ga import solve_ga

# The instances we want to run
INSTANCES_TO_RUN = [
    "Gen-n9",
    "Gen-n11",
    "Gen-n13",
    "Gen-n15",
    "Gen-n17",
    "Gen-n21",
    "Gen-n31",
    "Gen-n41",
    "Gen-n61",
    "Gen-n81",
]

def run_experiment():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "raw"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    results = []
    
    # We will limit exact solver strictly because it explodes
    EXACT_TIME_LIMIT = 60 # 1 minute per instance
    EXACT_MAX_N = 25 # Don't even try exact if N > 25 customers
    
    # Heuristics settings
    HEURISTIC_TIME_LIMIT = 2 # 2 seconds per instance
    SEEDS = 1 # Run each heuristic 1 times to get average
    
    for inst_name in INSTANCES_TO_RUN:
        filepath = data_dir / f"{inst_name}.vrp"
        if not filepath.exists():
            print(f"Skipping {inst_name} - file not found.")
            continue
            
        print(f"\n{'='*40}")
        print(f"Running Instance: {inst_name}")
        
        inst = load_vrp_instance(inst_name)
        dist_matrix = build_distance_matrix(inst.coords)
        
        print(f"Customers: {inst.n_customers}")
        
        res = {
            "instance": inst_name,
            "n_customers": inst.n_customers,
            "exact": None,
            "sa_avg_cost": None,
            "sa_avg_time": None,
            "sa_best_cost": None,
            "ga_avg_cost": None,
            "ga_avg_time": None,
            "ga_best_cost": None,
        }
        
        # 1. Exact Solver
        if inst.n_customers <= EXACT_MAX_N:
            print(f"  [Exact] Running PuLP/CBC (Limit {EXACT_TIME_LIMIT}s)...")
            exact_cost, _, exact_time = solve_exact(inst, dist_matrix, time_limit_secs=EXACT_TIME_LIMIT)
            if exact_cost is not None:
                print(f"    Found Exact: {exact_cost} in {exact_time:.2f}s")
                res["exact"] = {"cost": exact_cost, "time": exact_time}
            else:
                print(f"    Exact failed or timed out after {exact_time:.2f}s.")
                res["exact"] = {"cost": None, "time": exact_time}
        else:
            print(f"  [Exact] Skipping (N > {EXACT_MAX_N})")
            
        # 2. Simulated Annealing
        print(f"  [SA] Running {SEEDS} seeds (Limit {HEURISTIC_TIME_LIMIT}s per seed)...")
        sa_costs = []
        sa_times = []
        for seed in range(SEEDS):
            import random; random.seed(seed)
            c, _, t = solve_sa(inst, dist_matrix, time_limit_secs=HEURISTIC_TIME_LIMIT)
            sa_costs.append(c)
            sa_times.append(t)
        
        res["sa_avg_cost"] = sum(sa_costs) / SEEDS
        res["sa_avg_time"] = sum(sa_times) / SEEDS
        res["sa_best_cost"] = min(sa_costs)
        print(f"    Avg Cost: {res['sa_avg_cost']:.1f} (Best: {res['sa_best_cost']}) | Avg Time: {res['sa_avg_time']:.2f}s")
        
        # 3. Genetic Algorithm
        print(f"  [GA] Running {SEEDS} seeds (Limit {HEURISTIC_TIME_LIMIT}s per seed)...")
        ga_costs = []
        ga_times = []
        for seed in range(SEEDS):
            import random; random.seed(seed)
            c, _, t = solve_ga(inst, dist_matrix, time_limit_secs=HEURISTIC_TIME_LIMIT)
            ga_costs.append(c)
            ga_times.append(t)
            
        res["ga_avg_cost"] = sum(ga_costs) / SEEDS
        res["ga_avg_time"] = sum(ga_times) / SEEDS
        res["ga_best_cost"] = min(ga_costs)
        print(f"    Avg Cost: {res['ga_avg_cost']:.1f} (Best: {res['ga_best_cost']}) | Avg Time: {res['ga_avg_time']:.2f}s")
        
        results.append(res)
        
        # Save intermediate
        with open(results_dir / "experiment_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
    print("\nExperiment Complete. Results saved to results/experiment_results.json")

if __name__ == "__main__":
    run_experiment()

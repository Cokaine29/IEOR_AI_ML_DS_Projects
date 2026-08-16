# Project 07: Vehicle Routing Problem (VRP) — Exact vs. Metaheuristics

A Python and Streamlit project analyzing the computational trade-offs between exact mathematical solvers (MILP) and approximation metaheuristics (Simulated Annealing, Genetic Algorithms) for the Capacitated Vehicle Routing Problem (CVRP).

## Quick Start

The best way to explore this project is through the **Interactive Streamlit Dashboard**, which allows you to generate synthetic VRP instances, run exact and heuristic solvers live, and visualize the resulting routes.

```bash
# Install dependencies (if you haven't already)
pip install pulp vrplib streamlit pandas matplotlib numpy

# Run the interactive dashboard
streamlit run src/dashboard.py
```

## Detailed Analysis

For the complete technical breakdown, methodology, benchmark results (the "Crossover Point"), and resume bullets, please read the full report:

👉 **[Read the Full Project Report here (Report.md)](Report.md)**

## Code Structure

- `src/solvers/exact_pulp.py`: MTZ formulation using PuLP/CBC.
- `src/solvers/sa.py`: Simulated Annealing implementation.
- `src/solvers/ga.py`: Genetic Algorithm implementation.
- `src/dashboard.py`: The interactive Streamlit UI.
- `src/experiment.py`: The script used to run the static offline benchmarks.

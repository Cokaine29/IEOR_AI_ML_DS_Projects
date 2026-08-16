# Project 07: Vehicle Routing Problem (VRP) — Exact vs. Metaheuristics

## Executive Summary

Every major logistics provider (Amazon, FedEx, Swiggy) runs continuous routing optimizations. The core tension in production routing engines is the trade-off between **solution optimality** and **computational feasibility**.

The Capacitated Vehicle Routing Problem (CVRP) is NP-hard. For small numbers of customers, Exact Solvers (Mixed Integer Linear Programming) can guarantee an optimal minimum-distance route. However, as the customer count grows, the exact solve time explodes combinatorially. In production, a routing engine that takes 3 hours to run is useless; it must run in seconds. Thus, practitioners rely on Metaheuristics (Simulated Annealing, Genetic Algorithms) which sacrifice a guarantee of optimality for bounded compute time.

This project empirically quantifies that "Crossover Point" by benchmarking an exact PuLP/CBC MILP formulation against custom-built Simulated Annealing and Genetic Algorithm implementations across scaling instance sizes.

---

## Dataset

We use a set of CVRP instances structured identically to the standard **CVRPLIB Augerat** format. Instances scale from $N=8$ to $N=80$ customers, all served from a central depot with uniform vehicle capacities.

---

## Methodology

### 1. Exact Solver (MILP Baseline)
Formulated as a complete graph routing problem using the **Miller-Tucker-Zemlin (MTZ)** subtour elimination constraints.
- **Variables**: $x_{ij} \in \{0, 1\}$ (edge active), $u_i \in \mathbb{R}$ (cumulative load at node i).
- **Solver**: CBC (via PuLP) with a strict 60-second time limit per instance.

### 2. Construction Heuristic
**Clarke-Wright Savings Algorithm**
Used to generate a strong, feasible initial solution for the metaheuristics, drastically reducing convergence time compared to random initialization.

### 3. Metaheuristics

#### Simulated Annealing (SA)
- **Neighborhood Operators**: Relocate (move a customer), Swap (exchange two customers).
- **Cooling**: Geometric schedule.
- **Time Bound**: Strictly terminated after 2.0 seconds per instance.

#### Genetic Algorithm (GA)
- **Chromosome**: Flat permutation of customers, decoded into feasible routes via a split algorithm.
- **Crossover**: Order Crossover (OX) adapted for permutations.
- **Mutation**: Swap mutation.
- **Time Bound**: Strictly terminated after 2.0 seconds per instance.

---

## The Crossover Point (Results)

*Note: View the full interactive dashboard via `streamlit run src/dashboard.py`*

The experiment clearly demonstrates the exponential blowup of the exact solver:

1. **Small Scale ($N \le 10$)**: Exact solving takes $< 2$ seconds. Metaheuristics find the true optimal within their 2-second time budget (0% gap).
2. **The Crossover ($N \approx 14-16$)**: Exact solver time jumps to $> 60$ seconds (hitting the timeout wall). SA and GA continue returning high-quality solutions (within 1-5% of optimal) in precisely 2.0 seconds.
3. **Production Scale ($N \ge 30$)**: Exact solving is completely computationally infeasible. SA reliably outperforms GA in the strict 2-second time window, maintaining a tight optimality gap.

### Business Implication
In an interview context, this project proves a critical understanding: "We don't use metaheuristics because we don't know the exact math; we use them because at $N=30$, an exact solver would stall our entire dispatch system. By giving up ~3% optimality, SA guarantees our drivers get their routes in exactly 2 seconds."

---

## Interactive Solver (Demo UI)

To bring the analysis to life, the project includes an interactive Streamlit UI (`streamlit run src/dashboard.py`). This allows for live generation of custom VRP instances and head-to-head benchmarking of all three solvers in real time.
- **Custom Parameter Generation:** Sliders for customer count ($N$), fleet size, and vehicle capacity. Includes an automatic infeasibility check to prevent unsolvable routing constraints.
- **Live Benchmarking:** Watch the exact MILP solver hit a time-limit wall on large instances while Simulated Annealing and Genetic Algorithms compute routes in a strictly enforced 2-3 second budget.
- **Route Visualizations:** Visualizes the computed paths (customers, depot, and vehicle assignments) side-by-side for real-time comparison.

---

## Architecture / Code Structure

- `src/solvers/exact_pulp.py`: MTZ formulation using PuLP.
- `src/solvers/sa.py`: Simulated Annealing implementation.
- `src/solvers/ga.py`: Genetic Algorithm implementation.
- `src/experiment.py`: The benchmarking pipeline.
- `src/dashboard.py`: Streamlit visualization.

## Resume Bullets (OR / Supply Chain Target)

- Quantified the computational crossover point for the **Capacitated Vehicle Routing Problem (CVRP)**, benchmarking an exact **MILP** formulation (MTZ constraints via **PuLP/CBC**) against metaheuristics to justify heuristic usage in production.
- Engineered **Simulated Annealing** and **Genetic Algorithm** (Order Crossover) solvers in Python, seeded by a **Clarke-Wright Savings** construction heuristic to accelerate convergence.
- Demonstrated that for $N \ge 16$ customers, exact solve times explode exponentially ($>60$ seconds), whereas Simulated Annealing bounded to a 2-second compute budget maintained a sub-5% optimality gap, validating the industry-standard trade-off.
- Built an interactive **Streamlit** interface to generate synthetic CVRP instances and compare exact/heuristic solvers live, with enforced solver time limits and real-time route visualization.

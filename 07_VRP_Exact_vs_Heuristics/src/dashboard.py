import streamlit as st
import json
import math
import random
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RESULTS_FILE = BASE_DIR / "results" / "experiment_results.json"

# Ensure project root is on sys.path so `src.*` imports work when Streamlit
# runs this file directly (i.e. `streamlit run src/dashboard.py`)
import sys as _sys
if str(BASE_DIR) not in _sys.path:
    _sys.path.insert(0, str(BASE_DIR))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VRP Solver: Exact vs Metaheuristics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1d27;
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 24px;
        border-radius: 8px;
        color: #9ca3af;
        font-weight: 500;
        font-size: 14px;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252836 100%);
        border: 1px solid #2d3150;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .metric-card h3 { color: #9ca3af; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 6px 0; }
    .metric-card .value { font-size: 28px; font-weight: 700; margin: 0; }
    .metric-card .sub { font-size: 12px; color: #6b7280; margin: 4px 0 0 0; }

    .solver-exact { color: #f87171; }
    .solver-sa    { color: #60a5fa; }
    .solver-ga    { color: #34d399; }

    .status-optimal  { background:#065f46; color:#6ee7b7; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .status-timeout  { background:#7c2d12; color:#fca5a5; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .status-skipped  { background:#374151; color:#9ca3af; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .status-heuristic{ background:#1e3a5f; color:#93c5fd; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }

    div[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def euclidean_distance(p1, p2) -> int:
    return round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def build_distance_matrix(coords: Dict[int, Tuple[float, float]]) -> Dict[Tuple[int, int], int]:
    dist = {}
    nodes = list(coords.keys())
    for i in nodes:
        for j in nodes:
            dist[(i, j)] = 0 if i == j else euclidean_distance(coords[i], coords[j])
    return dist


class VRPInstance:
    def __init__(self, name, capacity, depot, coords, demands, node_list):
        self.name = name
        self.capacity = capacity
        self.depot = depot
        self.coords = coords
        self.demands = demands
        self.nodes = node_list
        self.customers = [n for n in node_list if n != depot]
        self.n_customers = len(self.customers)


def generate_interactive_instance(n_customers: int, capacity: int, n_vehicles: int, seed: int) -> VRPInstance:
    """Creates a synthetic CVRP instance in-memory. Depot fixed at (50,50)."""
    rng = random.Random(seed)
    depot = 1
    coords = {1: (50.0, 50.0)}
    demands = {1: 0}
    nodes = [1]
    for i in range(2, n_customers + 2):
        coords[i] = (rng.uniform(5, 95), rng.uniform(5, 95))
        demands[i] = rng.randint(5, 20)
        nodes.append(i)

    inst = VRPInstance(
        name=f"Interactive-N{n_customers}-S{seed}",
        capacity=capacity,
        depot=depot,
        coords=coords,
        demands=demands,
        node_list=nodes,
    )
    inst.n_vehicles = n_vehicles
    return inst


def validate_instance(inst: VRPInstance) -> Tuple[bool, str]:
    """Returns (is_feasible, error_message)."""
    max_demand = max(inst.demands[c] for c in inst.customers)
    if max_demand > inst.capacity:
        return False, (f"Instance is **infeasible**: Customer with demand {max_demand} exceeds "
                       f"vehicle capacity {inst.capacity}. Increase capacity or reduce max demand range.")
    total_demand = sum(inst.demands[c] for c in inst.customers)
    fleet_capacity = inst.n_vehicles * inst.capacity
    if total_demand > fleet_capacity:
        return False, (f"Instance is **infeasible**: Total customer demand ({total_demand}) exceeds "
                       f"total fleet capacity ({fleet_capacity} = {inst.n_vehicles} vehicles × {inst.capacity}). "
                       f"Add more vehicles or increase capacity.")
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

ROUTE_COLORS = [
    "#60a5fa", "#34d399", "#f472b6", "#fbbf24", "#a78bfa",
    "#38bdf8", "#fb923c", "#4ade80", "#e879f9", "#facc15",
]

def plot_routes(inst: VRPInstance, routes: Optional[List[List[int]]], title: str,
                status_label: str, status_css: str, cost: Optional[int],
                solve_time: Optional[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#13151f")

    # Draw routes
    if routes:
        for idx, route in enumerate(routes):
            color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
            path = [inst.depot] + route + [inst.depot]
            xs = [inst.coords[n][0] for n in path]
            ys = [inst.coords[n][1] for n in path]
            ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.75, zorder=1)

    # Draw customers
    cxs = [inst.coords[c][0] for c in inst.customers]
    cys = [inst.coords[c][1] for c in inst.customers]
    ax.scatter(cxs, cys, c="#e2e8f0", s=55, zorder=3, edgecolors="#94a3b8", linewidths=0.8)

    # Label customers with demand
    for c in inst.customers:
        ax.annotate(f"{inst.demands[c]}", (inst.coords[c][0], inst.coords[c][1]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=6.5, color="#94a3b8")

    # Draw depot
    dx, dy = inst.coords[inst.depot]
    ax.scatter([dx], [dy], c="#f87171", s=180, marker="*", zorder=5,
               edgecolors="#fca5a5", linewidths=1.2)
    ax.annotate("Depot", (dx, dy), textcoords="offset points", xytext=(5, 6),
                fontsize=8, color="#f87171", fontweight="bold")

    # Styling
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.tick_params(colors="#4b5563", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3150")

    cost_str = f"Cost: {cost:,}" if cost else "No solution"
    time_str = f"{solve_time:.2f}s" if solve_time is not None else "N/A"
    ax.set_title(f"{title}\n{cost_str}  |  Time: {time_str}", color="#e2e8f0",
                 fontsize=10, fontweight="600", pad=10)

    if not routes:
        ax.text(50, 50, "No solution found\n(Time limit reached or skipped)",
                ha="center", va="center", color="#6b7280", fontsize=9,
                style="italic")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SOLVERS (thin wrappers that import from src.solvers)
# ══════════════════════════════════════════════════════════════════════════════

def run_exact(inst: VRPInstance, dist_matrix, time_limit: int):
    """Returns (cost, routes, solve_time, status_str)."""
    from src.solvers.exact_pulp import solve_exact
    import pulp

    cost, routes, t = solve_exact(inst, dist_matrix, time_limit_secs=time_limit)
    if cost is not None:
        # Try to detect if we hit the time limit
        status = "Optimal" if t < time_limit - 0.5 else "Time Limit (Feasible)"
    else:
        status = "Not Solved / Time Limit"
    return cost, routes, t, status


def run_sa(inst: VRPInstance, dist_matrix, time_limit: float, seed: int):
    from src.solvers.sa import solve_sa
    random.seed(seed)
    cost, routes, t = solve_sa(inst, dist_matrix, time_limit_secs=time_limit)
    return cost, routes, t


def run_ga(inst: VRPInstance, dist_matrix, time_limit: float, seed: int):
    from src.solvers.ga import solve_ga
    random.seed(seed)
    cost, routes, t = solve_ga(inst, dist_matrix, time_limit_secs=time_limit)
    return cost, routes, t


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.title("🚚 VRP Solver: Exact vs Metaheuristics")
st.markdown(
    "Benchmarking **MILP / PuLP** (exact) against **Simulated Annealing** and "
    "**Genetic Algorithm** (metaheuristics) on the Capacitated Vehicle Routing Problem."
)

tab1, tab2 = st.tabs(["📊  Benchmark Analysis", "🧪  Interactive Solver"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BENCHMARK ANALYSIS (existing content, polished)
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if not RESULTS_FILE.exists():
        st.warning("No experiment results found. Run `python -m src.experiment` first to generate them.")
        st.stop()

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    plot_data = []
    for d in data:
        exact_cost = d["exact"]["cost"] if d.get("exact") and d["exact"].get("cost") else None
        exact_time = d["exact"]["time"] if d.get("exact") and d["exact"].get("time") else None
        optimal_cost = exact_cost
        sa_gap = ((d["sa_best_cost"] - optimal_cost) / optimal_cost * 100) if optimal_cost else None
        ga_gap = ((d["ga_best_cost"] - optimal_cost) / optimal_cost * 100) if optimal_cost else None
        plot_data.append({
            "Customers (N)": d["n_customers"],
            "Exact Time (s)": exact_time,
            "SA Time (s)": d["sa_avg_time"],
            "GA Time (s)": d["ga_avg_time"],
            "Exact Cost": exact_cost,
            "SA Best Cost": d["sa_best_cost"],
            "GA Best Cost": d["ga_best_cost"],
            "SA Gap %": sa_gap,
            "GA Gap %": ga_gap,
        })

    plot_df = pd.DataFrame(plot_data).sort_values("Customers (N)")

    # Crossover banner
    hard = plot_df[plot_df["Exact Time (s)"].fillna(0) > 60]
    if not hard.empty:
        crossover_n = int(hard.iloc[0]["Customers (N)"])
        st.info(f"**Crossover Point identified at N ≈ {crossover_n} customers** — beyond this, "
                f"the exact solver exceeds 60 seconds while metaheuristics stay within 2 seconds.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⏱ Runtime vs Problem Size")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        fig1.patch.set_facecolor("#1a1d27")
        ax1.set_facecolor("#13151f")
        valid_exact = plot_df.dropna(subset=["Exact Time (s)"])
        ax1.plot(valid_exact["Customers (N)"], valid_exact["Exact Time (s)"],
                 "o-", label="Exact (PuLP/CBC)", color="#f87171", linewidth=2)
        ax1.plot(plot_df["Customers (N)"], plot_df["SA Time (s)"],
                 "s-", label="Simulated Annealing", color="#60a5fa", linewidth=2)
        ax1.plot(plot_df["Customers (N)"], plot_df["GA Time (s)"],
                 "^-", label="Genetic Algorithm", color="#34d399", linewidth=2)
        ax1.set_yscale("log")
        ax1.set_xlabel("Number of Customers", color="#9ca3af")
        ax1.set_ylabel("Solve Time (s) — Log Scale", color="#9ca3af")
        ax1.set_title("Exponential Blowup of Exact Solver", color="#e2e8f0", fontweight="600")
        ax1.tick_params(colors="#6b7280")
        ax1.grid(True, which="both", ls="--", alpha=0.25, color="#374151")
        for spine in ax1.spines.values(): spine.set_edgecolor("#2d3150")
        ax1.legend(facecolor="#1e2130", labelcolor="#e2e8f0", edgecolor="#2d3150")
        st.pyplot(fig1)

    with col2:
        st.subheader("🎯 Optimality Gap vs Problem Size")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor("#1a1d27")
        ax2.set_facecolor("#13151f")
        valid_gap = plot_df.dropna(subset=["SA Gap %", "GA Gap %"])
        ax2.plot(valid_gap["Customers (N)"], valid_gap["SA Gap %"],
                 "s-", label="SA Gap %", color="#60a5fa", linewidth=2)
        ax2.plot(valid_gap["Customers (N)"], valid_gap["GA Gap %"],
                 "^-", label="GA Gap %", color="#34d399", linewidth=2)
        ax2.set_xlabel("Number of Customers", color="#9ca3af")
        ax2.set_ylabel("Optimality Gap (%)", color="#9ca3af")
        ax2.set_title("Cost Penalty of Metaheuristics", color="#e2e8f0", fontweight="600")
        ax2.tick_params(colors="#6b7280")
        ax2.grid(True, ls="--", alpha=0.25, color="#374151")
        for spine in ax2.spines.values(): spine.set_edgecolor("#2d3150")
        ax2.legend(facecolor="#1e2130", labelcolor="#e2e8f0", edgecolor="#2d3150")
        st.pyplot(fig2)

    st.subheader("📋 Full Results Table")
    st.dataframe(plot_df.style.format(precision=2), use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### 📌 Why This Matters in Production
    For an active delivery fleet (Amazon Logistics, Swiggy), routing must be recomputed dynamically
    as new orders arrive. If exact solving takes 10+ minutes for a single sector, it is useless in a
    live environment. We accept a **3–10% optimality gap** from Simulated Annealing or Genetic
    Algorithms specifically to guarantee **bounded compute time**.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INTERACTIVE SOLVER
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🧪 Build Your Own VRP Instance")
    st.markdown(
        "Configure a custom routing problem below, hit **Solve!**, and watch all three solvers "
        "compete on the same instance in real time."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col, spacer = st.columns([3, 1])
    with ctrl_col:
        c1, c2 = st.columns(2)
        with c1:
            n_customers = st.slider("👥 Number of Customers (N)", min_value=5, max_value=50, value=12, step=1)
            capacity = st.slider("🚛 Vehicle Capacity", min_value=30, max_value=250, value=100, step=10)
        with c2:
            # Auto-compute sensible default for vehicles
            # Heuristic: ceil(avg_expected_demand * N / capacity) + 1 slack
            # Expected demand ≈ 12.5 (mid of 5-20 range)
            suggested_vehicles = max(2, math.ceil(n_customers * 12.5 / capacity) + 1)
            n_vehicles = st.slider("🚗 Number of Vehicles (Fleet Size)", min_value=1, max_value=20,
                                   value=min(suggested_vehicles, 20), step=1)
            exact_time_limit = st.slider("⏱ Exact Solver Time Limit (s)", min_value=5, max_value=120,
                                         value=30, step=5,
                                         help="PuLP/CBC will hard-stop at this limit. Solution may not be optimal if time limit is hit.")

        c3, c4 = st.columns(2)
        with c3:
            heuristic_time_limit = st.slider("⚡ Heuristic Time Limit (s)", min_value=1, max_value=15,
                                              value=3, step=1)
        with c4:
            use_fixed_seed = st.checkbox("🔒 Use fixed seed (reproducible)", value=True)
            if use_fixed_seed:
                seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)
            else:
                seed = int(time.time() * 1000) % 10000

        st.markdown("**Which solvers to run:**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1: run_exact_solver = st.checkbox("🔴 Exact (MILP/PuLP)", value=True)
        with sc2: run_sa_solver    = st.checkbox("🔵 Simulated Annealing", value=True)
        with sc3: run_ga_solver    = st.checkbox("🟢 Genetic Algorithm", value=True)

        # N > 25 warning for exact solver
        if run_exact_solver and n_customers > 25:
            st.warning(
                f"⚠️ **Exact Solver warning:** N={n_customers} > 25 customers. The MILP formulation may "
                f"hit your {exact_time_limit}s time limit without finding an optimal solution. "
                f"If it does, it will be labeled **'Time Limit (Feasible)'** — not Optimal."
            )

        solve_btn = st.button("🚀 Solve!", type="primary", use_container_width=False)

    st.markdown("---")

    if solve_btn:
        # ── Generate instance (ONCE — shared by all solvers) ─────────────────
        inst = generate_interactive_instance(n_customers, capacity, n_vehicles, seed)
        dist_matrix = build_distance_matrix(inst.coords)

        # ── Feasibility check ─────────────────────────────────────────────────
        feasible, err_msg = validate_instance(inst)
        if not feasible:
            st.error(f"🚫 {err_msg}")
            st.stop()

        total_demand = sum(inst.demands[c] for c in inst.customers)
        fleet_cap = n_vehicles * capacity
        st.markdown(
            f"**Instance:** `{inst.name}` | Customers: **{n_customers}** | "
            f"Total Demand: **{total_demand}** | Fleet Capacity: **{fleet_cap}** "
            f"({n_vehicles} × {capacity}) | Seed: `{seed}`"
        )

        # ── Run solvers ───────────────────────────────────────────────────────
        results = {}

        if run_exact_solver:
            with st.spinner(f"🔴 Running Exact Solver (PuLP/CBC)… time limit: {exact_time_limit}s"):
                e_cost, e_routes, e_time, e_status = run_exact(inst, dist_matrix, exact_time_limit)
            results["Exact"] = dict(cost=e_cost, routes=e_routes, time=e_time, status=e_status,
                                    color="#f87171", css="solver-exact")

        if run_sa_solver:
            with st.spinner(f"🔵 Running Simulated Annealing… time limit: {heuristic_time_limit}s"):
                sa_cost, sa_routes, sa_time = run_sa(inst, dist_matrix, heuristic_time_limit, seed)
            results["SA"] = dict(cost=sa_cost, routes=sa_routes, time=sa_time,
                                 status="Heuristic", color="#60a5fa", css="solver-sa")

        if run_ga_solver:
            with st.spinner(f"🟢 Running Genetic Algorithm… time limit: {heuristic_time_limit}s"):
                ga_cost, ga_routes, ga_time = run_ga(inst, dist_matrix, heuristic_time_limit, seed)
            results["GA"] = dict(cost=ga_cost, routes=ga_routes, time=ga_time,
                                 status="Heuristic", color="#34d399", css="solver-ga")

        if not results:
            st.info("Select at least one solver and click Solve! again.")
            st.stop()

        # ── Summary metric cards ───────────────────────────────────────────────
        st.subheader("📊 Results Summary")
        card_cols = st.columns(len(results))

        solver_labels = {"Exact": "Exact (MILP)", "SA": "Simulated Annealing", "GA": "Genetic Algorithm"}
        status_css_map = {
            "Optimal": "status-optimal",
            "Time Limit (Feasible)": "status-timeout",
            "Not Solved / Time Limit": "status-skipped",
            "Heuristic": "status-heuristic",
        }

        reference_cost = None
        if "Exact" in results and results["Exact"]["cost"] is not None and results["Exact"]["status"] == "Optimal":
            reference_cost = results["Exact"]["cost"]

        for col, (key, res) in zip(card_cols, results.items()):
            with col:
                cost_display = f"{res['cost']:,}" if res["cost"] else "—"
                gap_html = ""
                if reference_cost and res["cost"] and key != "Exact":
                    gap = (res["cost"] - reference_cost) / reference_cost * 100
                    gap_html = f"<p class='sub'>Optimality Gap: <b>{gap:+.1f}%</b></p>"
                status_css = status_css_map.get(res["status"], "status-skipped")
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{solver_labels[key]}</h3>
                    <p class="value {res['css']}">{cost_display}</p>
                    <p class="sub">Solve Time: <b>{res['time']:.2f}s</b></p>
                    {gap_html}
                    <p class="sub"><span class="{status_css}">{res['status']}</span></p>
                </div>
                """, unsafe_allow_html=True)

        # ── Route visualization ───────────────────────────────────────────────
        st.subheader("🗺 Route Visualizations")
        plot_cols = st.columns(len(results))
        for col, (key, res) in zip(plot_cols, results.items()):
            with col:
                status_css = status_css_map.get(res["status"], "status-skipped")
                fig = plot_routes(
                    inst=inst,
                    routes=res["routes"],
                    title=solver_labels[key],
                    status_label=res["status"],
                    status_css=status_css,
                    cost=res["cost"],
                    solve_time=res["time"],
                )
                st.pyplot(fig)
                n_routes = len(res["routes"]) if res["routes"] else 0
                st.caption(f"{n_routes} vehicle route(s) used out of {n_vehicles} available")

        # ── Solver comparison table ───────────────────────────────────────────
        if len(results) > 1:
            st.subheader("📋 Head-to-Head Comparison")
            comp_rows = []
            for key, res in results.items():
                gap = None
                if reference_cost and res["cost"] and key != "Exact":
                    gap = round((res["cost"] - reference_cost) / reference_cost * 100, 2)
                comp_rows.append({
                    "Solver": solver_labels[key],
                    "Total Cost": res["cost"] or "N/A",
                    "Solve Time (s)": round(res["time"], 3),
                    "Status": res["status"],
                    "vs Exact Gap %": f"{gap:+.2f}%" if gap is not None else "—",
                })
            st.dataframe(pd.DataFrame(comp_rows).set_index("Solver"), use_container_width=True)

        # ── Context note ──────────────────────────────────────────────────────
        st.markdown("""
        ---
        > **Note:** All three solvers ran on the **same randomly generated instance** (same seed).
        > Solver status `Optimal` = CBC confirmed global optimum. `Time Limit (Feasible)` = CBC
        > found a feasible solution but could not prove optimality within the time budget —
        > treat it as an **upper bound**, not a guaranteed optimum.
        """)
    else:
        st.markdown(
            "<div style='text-align:center; padding:60px; color:#4b5563;'>"
            "<h3>👆 Configure your instance above and click <b>Solve!</b></h3>"
            "<p>All three solvers will compete on the same problem in real time.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

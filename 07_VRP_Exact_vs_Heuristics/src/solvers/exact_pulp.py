import pulp
import time
from typing import Dict, Tuple, List, Optional
from ..data_loader import VRPInstance
from ..cost_model import calculate_total_cost

def solve_exact(inst: VRPInstance, dist_matrix: Dict[Tuple[int, int], int], time_limit_secs: int = 600) -> Tuple[Optional[int], Optional[List[List[int]]], float]:
    """
    Solves CVRP optimally using PuLP and MTZ (Miller-Tucker-Zemlin) formulation.
    Returns: (best_cost, routes, solve_time)
    """
    start_time = time.time()
    
    # 1. Initialize Problem
    prob = pulp.LpProblem(f"CVRP_{inst.name}", pulp.LpMinimize)
    
    # 2. Decision Variables
    # x[i][j] = 1 if vehicle travels directly from i to j
    nodes = inst.nodes
    x = pulp.LpVariable.dicts("x", 
                              ((i, j) for i in nodes for j in nodes if i != j),
                              cat='Binary')
    
    # u[i] = cumulative demand picked up after visiting node i (for MTZ subtour elimination)
    # Range is from demand[i] up to Capacity
    u = pulp.LpVariable.dicts("u", 
                              (i for i in inst.customers), 
                              lowBound=0, upBound=inst.capacity, cat='Continuous')
    
    # 3. Objective Function
    prob += pulp.lpSum(dist_matrix[(i, j)] * x[(i, j)] for i in nodes for j in nodes if i != j)
    
    # 4. Constraints
    # Ensure exactly one outgoing edge from each customer
    for i in inst.customers:
        prob += pulp.lpSum(x[(i, j)] for j in nodes if i != j) == 1
        
    # Ensure exactly one incoming edge to each customer
    for j in inst.customers:
        prob += pulp.lpSum(x[(i, j)] for i in nodes if i != j) == 1
        
    # MTZ Subtour Elimination and Capacity Constraints
    for i in inst.customers:
        # u[i] is at least its own demand
        prob += u[i] >= inst.demands[i]
        
        for j in inst.customers:
            if i != j:
                # If x[i,j]=1, then u[j] >= u[i] + demand[j]
                # Using Big-M: M = Capacity
                prob += u[i] - u[j] + inst.capacity * x[(i, j)] <= inst.capacity - inst.demands[j]
    
    # 5. Solve
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_secs, msg=False)
    prob.solve(solver)
    
    solve_time = time.time() - start_time
    
    status = pulp.LpStatus[prob.status]
    if status not in ['Optimal', 'Feasible']:
        return None, None, solve_time
        
    # 6. Extract Routes
    routes = []
    
    # Find all edges starting from depot
    for j in inst.customers:
        if pulp.value(x[(inst.depot, j)]) and pulp.value(x[(inst.depot, j)]) > 0.5:
            # We found a starting point of a route
            curr_route = [j]
            curr_node = j
            while True:
                # Find the next node
                next_node = None
                for k in nodes:
                    if curr_node != k and pulp.value(x[(curr_node, k)]) and pulp.value(x[(curr_node, k)]) > 0.5:
                        next_node = k
                        break
                
                if next_node == inst.depot or next_node is None:
                    break
                else:
                    curr_route.append(next_node)
                    curr_node = next_node
                    
            routes.append(curr_route)
            
    best_cost = round(pulp.value(prob.objective))
    
    return best_cost, routes, solve_time

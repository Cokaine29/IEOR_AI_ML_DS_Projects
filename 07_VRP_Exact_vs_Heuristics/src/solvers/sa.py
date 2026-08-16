import random
import math
import time
import copy
from typing import Dict, Tuple, List
from ..data_loader import VRPInstance
from ..cost_model import calculate_total_cost
from .construction import clarke_wright_savings

def is_feasible(route: List[int], inst: VRPInstance) -> bool:
    load = sum(inst.demands[c] for c in route)
    return load <= inst.capacity

def get_neighborhood(routes: List[List[int]], inst: VRPInstance) -> List[List[int]]:
    """
    Generates a neighbor by applying a random routing operator:
    1. Relocate: move a customer from one route to another (or same route)
    2. Swap: swap two customers
    """
    new_routes = copy.deepcopy(routes)
    operator = random.choice(['relocate', 'swap'])
    
    if operator == 'relocate':
        if not new_routes: return new_routes
        # Pick source route and customer
        r1_idx = random.randint(0, len(new_routes) - 1)
        r1 = new_routes[r1_idx]
        if not r1: return new_routes
        c_idx = random.randint(0, len(r1) - 1)
        customer = r1.pop(c_idx)
        
        # Pick destination route and position (can be a new empty route)
        r2_idx = random.randint(0, len(new_routes))
        if r2_idx == len(new_routes):
            new_routes.append([customer])
        else:
            r2 = new_routes[r2_idx]
            insert_idx = random.randint(0, len(r2))
            r2.insert(insert_idx, customer)
            
    elif operator == 'swap':
        if len(new_routes) < 1: return new_routes
        r1_idx = random.randint(0, len(new_routes) - 1)
        r2_idx = random.randint(0, len(new_routes) - 1)
        
        if not new_routes[r1_idx] or not new_routes[r2_idx]:
            return new_routes
            
        c1_idx = random.randint(0, len(new_routes[r1_idx]) - 1)
        c2_idx = random.randint(0, len(new_routes[r2_idx]) - 1)
        
        # Swap
        new_routes[r1_idx][c1_idx], new_routes[r2_idx][c2_idx] = \
            new_routes[r2_idx][c2_idx], new_routes[r1_idx][c1_idx]
            
    # Clean up empty routes
    new_routes = [r for r in new_routes if r]
    return new_routes

def solve_sa(inst: VRPInstance, dist_matrix: Dict[Tuple[int, int], int], 
             time_limit_secs: float = 30.0, 
             initial_temp: float = 1000.0, 
             cooling_rate: float = 0.9995) -> Tuple[int, List[List[int]], float]:
    """
    Simulated Annealing metaheuristic for CVRP.
    """
    start_time = time.time()
    
    # 1. Initial solution (Clarke-Wright)
    current_routes = clarke_wright_savings(inst, dist_matrix)
    current_cost = calculate_total_cost(current_routes, inst.depot, dist_matrix)
    
    best_routes = copy.deepcopy(current_routes)
    best_cost = current_cost
    
    temp = initial_temp
    
    iters = 0
    while time.time() - start_time < time_limit_secs:
        iters += 1
        neighbor = get_neighborhood(current_routes, inst)
        
        # Check feasibility
        feasible = True
        for r in neighbor:
            if not is_feasible(r, inst):
                feasible = False
                break
                
        if not feasible:
            continue
            
        neighbor_cost = calculate_total_cost(neighbor, inst.depot, dist_matrix)
        delta = neighbor_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_routes = neighbor
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_routes = copy.deepcopy(current_routes)
                
        temp *= cooling_rate
        if temp < 0.01:
            temp = 0.01 # minimum temp
            
    solve_time = time.time() - start_time
    return best_cost, best_routes, solve_time

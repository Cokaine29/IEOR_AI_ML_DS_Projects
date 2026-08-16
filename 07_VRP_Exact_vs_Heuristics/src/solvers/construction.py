from typing import Dict, Tuple, List
from ..data_loader import VRPInstance

def clarke_wright_savings(inst: VRPInstance, dist_matrix: Dict[Tuple[int, int], int]) -> List[List[int]]:
    """
    Implements the classical Clarke-Wright Savings algorithm for CVRP.
    Returns a feasible list of routes.
    """
    # 1. Calculate savings for all pairs of customers (i, j)
    # S(i, j) = d(depot, i) + d(depot, j) - d(i, j)
    savings = []
    customers = inst.customers
    depot = inst.depot
    
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            c1 = customers[i]
            c2 = customers[j]
            s = dist_matrix[(depot, c1)] + dist_matrix[(depot, c2)] - dist_matrix[(c1, c2)]
            if s > 0:
                savings.append((s, c1, c2))
                
    # Sort savings in descending order
    savings.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Initially every customer is served by their own route
    routes = [[c] for c in customers]
    route_loads = {c: inst.demands[c] for c in customers}
    
    # Map a customer to their current route index for quick lookup
    cust_to_route = {c: i for i, c in enumerate(customers)}
    
    # 3. Merge routes based on savings
    for _, i, j in savings:
        r_i_idx = cust_to_route[i]
        r_j_idx = cust_to_route[j]
        
        # If they are already in the same route, skip
        if r_i_idx == r_j_idx:
            continue
            
        route_i = routes[r_i_idx]
        route_j = routes[r_j_idx]
        
        # To merge, i must be the last customer of its route, 
        # and j must be the first customer of its route (or vice versa).
        
        can_merge_i_j = (route_i[-1] == i and route_j[0] == j)
        can_merge_j_i = (route_j[-1] == j and route_i[0] == i)
        
        if not can_merge_i_j and not can_merge_j_i:
            continue
            
        # Check capacity
        load_i = route_loads[route_i[0]] # Just pick any customer in the route to find total load mapping
        load_j = route_loads[route_j[0]]
        
        if load_i + load_j <= inst.capacity:
            # Execute merge
            if can_merge_i_j:
                new_route = route_i + route_j
            else:
                new_route = route_j + route_i
                
            # Update data structures
            routes[r_i_idx] = new_route
            routes[r_j_idx] = [] # Empty out the old route
            
            # Update routing maps
            new_load = load_i + load_j
            for c in new_route:
                cust_to_route[c] = r_i_idx
                route_loads[c] = new_load
                
    # Filter out empty routes
    final_routes = [r for r in routes if len(r) > 0]
    return final_routes

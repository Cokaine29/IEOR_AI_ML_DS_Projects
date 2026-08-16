import math
from typing import Dict, Tuple

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> int:
    """
    Standard CVRP Euclidean distance, conventionally rounded to nearest integer
    in TSPLIB/CVRPLIB formats to avoid floating point inconsistencies across solvers.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return round(math.sqrt(dx*dx + dy*dy))

def build_distance_matrix(coords: Dict[int, Tuple[float, float]]) -> Dict[Tuple[int, int], int]:
    """Precomputes all pairwise distances."""
    dist = {}
    nodes = list(coords.keys())
    for i in nodes:
        for j in nodes:
            if i == j:
                dist[(i, j)] = 0
            else:
                dist[(i, j)] = euclidean_distance(coords[i], coords[j])
    return dist

def calculate_route_cost(route: list, depot: int, dist_matrix: Dict[Tuple[int, int], int]) -> int:
    """Calculates total cost of a single route starting and ending at depot."""
    if not route:
        return 0
    
    cost = dist_matrix[(depot, route[0])]
    for i in range(len(route) - 1):
        cost += dist_matrix[(route[i], route[i+1])]
    cost += dist_matrix[(route[-1], depot)]
    
    return cost

def calculate_total_cost(routes: list, depot: int, dist_matrix: Dict[Tuple[int, int], int]) -> int:
    """Calculates total cost of a set of routes."""
    return sum(calculate_route_cost(r, depot, dist_matrix) for r in routes)

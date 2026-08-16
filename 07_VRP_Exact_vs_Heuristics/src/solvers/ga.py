import random
import time
import copy
from typing import Dict, Tuple, List
from ..data_loader import VRPInstance
from ..cost_model import calculate_total_cost
from .construction import clarke_wright_savings

def is_feasible(route: List[int], inst: VRPInstance) -> bool:
    load = sum(inst.demands[c] for c in route)
    return load <= inst.capacity

def evaluate_chromosome(chromosome: List[int], inst: VRPInstance, dist_matrix: Dict[Tuple[int, int], int]) -> Tuple[int, List[List[int]]]:
    """
    Decodes a permutation of customers into feasible routes using a split algorithm,
    and calculates the total cost.
    """
    routes = []
    current_route = []
    current_load = 0
    
    for c in chromosome:
        if current_load + inst.demands[c] > inst.capacity:
            routes.append(current_route)
            current_route = [c]
            current_load = inst.demands[c]
        else:
            current_route.append(c)
            current_load += inst.demands[c]
            
    if current_route:
        routes.append(current_route)
        
    cost = calculate_total_cost(routes, inst.depot, dist_matrix)
    return cost, routes

def order_crossover(p1: List[int], p2: List[int]) -> List[int]:
    """Order Crossover (OX) for permutations."""
    size = len(p1)
    c1, c2 = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    child[c1:c2] = p1[c1:c2]
    
    p2_idx = c2
    child_idx = c2
    
    while -1 in child:
        if p2[p2_idx % size] not in child:
            child[child_idx % size] = p2[p2_idx % size]
            child_idx += 1
        p2_idx += 1
        
    return child

def mutate(chromosome: List[int], mutation_rate: float) -> List[int]:
    """Swap mutation."""
    mutated = list(chromosome)
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    return mutated

def get_chromosome_from_routes(routes: List[List[int]]) -> List[int]:
    chromosome = []
    for r in routes:
        chromosome.extend(r)
    return chromosome

def solve_ga(inst: VRPInstance, dist_matrix: Dict[Tuple[int, int], int], 
             time_limit_secs: float = 30.0, 
             pop_size: int = 50, 
             mutation_rate: float = 0.1) -> Tuple[int, List[List[int]], float]:
    """
    Genetic Algorithm metaheuristic for CVRP.
    """
    start_time = time.time()
    
    # 1. Initialize population
    # Include Clarke-Wright solution as one of the seeds
    cw_routes = clarke_wright_savings(inst, dist_matrix)
    cw_chrom = get_chromosome_from_routes(cw_routes)
    
    population = [cw_chrom]
    for _ in range(pop_size - 1):
        chrom = list(inst.customers)
        random.shuffle(chrom)
        population.append(chrom)
        
    best_cost = float('inf')
    best_routes = []
    
    iters = 0
    while time.time() - start_time < time_limit_secs:
        iters += 1
        
        # Evaluate population
        evaluated = []
        for chrom in population:
            cost, routes = evaluate_chromosome(chrom, inst, dist_matrix)
            evaluated.append((cost, chrom, routes))
            
            if cost < best_cost:
                best_cost = cost
                best_routes = routes
                
        # Sort and select best
        evaluated.sort(key=lambda x: x[0])
        
        # Next generation
        next_gen = []
        # Elitism: keep top 10%
        elite_count = int(pop_size * 0.1)
        next_gen.extend([x[1] for x in evaluated[:elite_count]])
        
        # Tournament selection and crossover
        while len(next_gen) < pop_size:
            # Tournament selection (size 3)
            p1 = min(random.sample(evaluated, 3), key=lambda x: x[0])[1]
            p2 = min(random.sample(evaluated, 3), key=lambda x: x[0])[1]
            
            child = order_crossover(p1, p2)
            child = mutate(child, mutation_rate)
            next_gen.append(child)
            
        population = next_gen
        
    solve_time = time.time() - start_time
    return best_cost, best_routes, solve_time

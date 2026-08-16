import os
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Generate synthetic instances since SSL downloading is blocked
# Format exactly matches CVRPLIB
SIZES = [8, 10, 12, 14, 16, 20, 30, 40, 60, 80]

def generate_instance(n_customers, seed):
    random.seed(seed)
    
    inst_name = f"Gen-n{n_customers + 1}"
    capacity = 100 if n_customers <= 30 else 200
    
    lines = []
    lines.append(f"NAME : {inst_name}")
    lines.append(f"COMMENT : Synthetic instance with {n_customers} customers")
    lines.append("TYPE : CVRP")
    lines.append(f"DIMENSION : {n_customers + 1}")
    lines.append("EDGE_WEIGHT_TYPE : EUC_2D")
    lines.append(f"CAPACITY : {capacity}")
    
    lines.append("NODE_COORD_SECTION")
    # Depot at center
    lines.append(" 1 50 50")
    for i in range(1, n_customers + 1):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        lines.append(f" {i + 1} {x} {y}")
        
    lines.append("DEMAND_SECTION")
    lines.append(" 1 0")
    for i in range(1, n_customers + 1):
        demand = random.randint(5, 25)
        lines.append(f" {i + 1} {demand}")
        
    lines.append("DEPOT_SECTION")
    lines.append(" 1")
    lines.append(" -1")
    lines.append("EOF")
    
    out_path = RAW_DIR / f"{inst_name}.vrp"
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Generated {inst_name}.vrp")

if __name__ == "__main__":
    for size in SIZES:
        generate_instance(size, seed=size)

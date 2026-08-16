import os
import vrplib
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

class VRPInstance:
    def __init__(self, name: str, capacity: int, depot: int, 
                 coords: dict, demands: dict, node_list: list):
        self.name = name
        self.capacity = capacity
        self.depot = depot
        self.coords = coords
        self.demands = demands
        self.nodes = node_list
        self.customers = [n for n in self.nodes if n != depot]
        self.n_customers = len(self.customers)

def load_vrp_instance(instance_name: str) -> VRPInstance:
    """Downloads (if needed) and parses a CVRPLIB instance using vrplib."""
    filepath = RAW_DIR / f"{instance_name}.vrp"
    
    if not filepath.exists():
        print(f"Downloading {instance_name}...")
        vrplib.download_instance(instance_name, str(filepath))
    
    inst = vrplib.read_instance(str(filepath))
    
    name = inst.get('name', instance_name)
    capacity = inst.get('capacity', 0)
    
    # coords is a numpy array (N, 2). Node IDs usually 1..N or 0..N-1 depending on vrplib.
    # vrplib usually keeps 0-indexed or matches the instance. Let's force 1-indexed to be safe.
    node_coords = inst.get('node_coord', [])
    demands_raw = inst.get('demand', [])
    
    coords = {}
    demands = {}
    nodes = []
    depot = 1 # We will treat the first node in the array as the depot (usually index 0, so node 1)
    
    for i, (x, y) in enumerate(node_coords):
        node_id = i + 1
        coords[node_id] = (x, y)
        demands[node_id] = demands_raw[i]
        nodes.append(node_id)
        
    return VRPInstance(name, capacity, depot, coords, demands, nodes)

if __name__ == "__main__":
    # Test loader with a very small instance
    inst = load_vrp_instance("P-n16-k8")
    print(f"Loaded {inst.name}: {inst.n_customers} customers, Capacity: {inst.capacity}")

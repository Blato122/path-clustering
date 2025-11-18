from pathlib import Path
import janux as jx
import time

"""
Generate routes CSV for all networks found in a given directory.

The target use case is generating csv (origin,destination,path,other-features,...) for all
29 networks from the URB Networks dataset. 

Assumptions:
- Networks can be found under ../data relative to this script by default.
- Results are written to ../results relative to this script by default.
"""

NUM_SAMPLES = 5       # Number of samples to generate per OD
NUMBER_OF_PATHS = 2     # Number of paths to find for each origin-destination pair
BETA = -3.0             # Beta parameter for the path generation
MAX_ITERATIONS = 5    # Sampler safeguard
SEED = 42               # For reproducibility

RESULTS_DIR = Path("/results")  # Output folder

def find_networks(base_dir: Path) -> dict[str, Path]:
    """
    Look for network folders under base_dir. 
    
    A network directory is valid if it has:
    - <name>.con.xml
    - <name>.edg.xml
    - <name>.rou.xml
    - od_<name>.json
    """
    out = {}
    if not base_dir.exists():
        return out
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        con = d / f"{name}.con.xml"
        edg = d / f"{name}.edg.xml"
        rou = d / f"{name}.rou.xml"
        od = d / f"od_{name}.json"
        if con.exists() and edg.exists() and rou.exists() and od.exists():
            out[name] = d
    return out

def main():
    this_file = Path(__file__).resolve()
    # .../path-clustering/scripts/generate_csv_routes.py -> .../path-clustering
    repo_root = this_file.parents[1]
    data_dir = repo_root / "data"
    results_dir = repo_root / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    network_dict: dict[str, Path] = find_networks(data_dir)
    if not network_dict:
        print("No networks found. Check the directory structure.")
        return
    
    path_gen_kwargs = {
        "random_seed": SEED,
        "num_samples": NUM_SAMPLES,
        "number_of_paths": NUMBER_OF_PATHS,
        "beta": BETA,
        "verbose": False, # Print the progress of the path generation
    }

    for name, dir in network_dict.items():
        print(f"\nProcessing network: {name}")

        connection_file_path = f"{dir}/{name}.con.xml"
        edge_file_path = f"{dir}/{name}.edg.xml"
        route_file_path = f"{dir}/{name}.rou.xml"

        ods = jx.utils.read_json(f"{dir}/od_{name}.json")
        origins = ods["origins"]
        destinations = ods["destinations"]
        
        start_time = time.time()
        # Generate network
        network = jx.build_digraph(connection_file_path, edge_file_path, route_file_path)
        # Generate routes
        try:
            routes = jx.basic_generator(
                network, 
                origins, 
                destinations,
                max_iterations=MAX_ITERATIONS,
                as_df=True,
                calc_free_flow=True,
                **path_gen_kwargs
            )
        except AssertionError as e:
            print(f"Skipping network {name}: {e}")
            continue

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        # Save the routes to a CSV file    
        csv_save_path = results_dir / f"{name}_routes.csv"
        routes.to_csv(csv_save_path, index=False)
        print(f"Saved routes to: {csv_save_path}")

if __name__ == "__main__":
    main()
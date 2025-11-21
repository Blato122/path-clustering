from pathlib import Path
import janux as jx
import time
import pandas as pd

"""
Generate routes CSV for all networks found in a given directory.

The target use case is generating csv (origin,destination,path,other-features,...) for all
29 networks from the URB Networks dataset. 

Assumptions:
- Networks can be found under ../data relative to this script by default.
- Results are written to ../results/routes relative to this script by default.
"""

NUM_SAMPLES = 50       # Number of samples to generate per OD
NUMBER_OF_PATHS = 3     # Number of paths to find for each origin-destination pair
BETA = -3.0             # Beta parameter for the path generation
MAX_ITERATIONS = 50    # Sampler safeguard
SEED = 42               # For reproducibility

RESULTS_DIR = Path("/results/routes")  # Output folder

def find_networks(base_dir: Path) -> dict[str, Path]:
    """
    Look for network folders under base_dir. 
    
    A network directory is valid if it has:
    - <name>.con.xml
    - <name>.edg.xml
    - <name>.rou.xml
    - od_<name>.json
    - agents.csv
    """
    out = {}
    if not base_dir.exists():
        return out
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        files = [
            d / f"{name}.con.xml",
            d / f"{name}.edg.xml",
            d / f"{name}.rou.xml",
            d / f"od_{name}.json",
            d / "agents.csv"
        ]
        if all(file.exists() for file in files):
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
        "verbose": True, # Print the progress of the path generation
    }

    for name, dir in network_dict.items():
        print(f"\nProcessing network: {name}")

        connection_file_path = f"{dir}/{name}.con.xml"
        edge_file_path = f"{dir}/{name}.edg.xml"
        route_file_path = f"{dir}/{name}.rou.xml"

        ods = jx.utils.read_json(f"{dir}/od_{name}.json")
        origins = ods["origins"]
        destinations = ods["destinations"]

        # 300 origins and 300 destinations -> 90000 OD pairs -> too long to process!
        # instead, take agents.csv (order of hundreds)
        agents = pd.read_csv(f"{dir}/agents.csv")
        
        all_routes = []
        start_time = time.time()
        network = jx.build_digraph(connection_file_path, edge_file_path, route_file_path)

        for o_id, d_id in zip(agents["origin"], agents["destination"]):
            try:
                # print(o_id, origins[o_id], d_id, destinations[d_id])
                routes = jx.basic_generator(
                    network, 
                    [origins[o_id]], 
                    [destinations[d_id]],
                    max_iterations=MAX_ITERATIONS,
                    as_df=True,
                    calc_free_flow=True,
                    **path_gen_kwargs
                )
                all_routes.append(routes)
            except AssertionError as e:
                print(f"Skipping network {name}: {e}")
                continue

        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # Save the routes to a CSV file    
        if all_routes:
            all_routes_merged = pd.concat(all_routes)
            csv_save_path = results_dir / f"{name}_routes.csv"
            all_routes_merged.to_csv(csv_save_path, index=False)
            print(f"Saved routes to: {csv_save_path}")
        else:
            print(f"No routes generated for {name}")

if __name__ == "__main__":
    main()
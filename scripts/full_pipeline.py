from pathlib import Path
import janux as jx
import time
import pandas as pd
from lxml import etree
import osmnx as ox
import math

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

def get_osm_id_from_sumo(sumo_id):
    """
    Convert SUMO edge id formats back to original OSM way id:
    - "-123456"    → 123456
    - "123456#1"   → 123456
    - "-78910#2"   → 78910
    """
    # Ignore internal SUMO edges
    if sumo_id.startswith(":") or not any(c.isdigit() for c in sumo_id): # or make sure all digits?
        return None
    
    s = sumo_id.lstrip("-")
    
    if "#" in sumo_id:
        s = sumo_id.split("#")[0]

    try:
        return int(s)
    except ValueError:
        return None  # internal SUMO edge, cannot map to OSM
    
def load_sumo_edges_from_edg(edg_file):
    tree = etree.parse(edg_file)
    root = tree.getroot()

    records = []

    for edge in root.xpath("//edge"):
        edge_id   = edge.get("id")
        from_node = edge.get("from")
        to_node   = edge.get("to")
        priority  = edge.get("priority")
        edge_type = edge.get("type")
        speed     = edge.get("speed")
        num_lanes = edge.get("numLanes")
        allow     = edge.get("allow")
        disallow  = edge.get("disallow")
        shape     = edge.get("shape")

        speed = float(speed) if speed else None
        num_lanes = int(num_lanes) if num_lanes else None
        priority = int(priority) if priority else None

        # Parse shape: "x1,y1 x2,y2 ..." -> [(x1,y1), (x2,y2), ...]
        if shape:
            coords = [
                tuple(map(float, p.split(",")))
                for p in shape.strip().split(" ")
            ]
        else:
            coords = None

        records.append({
            "sumo_id": edge_id,
            "from": from_node,
            "to": to_node,
            "priority": priority,
            "type": edge_type,
            "speed": speed,
            "lanes": num_lanes,
            "allow": allow,
            "disallow": disallow,
            "shape": coords,
        })

    return pd.DataFrame(records)

def compute_length_from_shape(coords):
    if not coords or len(coords) < 2:
        return None
    total = 0
    for i in range(1, len(coords)):
        x0, y0 = coords[i-1]
        x1, y1 = coords[i]
        total += math.hypot(x1 - x0, y1 - y0)
    return total

def get_traffic_light_nodes(net_file):
    tree = etree.parse(net_file)
    root = tree.getroot()
    tls_nodes = set() # junction ids

    for junc in root.xpath("//junction[@type='traffic_light']"):
        tls_nodes.add(junc.get("id"))

    return tls_nodes

def generate_csv_routes(name: str | Path, dir: str | Path, path_gen_kwargs: dict, results_dir: str | Path) -> None:
    """
    Generate paths for each OD pair in agents.csv using JanuX.
    Results are saved in path-clustering/results directory.
    """
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

def generate_feature_files(name: str, net_dir: Path, osm_dir: Path, results_dir: Path) -> None:
    print("\n=== Extracting SUMO edges for {name} ===")

    edg_file = net_dir / f"{name}.edg.xml"
    net_file = net_dir / f"{name}.net.xml"
    osm_file = osm_dir / f"{name}.osm"

    df_sumo = load_sumo_edges_from_edg(edg_file)
    df_sumo["osmid"] = df_sumo["sumo_id"].apply(get_osm_id_from_sumo) # osmid also appears in the osm df
    traffic_light_nodes = get_traffic_light_nodes(net_file)
    df_sumo["has_traffic_light"] = df_sumo["to"].isin(traffic_light_nodes).astype(int)
    # df_sumo.to_csv("sumo_edges.csv", index=False)
    # print("SUMO edges extracted to sumo_edges.csv")
    print(df_sumo.head())

    print("\n=== Loading OSM with OSMnx ===")

    G = ox.graph_from_xml(osm_file)
    _ , edges_osm = ox.graph_to_gdfs(G)
    # some osmid values can be lists - explode so as not to lose any edges
    # reset_index() because after exploding, new rows share the same index
    edges_osm_exploded = edges_osm.explode("osmid").reset_index(drop=True)

    # ? flatten list attributes if they appear
    osm_cols_to_keep = ["highway", "maxspeed", "lanes"] # add more?
    for col in osm_cols_to_keep:
        if col in edges_osm_exploded.columns:
            edges_osm_exploded[col] = edges_osm_exploded[col].apply(lambda x: x[0] if isinstance(x, list) else x)

    # edges_osm.to_csv("osm_edges.csv", index=False)
    # print("OSM edges extracted to osm_edges.csv")
    print(edges_osm_exploded[["osmid", *osm_cols_to_keep]].head())

    print("\n=== Merging SUMO edges with OSM features ===")

    # if multiple edges have the same id, take the 1st one
    edges_osm_unique = edges_osm_exploded.groupby("osmid")[osm_cols_to_keep].first().reset_index()

    # Left join: keep all SUMO edges, attach OSM info where matches exist
    df_merged = pd.merge(
        df_sumo,
        edges_osm_unique,
        on="osmid",
        how="left"
    )

    out_path = results_dir / f"{name}_merged_edges.csv"
    df_merged.to_csv(out_path, index=False)

    print(f"Merged SUMO + OSM features too {out_path}")
    print(df_merged.head())

def main():
    this_file = Path(__file__).resolve()
    # .../path-clustering/scripts/generate_csv_routes.py -> .../path-clustering
    repo_root = this_file.parents[1]
    data_dir = repo_root / "data"
    results_dir = repo_root / "results"
    routes_dir = results_dir / "routes"
    merged_edges_dir = results_dir / "merged_edges"
    osm_dir = repo_root / "osm"
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
        # use extended generator? + visualize?
        # dir same as data_dir?
        generate_csv_routes(name, dir, path_gen_kwargs, routes_dir)
        generate_feature_files(name, osm_dir, data_dir, merged_edges_dir)

if __name__ == "__main__":
    main()
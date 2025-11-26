from pathlib import Path
import janux as jx
import time
import pandas as pd
from lxml import etree
import osmnx as ox
import math

"""
1. Generate routes CSV for all networks found in a given directory.
    Requires:
        - <name>.con.xml
        - <name>.edg.xml
        - <name>.rou.xml
        - od_<name>.json
        - agents.csv
    Outputs:
        - <name>_routes.csv

2. Generate a feature file with SUMO edges as rows and OSM + SUMO columns.
    Requires:
        - <name>_routes.
        - <name>.edg.xml
        - <name>.net.xml
        - <name>.csm
    Outputs:
        - <name>_merged_edges.csv

3. Enrich merged edges file with newly calculated edge features.
    Requires:
        - <name>_routes.csv
        - <name>_merged_edges.csv
    Outputs:
        - <name>_enriched_routes.csv
"""

NUM_SAMPLES = 50       # Number of samples to generate per OD
NUMBER_OF_PATHS = 3     # Number of paths to find for each origin-destination pair
BETA = -3.0             # Beta parameter for the path generation
MAX_ITERATIONS = 50    # Sampler safeguard
SEED = 42               # For reproducibility

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

def generate_csv_routes(name: str, net_dir: Path, path_gen_kwargs: dict, results_dir: Path) -> None:
    """
    Generate paths for each OD pair in agents.csv using JanuX.
    Results are saved in path-clustering/results/routes directory.
    """
    print(f"\nProcessing network: {name}")

    required_files = [
        net_dir / f"{name}.con.xml",
        net_dir / f"{name}.edg.xml",
        net_dir / f"{name}.rou.xml",
        net_dir / f"od_{name}.json",
        net_dir / "agents.csv"
    ]
    
    if not all(f.exists() for f in required_files):
        print(f"Skipping {name}: missing required files for route generation")
        return

    con_file, edg_file, rou_file, ods_file, agents_file = required_files

    ods = jx.utils.read_json(ods_file)
    agents = pd.read_csv(agents_file)
    origins = ods["origins"]
    destinations = ods["destinations"]

    all_routes = []
    start_time = time.time()
    network = jx.build_digraph(str(con_file), str(edg_file), str(rou_file))

    # 300 origins and 300 destinations -> 90000 OD pairs -> too long to process!
    # instead, take agents.csv (order of hundreds)
    for o_id, d_id in zip(agents["origin"], agents["destination"]):
        try:
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

def generate_feature_file(name: str, net_dir: Path, osm_dir: Path, results_dir: Path) -> None:
    print(f"\n=== Extracting SUMO edges for {name} ===")

    edg_file = net_dir / f"{name}.edg.xml"
    net_file = net_dir / f"{name}.net.xml"
    osm_file = osm_dir / f"{name}.osm"

    if not edg_file.exists() or not net_file.exists() or not osm_file.exists():
        print(f"Skipping {name}: missing files")
        return

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
    df_merged["length"] = df_merged["shape"].apply(compute_length_from_shape)

    out_path = results_dir / f"{name}_merged_edges.csv"
    df_merged.to_csv(out_path, index=False)

    print(f"Merged SUMO + OSM features too {out_path}")
    print(df_merged.head())

def enrich_routes(name: str, routes_dir: Path, merged_edges_dir: Path, output_dir: Path) -> None:
    """Compute route features from edge data."""
    print(f"\n=== Enriching routes for {name} ===")
    
    routes_file = routes_dir / f"{name}_routes.csv"
    edges_file = merged_edges_dir / f"{name}_merged_edges.csv"

    if not routes_file.exists() or not edges_file.exists():
        print(f"Skipping {name}: missing input files")
        return
    
    routes = pd.read_csv(routes_file)
    edge_features = pd.read_csv(edges_file)
    
    # Prepare edge data
    NUMERIC_FIELDS = ["length", "speed", "lanes"]
    for col in NUMERIC_FIELDS:
        if col in edge_features.columns:
            edge_features[col] = pd.to_numeric(edge_features[col], errors="coerce")
    
    edge_features = edge_features.set_index("sumo_id")
    
    def clean_hwy(x):
        return x.split(".")[-1] if isinstance(x, str) else x
    
    edge_features["highway_clean"] = edge_features["highway"].apply(clean_hwy)

    # Compute features for each path
    HIGHWAY_VALUES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"]
    
    def compute_features_for_path(path_str):
        edges = path_str.split(",")
        # Only select the rows that make up the path (index is sumo_id)
        df = edge_features.loc[edge_features.index.intersection(edges)] 
        
        feature_dict = {
            "num_edges": len(edges),
            "total_length": df["length"].sum(skipna=True),
            "mean_speed": df["speed"].mean(skipna=True),
        }
        
        total_len = feature_dict["total_length"]
        for hv in HIGHWAY_VALUES:
            feature_dict[f"pct_{hv}"] = (
                # Select rows with a given highway type, take the length column and sum the values
                df.loc[df["highway_clean"] == hv, "length"].sum() / total_len
                if total_len > 0 else 0.0
            )
        
        feature_dict["num_traffic_lights"] = df["has_traffic_light"].sum()
        return feature_dict
    
    # Enrich and save
    enriched = pd.concat([routes, routes["path"].apply(compute_features_for_path).apply(pd.Series)], axis=1)
    out_path = output_dir / f"{name}_routes_enriched.csv"
    enriched.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

def main():
    this_file = Path(__file__).resolve()
    # .../path-clustering/scripts/generate_csv_routes.py -> .../path-clustering
    repo_root = this_file.parents[1]

    # Network data
    data_dir = repo_root / "data"
    osm_dir = repo_root / "osm"

    # Results
    results_dir = repo_root / "results"
    routes_dir = results_dir / "routes"
    merged_edges_dir = results_dir / "merged_edges"
    enriched_routes_dir = results_dir / "enriched_routes"

    results_dir.mkdir(parents=True, exist_ok=True)
    routes_dir.mkdir(parents=True, exist_ok=True)
    merged_edges_dir.mkdir(parents=True, exist_ok=True)
    enriched_routes_dir.mkdir(parents=True, exist_ok=True)
    
    path_gen_kwargs = {
        "random_seed": SEED,
        "num_samples": NUM_SAMPLES,
        "number_of_paths": NUMBER_OF_PATHS,
        "beta": BETA,
        "verbose": True, # Print the progress of the path generation
    }

    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        # use extended generator? + visualize?
        generate_csv_routes(name, data_dir, path_gen_kwargs, routes_dir) # generates routes
        generate_feature_file(name, data_dir, osm_dir, merged_edges_dir) # generates merged_edges
        enrich_routes(name, routes_dir, merged_edges_dir, enriched_routes_dir) # uses routes and merged_edges to generate enriched_routes

if __name__ == "__main__":
    main()
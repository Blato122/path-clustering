from pathlib import Path
import janux as jx
import time
import pandas as pd
from lxml import etree
import osmnx as ox
import math
import argparse

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

    Ignores internal edges:
    Usually very short and describe micro-movements inside a junction.
    Normal edge:     A ----edge1----> [Junction] ----edge2----> B
    Internal edge:   A ----edge1----> :----internal----> edge2----> B

    OSM Way 1020078541 (a road)
    ├─ Node A (lat/lon)
    ├─ Node B (lat/lon)
    ├─ Node C (lat/lon)
    └─ Node D (lat/lon)

    SUMO Edge "-1020078541#0":  Node A → Node B  (osmid: 1020078541)
    SUMO Edge "-1020078541#1":  Node B → Node C  (osmid: 1020078541)
    SUMO Edge "1020078541#0":   Node C → Node D  (osmid: 1020078541, forward)

    Multiple SUMO edges can share the same osmid (they're segments of the same OSM way)
    """
    # Ignore internal SUMO edges
    if sumo_id.startswith(":") or not any(c.isdigit() for c in sumo_id):
        return None
    
    base = sumo_id.split("#")[0] # "-102015882#0" → "-102015882"
    base = base.lstrip("-") # "-102015882" → "102015882"
    
    try:
        return int(base)
    except ValueError:
        return None
    
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

    # //junction - find all <junction> elements anywhere in the document
    # [@type='traffic_light'] - where attribute type is traffic_light
    # e.g. <junction id="241790299" type="traffic_light" x="2964.06" y="4630.25" ...>
    for junc in root.xpath("//junction[@type='traffic_light']"):
        tls_nodes.add(junc.get("id"))

    # e.g. <tlLogic id="370504753" type="static" programID="0" offset="0">
    for tl in root.xpath("//tlLogic"):
        tls_nodes.add(tl.get("id"))

    return tls_nodes

def clean_hwy(x):
    if pd.isna(x):
        return "unknown"
    return x.split(".")[-1] if isinstance(x, str) else x

def count_turns(df, turn_threshold_deg=30):
    """Count significant direction changes for a given path"""

    # bearings = df["bearing"].dropna().values
    # if len(bearings) < 2:
    #     return 0
    
    # turns = 0
    # for i in range(1, len(bearings)):
    #     angle = abs(bearings[i] - bearings[i-1]) # e.g. 90 - 45 (already in degrees)
    #     if angle > 180:
    #         angle = 360 - angle
    #     if angle > turn_threshold_deg:
    #         turns += 1

    diffs = df["bearing"].dropna().diff().abs()
    diffs = diffs.where(diffs <= 180, 360 - diffs)
    return (diffs > turn_threshold_deg).sum()

def calculate_circuity(df, edges, total_len):
    """
    Circuity: actual_road_length / straight_line_length

    1.0 = perfectly straight road
    2.0 = road is 2x longer than the straight line (very winding)
    Range: 1-inf
    
    It tells us how much the road deviates from the straight line
    connecting its first and last nodes.

    Circuity of a road might be connected to its type, for example, highways
    might have values close to 1.0, while mountain roads will likely have much
    higher values.

    Straightness = 1 / Circuity
    Circuity = Sinuosity
    """

    first_edge_id = edges[0]
    last_edge_id = edges[-1]

    first_node = df.loc[first_edge_id, "from"] 
    last_node = df.loc[last_edge_id, "to"]
    
    straight_len = 

    circuity = straight_len / total_len
    return circuity

def generate_csv_routes(name: str, net_dir: Path, path_gen_kwargs: dict, results_dir: Path) -> None:
    """
    Generate paths for each OD pair in agents.csv using JanuX.
    Results are saved in path-clustering/results/routes directory.
    """
    print(f"\nProcessing network: {name}")

    required_files = [
        net_dir / name / f"{name}.con.xml",
        net_dir / name / f"{name}.edg.xml",
        net_dir / name / f"{name}.rou.xml",
        net_dir / name / f"od_{name}.json",
        net_dir / name / "agents.csv"
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

    edg_file = net_dir / name / f"{name}.edg.xml"
    net_file = net_dir / name / f"{name}.net.xml"
    osm_file = osm_dir / f"{name}.osm"

    if not edg_file.exists() or not net_file.exists() or not osm_file.exists():
        print(f"Skipping {name}: missing files")
        return

    df_sumo = load_sumo_edges_from_edg(edg_file)
    df_sumo["osmid"] = df_sumo["sumo_id"].apply(get_osm_id_from_sumo) # osmid also appears in the osm df
    df_sumo["has_traffic_light"] = df_sumo["to"].isin(get_traffic_light_nodes(net_file)).astype(int)
    df_sumo["speed"] *= 3.6 # maxspeed, originally in m/s

    print("\n=== Loading OSM with OSMnx ===")

    # Surface and lit are not included by default!
    ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + ["surface", "lit"]
    # In OSM files, each <way> contains a sequence of node references: <nd ref="..."/>, e.g.
    # <way id="22561497" version="11" timestamp="2021-09-11T20:13:25Z">
    # <nd ref="241790699"/>
    # Each <nd> references a <node> with actual coordinates, e.g. 
    # <node id="241790699" version="2" timestamp="2010-05-01T09:41:12Z" lat="48.8484445" lon="1.8765013"/>
    # OSMnx converts OSM ways to WKT (Well-Known Text) and stores the shape in the "geometry" column
    # And stores the actual road length in meters as "length"
    G = ox.graph_from_xml(osm_file)
    # Add compass direction (angle) for each edge (between points A and B)
    # Ignores intermediate points (works on straight lines)
    # Bearing changing frequently -> winding route, consistent bearing -> straight route
    G = ox.bearing.add_edge_bearings(G)
    nodes_osm , edges_osm = ox.graph_to_gdfs(G)
    # Some osmid values can be lists - explode so as not to lose any edges
    # reset_index() because after exploding, new rows share the same index
    edges_osm_exploded = edges_osm.explode("osmid").reset_index(drop=True)
    # For calculating circuity (node coords required):
    nodes_osm = nodes_osm.reset_index()
    # Extract just the coords:
    node_coords = nodes_osm.set_index("osmid")[["x", "y"]] # x=lon, y=lat

    # Already in SUMO: lanes, highway (type), maxspeed (speed)
    # Keep the length instead of calculating it by hand (like before)
    # OSMnx length accounts for Earth's curvature instead of calculating simple Euclidean distances
    osm_cols_to_keep = ["surface", "bridge", "tunnel", "lit", "bearing", "length"]
    for col in osm_cols_to_keep:
        if col in edges_osm_exploded.columns:
            edges_osm_exploded[col] = edges_osm_exploded[col].apply(lambda x: x[0] if isinstance(x, list) else x)

    print("\n=== Merging SUMO edges with OSM features ===")

    # if multiple edges have the same id, take the 1st one ???
    edges_osm_unique = edges_osm_exploded.groupby("osmid")[osm_cols_to_keep].first().reset_index()
    edges_osm_exploded["from_lon"] = edges_osm_exploded["u"].map(node_coords["x"])
    edges_osm_exploded["from_lat"] = edges_osm_exploded["u"].map(node_coords["y"])
    edges_osm_exploded["to_lon"] = edges_osm_exploded["v"].map(node_coords["x"])
    edges_osm_exploded["to_lat"] = edges_osm_exploded["v"].map(node_coords["y"])

    # Left join: keep all SUMO edges, attach OSM info where matches exist
    df_merged = pd.merge(
        df_sumo,
        edges_osm_unique,
        on="osmid",
        how="left"
    )
    # After merging, handle missing lengths:
    df_merged["length"] = df_merged["length"].fillna(
        df_merged["shape"].apply(compute_length_from_shape)
    )
    df_merged = df_merged.drop(columns=["shape"]) # big and not used in enrich anyway

    out_path = results_dir / f"{name}_merged_edges.csv"
    df_merged.to_csv(out_path, index=False)

def enrich_routes(name: str, routes_dir: Path, merged_edges_dir: Path, output_dir: Path) -> None:
    """Compute route features from edge data"""
    print(f"\n=== Enriching routes for {name} ===")
    
    routes_file = routes_dir / f"{name}_routes.csv"
    edges_file = merged_edges_dir / f"{name}_merged_edges.csv"

    if not routes_file.exists() or not edges_file.exists():
        print(f"Skipping {name}: missing input files")
        return
    
    routes = pd.read_csv(routes_file)
    edge_features = pd.read_csv(edges_file)
    
    # Prepare edge data
    NUMERIC_FIELDS = ["length", "speed", "lanes", "priority", "bearing"]
    for col in NUMERIC_FIELDS:
        if col in edge_features.columns:
            edge_features[col] = pd.to_numeric(edge_features[col], errors="coerce")
    
    edge_features = edge_features.set_index("sumo_id")    
    edge_features["type_clean"] = edge_features["type"].apply(clean_hwy)
    
    def compute_features_for_path(path_str):
        edges = path_str.split(",")
        # Only select the rows that make up the path (index is sumo_id)
        df = edge_features.loc[edge_features.index.intersection(edges)] 

        # Handle empty dataframe
        if df.empty:
            return {
                "num_edges": len(edges),
                "total_length": 0.0,
                "mean_speed": 0.0,
                "pct_paved": 0.0,
                "pct_unpaved": 0.0,
                **{f"pct_{hv}": 0.0 for hv in ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"]},
                "pct_lit": 0.0,
                "bearing_variance": 0.0,
                "mean_bearing": 0.0,
                "num_turns": 0,
                "num_bridges": 0,
                "num_tunnels": 0,
                "speed_std": 0.0,
                "speed_range": 0.0,
                "lane_changes": 0.0,
                "priority_changes": 0.0,
                "num_traffic_lights": 0,
            }
        
        feature_dict = {
            "num_edges": len(edges),
            "total_length": df["length"].sum(skipna=True),
            "mean_speed": df["speed"].mean(skipna=True),
        }
        total_len = feature_dict["total_length"]
        
        PAVED_SURFACES = ["asphalt", "paved", "concrete", "bricks", "sett", "grass_paver", "paving_stones"]
        feature_dict["pct_paved"] = (
            df.loc[df["surface"].isin(PAVED_SURFACES), "length"].sum() / total_len
            if total_len > 0 else 0.0
        )

        UNPAVED_SURFACES = ["sand", "gravel", "dirt", "compacted"]
        feature_dict["pct_unpaved"] = (
            df.loc[df["surface"].isin(UNPAVED_SURFACES), "length"].sum() / total_len
            if total_len > 0 else 0.0
        )

        HIGHWAY_VALUES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"]
        for hv in HIGHWAY_VALUES:
            feature_dict[f"pct_{hv}"] = (
                # Select rows with a given highway type, take the length column and sum the values
                df.loc[df["type_clean"] == hv, "length"].sum() / total_len
                if total_len > 0 else 0.0
            )

        feature_dict["pct_lit"] = (
            df.loc[df["lit"] == "yes", "length"].sum() / total_len
            if total_len > 0 else 0.0
        )

        # Bearing variance (high = winding route, low = straight) ~ sinuosity?
        feature_dict["bearing_std"] = df["bearing"].std(skipna=True)
        # use diff instead? df["bearing"].diff() and above 30 and sum number
        feature_dict["num_turns"] = count_turns(df) # copy not needed, df isn't modified
        feature_dict["mean_circuity"] = calculate_circuity(df) # copy not needed, df isn't modified 
        feature_dict["num_bridges"] = df["bridge"].notna().sum(skipna=True)
        feature_dict["num_tunnels"] = df["tunnel"].notna().sum(skipna=True)
        feature_dict["speed_std"] = df["speed"].std(skipna=True)
        feature_dict["speed_range"] = df["speed"].max(skipna=True) - df["speed"].min(skipna=True)
        feature_dict["lane_changes"] = df["lanes"].diff().abs().sum(skipna=True)
        feature_dict["priority_changes"] = df["priority"].diff().abs().sum(skipna=True)
        # feature_dict["yield_priority_changes"] = df["priority"].diff().abs().sum(skipna=True)
        feature_dict["num_traffic_lights"] = df["has_traffic_light"].sum(skipna=True)
        return feature_dict
    
    # Enrich and save
    enriched = pd.concat([routes, routes["path"].apply(compute_features_for_path).apply(pd.Series)], axis=1)
    out_path = output_dir / f"{name}_routes_enriched.csv"
    enriched.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline to generate and enrich route data from SUMO networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all pipeline stages (routes, features, enrich)"
    )
    parser.add_argument(
        "--routes", 
        action="store_true", 
        help="Generate routes CSV files using JanuX"
    )
    parser.add_argument(
        "--features", 
        action="store_true", 
        help="Generate merged edge feature files (SUMO + OSM)"
    )
    parser.add_argument(
        "--enrich", 
        action="store_true", 
        help="Enrich routes with calculated features"
    )
    
    args = parser.parse_args()
    if not any(vars(args).values()):
        print("No arguments provided. Use -h for help.")
        return
    if args.all:
        args.routes = args.features = args.enrich = True

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

    for d in [results_dir, routes_dir, merged_edges_dir, enriched_routes_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    path_gen_kwargs = {
        "random_seed": SEED,
        "num_samples": NUM_SAMPLES,
        "number_of_paths": NUMBER_OF_PATHS,
        "beta": BETA,
        "verbose": True, # Print the progress of the path generation
    }

    stages = [
        name for name, enabled in [
            ("routes", args.routes),
            ("features", args.features),
            ("enrich", args.enrich),
        ] if enabled
    ]
    stage_str = ", ".join(stages) if stages else ""

    print(
        f"\n{'='*60}\n"
        f"Running pipeline stages: {stage_str}\n"
        f"{'='*60}\n"
    )

    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if args.routes: generate_csv_routes(name, data_dir, path_gen_kwargs, routes_dir) # generates routes
        if args.features: generate_feature_file(name, data_dir, osm_dir, merged_edges_dir) # generates merged_edges
        if args.enrich: enrich_routes(name, routes_dir, merged_edges_dir, enriched_routes_dir) # uses routes and merged_edges to generate enriched_routes

if __name__ == "__main__":
    main()
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
    
def load_sumo_nodes(nod_file: Path) -> dict:
    """Parses .nod.xml to get a mapping of node_id -> (x, y)."""
    tree = etree.parse(nod_file)
    root = tree.getroot()
    node_coords = {}
    for node in root.xpath("//node"):
        nid = node.get("id")
        x = node.get("x")
        y = node.get("y")
        if x and y:
            node_coords[nid] = (float(x), float(y))
    return node_coords

def load_sumo_edges(edg_file: Path) -> pd.DataFrame:
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

def count_turns(df, turn_threshold_deg=30):
    """
    Count significant direction changes for a given path.
    
    df is a dataframe with rows corresponding only to edges
    present in a given path and columns with features
    """

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
    diffs = diffs.where(diffs <= 180, 360 - diffs) # replace values where the condition is false
    return (diffs > turn_threshold_deg).sum()

def calculate_circuity(df, total_len):
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

    ===

    df is a dataframe with rows corresponding only to edges
    present in a given path and columns with features

    edges are the edges that make up a path in a correct order
    (for edges [A, B], A comes before B in a path)

    """

    if total_len is None or total_len <= 0 or df.empty:
        return None

    # Use first/last present in df (guaranteed to follow path order because of reindex)
    x1 = float(df["start_x"].iloc[0])
    y1 = float(df["start_y"].iloc[0])
    x2 = float(df["end_x"].iloc[-1])
    y2 = float(df["end_y"].iloc[-1])
    
    # Euclidean distance for straight line (consistent with SUMO projection)
    straight_len = math.hypot(x2-x1, y2-y1)

    if straight_len == 0:
        return None

    return total_len / straight_len

def compute_sumo_length(shape: list[tuple[float, float]]) -> float:
    """
    Computes length of a SUMO edge based on its shape.
    Because SUMO shape data is in meters (projected x/y coordinates instead of lat/lon, e.g. 123.02, 652.14),
    we calculate just the Euclidean distance.
    """
    total = 0.0
    for i in range(len(shape)-1):
        # geopy expects (lat, lon), SUMO shape is (lon, lat)
        x1, y1 = shape[i]
        x2, y2 = shape[i+1]
        # Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)
        total += math.hypot(x2-x1, y2-y1)
    return total

def compute_sumo_bearing(shape: list[tuple[float, float]]) -> float | None:
    """
    Computes bearing - the general direction of the edge (in degrees).
    Only uses the straight line direction between the first and the last node.
    0 = north, 90 = east, 180 = south, 270 = west
    """
    start = shape[0]
    end = shape[-1]
    dx = end[0] - start[0] # change in easting
    dy = end[1] - start[1] # change in northing
    # atan2(dx, dy) gives angle from north (Y-axis) clockwise
    angle = math.degrees(math.atan2(dx, dy))
    return (angle + 360) % 360

def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0

def generate_csv_routes(name: str, net_dir: Path, path_gen_kwargs: dict, results_dir: Path) -> None:
    """
    Generate paths for each OD pair in agents.csv using JanuX.
    Results are saved in path-clustering/results/routes directory.
    """
    print(f"\n=== Generating routes for {name} ===")

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
    print(f"\n=== Generating feature file for {name} ===")

    edg_file = net_dir / name / f"{name}.edg.xml"
    nod_file = net_dir / name / f"{name}.nod.xml"
    net_file = net_dir / name / f"{name}.net.xml"
    osm_file = osm_dir / f"{name}.osm"

    if not edg_file.exists() or not net_file.exists() or not osm_file.exists():
        print(f"Skipping {name}: missing files")
        return

    # Final approach:

    # OSM Way - represents a "logical road entity" (e.g. main street), keyed by osmid.
    # Such OSM Way objects may consist of multiple edges - straight lines connecting consecutive nodes - 
    # smallest possible segments. Edges are keyed by their start and target nodes as well as an extra number. 
    # However, OSMnx simplifies edges. What this means:
    """
    This simplifies the graph's topology by removing all nodes that are not intersections or dead-ends, 
    by creating an edge directly between the end points that encapsulate them while retaining the 
    full geometry of the original edges, saved as a new geometry attribute on the new edge.
    """
    # SUMO - some internal splitting algorithm, neither an edge nor a Way.
    # However, SUMO segment ids generally correspond to OSM Way ids, it's just that SUMO divides 
    # them into multiple sub-Ways.

    # The result is that there is no 1:1 mapping based on ids alone.
    # Disabling graph simplification wouldn't help either because then we would get multiple edges
    # per one SUMO sub-Way.

    # The solution is to calculate geometric features based on SUMO (e.g. length, bearing, circuity, num_turns)
    # And get "semantic" features from OSM (e.g. lit, surface, tunnel, bridge, highway_type) because they apply
    # to the entire Way anyway and thus to all SUMO sub-Ways that were created from it.
    # When it comes to, say, distance, we cannot use the same approach because then, if the Way is 100 m long
    # and SUMO divides it into 2 sub-Ways with ids (e.g. 123#1, 123#2) that point to the same OSM Way id (e.g. 123)
    # it means they both are assigned this length and we get 200 m instead of 100 m.

    df_sumo = load_sumo_edges(edg_file)
    sumo_node_coords = load_sumo_nodes(nod_file)

    def fill_shape(row):
        if row["shape"] is not None:
            return row["shape"]
        u, v = row["from"], row["to"]
        if u in sumo_node_coords and v in sumo_node_coords:
            return [sumo_node_coords[u], sumo_node_coords[v]]
        return None # should not happen if data is consistent
    
    df_sumo["shape"] = df_sumo.apply(fill_shape, axis=1)
    dropped = df_sumo["shape"].isna().sum()
    df_sumo = df_sumo.dropna(subset=["shape"])
    if dropped:
        print(f"Dropped {dropped} rows - shape missing")

    df_sumo["osmid"] = df_sumo["sumo_id"].apply(get_osm_id_from_sumo) # osmid also appears in the osm df
    df_sumo["has_traffic_light"] = df_sumo["to"].isin(get_traffic_light_nodes(net_file)).astype(int)
    def clean_hwy(x):
        if pd.isna(x):
            return "unknown"
        return x.split(".")[-1] if isinstance(x, str) else x
    df_sumo["type_clean"] = df_sumo["type"].apply(clean_hwy)
    df_sumo["speed"] = df_sumo["speed"].astype(float) * 3.6  # maxspeed, originally in m/s
    df_sumo["length"] = df_sumo["shape"].apply(compute_sumo_length)
    df_sumo["bearing"] = df_sumo["shape"].apply(compute_sumo_bearing)

    # Extract start/end coordinates for circuity calculation later
    # Use these instead of OSM nodes because SUMO edges are splits of OSM ways
    def get_coords(shape):
        return shape[0][0], shape[0][1], shape[-1][0], shape[-1][1]
    coords = df_sumo["shape"].apply(get_coords).apply(pd.Series).astype(float)
    df_sumo[["start_x", "start_y", "end_x", "end_y"]] = coords    

    df_sumo = df_sumo.drop(columns=["shape"]) # big and not used later -> drop

    # Surface and lit are not included by default!
    # Default is ["access", "area", "bridge", "est_width", "highway", "junction", "landuse", 
    # "lanes", "maxspeed", "name", "oneway", "ref", "service", "tunnel", "width"]
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
    # G = ox.bearing.add_edge_bearings(G)
    # nodes are identified by OSM ID and each must contain a data attribute dictionary that must have “x” and “y” keys defining its coordinates
    # edges are identified by a 3-tuple of “u” (source node ID), “v” (target node ID), and “key” (to differentiate parallel edges), 
    # and each must contain a data attribute dictionary that must have an “osmid” key defining its OSM ID and a “length” key defining its length in meters
    _, edges_osm = ox.graph_to_gdfs(G) # 10108 rows for beynes
    # print(edges_osm)
    # Some osmid values can be lists - explode so as not to lose any edges
    # reset_index() because after exploding, new rows share the same index
    # NO drop=True because u and v, which are needed later, become columns!
    edges_osm_exploded = edges_osm.explode("osmid").reset_index() # 10557 rows for beynes
    # print(edges_osm_exploded)
    # Already in SUMO: lanes, highway (type), maxspeed (speed)
    # OSMnx length accounts for Earth's curvature instead of calculating simple Euclidean distances
    # BUT it has to be excluded! Each edge has a length but sumo might split an edge into
    # multiple segments, all pointing to the same edge osmid. They all get the same length
    # while in reality it should be their sum.
    osm_cols_to_keep = ["surface", "bridge", "tunnel", "lit"] #, "bearing", "length", "u", "v"]
    # If multiple edges have the same id, take the 1st one (???)
    edges_osm_unique = edges_osm_exploded.groupby("osmid")[osm_cols_to_keep].first().reset_index() # 1272 rows for beynes
    # print(edges_osm_unique)
    for col in osm_cols_to_keep:
        edges_osm_unique[col] = edges_osm_unique[col].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Left join: keep all SUMO edges, attach OSM info where matches exist
    df_merged = pd.merge(
        df_sumo,
        edges_osm_unique,
        on="osmid",
        how="left"
    )

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
    edge_features = edge_features.set_index("sumo_id")

    def compute_features_for_path(path_str):
        edges = path_str.split(",")
        # Only select the rows that make up the path (index is sumo_id)
        # df = edge_features.loc[edge_features.index.intersection(edges)] 
        # reindex preserves order and inserts NaNs for missing ids
        path_df = edge_features.reindex(edges)
        # Drop rows that are completely missing
        path_df = path_df.dropna(how="all")
        total_len = path_df["length"].sum(skipna=True)
        total_len_km = total_len / 1000.0

        feature_dict = {}
        # Basic features
        feature_dict["total_length"] = total_len
        # Speed
        feature_dict["mean_speed"] = path_df["speed"].mean(skipna=True)
        feature_dict["speed_std"] = path_df["speed"].std(skipna=True)
        feature_dict["speed_range"] = path_df["speed"].max(skipna=True) - path_df["speed"].min(skipna=True)
        feature_dict["pct_high_speed"] = safe_div(
            path_df.loc[path_df["speed"] > 50, "length"].sum(skipna=True),
            total_len
        )
        # Surface type
        PAVED_SURFACES = ["asphalt", "paved", "concrete", "bricks", "sett", "grass_paver", "paving_stones"]
        UNPAVED_SURFACES = ["sand", "gravel", "dirt", "compacted"]
        feature_dict["pct_paved"] = safe_div(
            path_df.loc[path_df["surface"].isin(PAVED_SURFACES), "length"]
            .sum(skipna=True), 
            total_len
        )
        feature_dict["pct_unpaved"] = safe_div(
            path_df.loc[path_df["surface"].isin(UNPAVED_SURFACES), "length"]
            .sum(skipna=True), 
            total_len
        )
        # Road class
        HIGHWAY_VALUES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"]
        for hv in HIGHWAY_VALUES:
            feature_dict[f"pct_{hv}"] = safe_div(
                path_df.loc[path_df["type_clean"] == hv, "length"]
                .sum(skipna=True), 
                total_len
            )
        # Lit?
        feature_dict["pct_lit"] = safe_div(
            path_df.loc[path_df["lit"] == "yes", "length"]
            .sum(skipna=True), 
            total_len
        )
        # Lane/priotity changes, yieldings
        feature_dict["lane_changes_per_km"] = safe_div(path_df["lanes"].diff().abs().sum(skipna=True), total_len_km)
        feature_dict["priority_changes_per_km"] = safe_div(path_df["priority"].diff().abs().sum(skipna=True), total_len_km) # number of priority changes or their total value? total value says more I think???
        feature_dict["yield_priority_changes_per_km"] = safe_div(path_df["priority"].diff().lt(0).sum(skipna=True), total_len_km)
        feature_dict["traffic_lights_per_km"] = safe_div(path_df["has_traffic_light"].sum(skipna=True), total_len_km)
        # Structures
        feature_dict["bridges_per_km"] = safe_div(path_df["bridge"].notna().sum(skipna=True), total_len_km)
        feature_dict["tunnels_per_km"] = safe_div(path_df["tunnel"].notna().sum(skipna=True), total_len_km)
        # Road geometry: shape, turns, ...
        feature_dict["bearing_std"] = path_df["bearing"].std(skipna=True) # bearing variance (high = winding route, low = straight)
        feature_dict["turns_per_km"] = safe_div(count_turns(path_df), total_len_km) # copy not needed, df isn't modified
        feature_dict["mean_circuity"] = calculate_circuity(path_df, total_len) # copy not needed, df isn't modified 
        feature_dict["edge_length_std"] = path_df["length"].std(skipna=True) # Urban centers tend to have many short edges (blocks). Highways have long, consistent edges. High variance might indicate a route that transitions between highway and city.
        feature_dict["edges_per_km"] = safe_div(len(edges), total_len_km)

        return feature_dict

    # Enrich and save
    enriched = pd.concat([routes, routes["path"].apply(compute_features_for_path).apply(pd.Series)], axis=1)
    out_path = output_dir / f"{name}_routes_enriched.csv"
    enriched.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline to generate and enrich route data from SUMO networks",
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
    parser.add_argument(
        "--name", "-n",
        nargs="+",
        help="One or more network names to process (e.g. --name city another_city). If omitted, all networks in data/ are processed."
    )
    
    args = parser.parse_args()
    if not any([args.all, args.routes, args.features, args.enrich]):
        print("No stages provided. Use -h for help.")
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
        f"{'='*60}"
    )

    names = set(s.strip().lower() for s in args.name) if args.name else None
    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if names and name.lower() not in names:
            continue
        if args.routes: generate_csv_routes(name, data_dir, path_gen_kwargs, routes_dir) # generates routes
        if args.features: generate_feature_file(name, data_dir, osm_dir, merged_edges_dir) # generates merged_edges
        if args.enrich: enrich_routes(name, routes_dir, merged_edges_dir, enriched_routes_dir) # uses routes and merged_edges to generate enriched_routes

if __name__ == "__main__":
    main()
from pathlib import Path
import ast
import hashlib
from importlib.resources import files
import janux as jx
import time
import pandas as pd
from lxml import etree
import math
import argparse
import numpy as np
import json

CLUSTER_FEATURES = [
    # Cost / efficiency
    "free_flow_time",
    "total_length",
    "mean_speed",
    "speed_std",

    # Road environment
    "pct_high_capacity",
    "pct_mid_capacity",
    "pct_local",
    "road_type_entropy",

    # Intersections / complexity
    "traffic_lights_per_km",
    "turns_per_km",
    "yield_priority_changes_per_km",
    "left_yield_turns_per_km",

    # Geometry / urban complexity
    "edges_per_km",

    # Relative spatial distinctness
    "edge_dist_from_shortest",
    "mean_edge_dist_to_other_routes",
    "max_edge_dist_to_other_routes",
]

"""
1. Generate routes CSV for all networks found in a given directory.
    Requires:
        - <name>.con.xml
        - <name>.edg.xml
        - <name>.rou.xml
        - od_<name>.json or od_<name>.txt
        - agents.csv
    Outputs:
        - <name>_routes.csv

2. Generate a feature file with SUMO edges as rows and SUMO columns.
    Requires:
        - <name>.edg.xml
        - <name>.nod.xml
        - <name>.net.xml
    Outputs:
        - <name>_merged_edges.csv

3. Enrich merged edges file with newly calculated edge features.
    Requires:
        - <name>_routes.csv
        - <name>_merged_edges.csv
    Outputs:
        - <name>_enriched_routes.csv
        - <name>_ranking_matrix.csv
"""

def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0

def canonical_route_key_part(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()

def make_route_id(origin, destination, path: str) -> str:
    route_key = "|".join([
        canonical_route_key_part(origin),
        canonical_route_key_part(destination),
        path,
    ]).encode("utf-8")
    return hashlib.sha256(route_key).hexdigest()

def validate_and_prepare_routes(routes: pd.DataFrame, context: str) -> pd.DataFrame:
    required_columns = {"origins", "destinations", "path", "free_flow_time"}
    missing_columns = required_columns - set(routes.columns)
    if missing_columns:
        raise ValueError(f"{context}: missing route columns: {sorted(missing_columns)}")

    routes = routes.copy()
    if routes[["origins", "destinations", "path"]].isna().any().any():
        raise ValueError(f"{context}: found a route with a missing OD or path")
    routes["path"] = routes["path"].astype(str).apply(
        lambda path: ",".join(edge.strip() for edge in path.split(",") if edge.strip())
    )
    if (routes["path"] == "").any():
        raise ValueError(f"{context}: found an empty route")

    routes["free_flow_time"] = pd.to_numeric(routes["free_flow_time"], errors="coerce")
    invalid_fft = ~np.isfinite(routes["free_flow_time"]) | (routes["free_flow_time"] <= 0.0)
    if invalid_fft.any():
        examples = routes.loc[
            invalid_fft,
            ["origins", "destinations", "path", "free_flow_time"],
        ].head(5)
        raise ValueError(
            f"{context}: found {int(invalid_fft.sum())} routes with non-positive or "
            f"non-finite free-flow time:\n{examples.to_string(index=False)}"
        )

    routes = routes.drop_duplicates(
        subset=["origins", "destinations", "path"],
        keep="first",
    ).reset_index(drop=True)
    routes["route_id"] = [
        make_route_id(origin, destination, path)
        for origin, destination, path in routes[
            ["origins", "destinations", "path"]
        ].itertuples(index=False, name=None)
    ]

    if routes["route_id"].duplicated().any():
        raise ValueError(f"{context}: generated duplicate route IDs")

    return routes


def load_od_file(network_dir: Path, name: str) -> dict:
    json_path = network_dir / f"od_{name}.json"
    txt_path = network_dir / f"od_{name}.txt"

    if json_path.exists():
        return jx.utils.read_json(json_path)
    if txt_path.exists():
        with txt_path.open("r", encoding="utf-8") as file:
            data = ast.literal_eval(file.read())
        if not isinstance(data, dict):
            raise ValueError(f"OD file must contain a dictionary: {txt_path}")
        return data

    raise FileNotFoundError(
        f"{name}: missing OD file. Expected one of: {json_path}, {txt_path}"
    )

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

def circular_std_degrees(values: pd.Series) -> float:
    values = values.dropna().to_numpy(dtype=float)
    if values.size <= 1:
        return 0.0

    radians = np.deg2rad(values)
    mean_resultant_length = math.hypot(
        float(np.cos(radians).mean()),
        float(np.sin(radians).mean()),
    )
    mean_resultant_length = min(max(mean_resultant_length, 1e-12), 1.0)
    return min(
        180.0,
        float(np.rad2deg(math.sqrt(-2.0 * math.log(mean_resultant_length)))),
    )

def get_traffic_light_nodes(net_file: Path) -> set:
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

def _prepare_turn_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare bearing, bearing signed and priority diffs.
    
    350° → 10° (right turn)
    Raw diff = -340 (left turn)
    Normalized = +20 → (right turn)

    10° → 350° (left turn)
    Raw diff = +340 (right turn)
    Normalized = -20 → (left turn)
    """
    data = df[['bearing', 'priority']].copy().dropna()

    data['bearing_diff_signed'] = data['bearing'].diff()
    data['bearing_diff_signed'] = data['bearing_diff_signed'] \
        .where(data['bearing_diff_signed'] >= -180, data['bearing_diff_signed'] + 360)
    data['bearing_diff_signed']= data['bearing_diff_signed'] \
        .where(data['bearing_diff_signed'] <= 180, data['bearing_diff_signed'] - 360)

    data['bearing_diff'] = data['bearing'].diff()
    data['bearing_diff'] = data['bearing_diff'].abs()
    data['bearing_diff'] = data['bearing_diff'].where(
        data['bearing_diff'] <= 180,
        360 - data['bearing_diff']
    )

    data['priority_diff'] = data['priority'].diff()

    return data.dropna()

def count_turns(df: pd.DataFrame, turn_threshold_deg: int=30) -> int:
    """
    Count significant direction changes for a given path.
    Only count turns where the priority changes (real intersections)
    in case one road is divided into multiple segments.
    
    df is a dataframe with rows corresponding only to edges
    present in a given path and columns with features
    """
    data = _prepare_turn_data(df)

    is_significant = data['bearing_diff'] > turn_threshold_deg
    is_real_turn = data['priority_diff'] != 0

    return (is_significant & is_real_turn).sum()

def count_yield_left_turns(df: pd.DataFrame) -> int:
    """
    Count left turns where the driver must yield (enters lower priority road).
    Only counts at real intersections (priority changes).
    """
    data = _prepare_turn_data(df)

    is_left_turn = data['bearing_diff_signed'] < 0 # count negative (counter-clockwise) bearing changes
    is_yield = data['priority_diff'] < 0

    left_yield_turns = (is_left_turn & is_yield).sum()
    return left_yield_turns

def calculate_circuity(df: pd.DataFrame, total_len: float) -> float | None:
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

def load_config(config_arg: str | Path) -> tuple[dict, str]:
    """
    Load a config from an explicit filesystem path or from the package's
    bundled configs when given a bare name such as ``clustering-default``.
    """
    config_path = Path(config_arg)
    if config_path.exists():
        with config_path.open() as file:
            config = json.load(file)
        source = str(config_path.resolve())
    elif config_path.parent == Path("."):
        config_name = (
            config_path.name
            if config_path.suffix
            else f"{config_path.name}.json"
        )
        config_resource = files("path_clustering").joinpath(
            "configs",
            config_name,
        )
        if not config_resource.is_file():
            raise FileNotFoundError(
                f"Config file not found: {config_arg} "
                f"(also checked packaged config '{config_name}')"
            )
        with config_resource.open() as file:
            config = json.load(file)
        source = f"package:path_clustering/configs/{config_name}"
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {source}")
    return config, source


def load_path_gen_kwargs(config_arg: str | Path) -> tuple[dict, str]:
    config, source = load_config(config_arg)

    # Supports either:
    #   { "number_of_paths": 5, ... }
    # or:
    #   { "path_gen_kwargs": { "number_of_paths": 5, ... } }
    path_gen_kwargs = config.get("path_gen_kwargs", config)

    if not isinstance(path_gen_kwargs, dict):
        raise ValueError(
            f"Config must contain a JSON object of path generator kwargs: {source}"
        )

    return path_gen_kwargs, source

def get_route_generator(generator_name: str):
    generator_name = generator_name.lower()

    if generator_name == "clustering":
        return jx.clustering_generator

    if generator_name == "alternative":
        generator = getattr(jx, "alternative_generator", None)
        if generator is None:
            raise RuntimeError(
                "The installed JanuX version does not provide "
                "alternative_generator."
            )
        return generator

    raise ValueError(
        f"Unknown generator '{generator_name}'. "
        "Choose one of: clustering, alternative"
    )

def generate_csv_routes(
    name: str,
    net_dir: Path,
    path_gen_kwargs: dict,
    run_dir: Path,
    route_generator,
) -> Path:
    """
    Generate paths for each OD pair in agents.csv using JanuX.
    Results are saved in the selected run directory.
    """
    print(f"\n=== Generating routes for {name} ===")

    required_files = [
        net_dir / name / f"{name}.con.xml",
        net_dir / name / f"{name}.edg.xml",
        net_dir / name / f"{name}.rou.xml",
        net_dir / name / "agents.csv"
    ]
    
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"{name}: missing required files for route generation: {missing_files}"
        )

    con_file, edg_file, rou_file, agents_file = required_files

    ods = load_od_file(net_dir / name, name)
    agents = pd.read_csv(agents_file)
    try:
        origins = ods["origins"]
        destinations = ods["destinations"]
    except KeyError as exc:
        raise ValueError(
            f"{name}: OD file must contain 'origins' and 'destinations'"
        ) from exc

    all_routes = []
    failed_ods = []
    generation_rows = []
    start_time = time.time()
    network = jx.build_digraph(str(con_file), str(edg_file), str(rou_file), use_clustered_routes=True)

    # sample routes for each OD pair, not for each agent because multiple agents might have the same OD pair
    unique_od_pairs = agents[["origin", "destination"]].drop_duplicates()

    # 300 origins and 300 destinations -> 90000 OD pairs -> too long to process!
    # instead, take agents.csv (order of hundreds)
    for o_id, d_id in zip(unique_od_pairs["origin"], unique_od_pairs["destination"]):
        try:
            routes = route_generator(
                network,
                [origins[o_id]],
                [destinations[d_id]],
                as_df=True,
                calc_free_flow=True,
                **path_gen_kwargs,
            )
            routes = validate_and_prepare_routes(
                routes,
                f"{name} OD ({o_id}, {d_id}) generation",
            )

            if routes.empty:
                failed_ods.append((int(o_id), int(d_id), "no routes generated"))
                generation_rows.append({
                    "origin": int(o_id),
                    "destination": int(d_id),
                    "routes": 0,
                    "status": "failed: no routes generated",
                })
                continue

            all_routes.append(routes)
            generation_rows.append({
                "origin": int(o_id),
                "destination": int(d_id),
                "routes": len(routes),
                "status": "ok",
            })
        except AssertionError as e:
            failed_ods.append((int(o_id), int(d_id), str(e)))
            generation_rows.append({
                "origin": int(o_id),
                "destination": int(d_id),
                "routes": 0,
                "status": f"failed: {e}",
            })

    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"Generated routes for {len(all_routes)}/{len(unique_od_pairs)} ODs.")
    route_counts = [row["routes"] for row in generation_rows if row["status"] == "ok"]
    if route_counts:
        print(
            "Routes per successful OD: "
            f"min={min(route_counts)}, mean={np.mean(route_counts):.2f}, "
            f"max={max(route_counts)}"
        )

    with open(run_dir / f"{name}_generation_summary.json", "w") as f:
        json.dump(generation_rows, f, indent=2)

    if failed_ods:
        examples = failed_ods[:5]
        raise RuntimeError(
            f"{name}: route generation failed for {len(failed_ods)} ODs. "
            f"Examples: {examples}"
        )

    # Save the routes to a CSV file    
    if all_routes:
        all_routes_merged = validate_and_prepare_routes(
            pd.concat(all_routes, ignore_index=True),
            f"{name} combined route set",
        )
        csv_save_path = run_dir / f"{name}_routes.csv"
        all_routes_merged.to_csv(csv_save_path, index=False)
        print(f"Saved routes to: {csv_save_path}")
        return csv_save_path
    else:
        raise RuntimeError(f"No routes generated for {name}")

def generate_feature_file(name: str, net_dir: Path, run_dir: Path) -> Path:
    print(f"\n=== Generating feature file for {name} ===")

    edg_file = net_dir / name / f"{name}.edg.xml"
    nod_file = net_dir / name / f"{name}.nod.xml"
    net_file = net_dir / name / f"{name}.net.xml"
    required_files = [edg_file, nod_file, net_file]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"{name}: missing required files for edge features: {missing_files}"
        )

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

    # OSM features not used anymore
    # df_sumo["osmid"] = df_sumo["sumo_id"].apply(get_osm_id_from_sumo) # osmid also appears in the osm df
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

    out_path = run_dir / f"{name}_merged_edges.csv"
    df_sumo.to_csv(out_path, index=False)
    return out_path
    
def enrich_routes(name: str, run_dir: Path) -> tuple[Path, Path]:
    """Compute route features from edge data"""
    print(f"\n=== Enriching routes for {name} ===")

    routes_file = run_dir / f"{name}_routes.csv"
    edges_file  = run_dir / f"{name}_merged_edges.csv"

    missing_files = [
        str(path)
        for path in (routes_file, edges_file)
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{name}: missing route-enrichment inputs: {missing_files}"
        )

    routes = validate_and_prepare_routes(
        pd.read_csv(routes_file),
        f"{name} route enrichment input",
    )
    edge_features = pd.read_csv(edges_file)
    if edge_features["sumo_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate SUMO edge IDs in {edges_file}")
    edge_features = edge_features.set_index("sumo_id")
    # all SUMO edge IDs that have feature rows in edge_features
    valid_sumo_edges = set(edge_features.index)

    # New spatial features - how spatially distinct the route is relative to other alternatives for that OD
    # Not where in the city it is, which could make action labels city-location-specific and less general
    # Outside compute_features_for_path() because they need access to all routes for a given OD
    def path_to_edges(path_str: str) -> list[str]:
        edges = [edge.strip() for edge in path_str.split(",") if edge.strip()]
        missing_edges = [
            edge
            for edge in edges
            if edge not in valid_sumo_edges and not edge.startswith(":")
        ]
        if missing_edges:
            raise ValueError(
                f"Route contains SUMO edges without feature data: {missing_edges[:5]}"
            )
        return [edge for edge in edges if edge in valid_sumo_edges]

    def edge_jaccard_distance(edges_a: list[str], edges_b: list[str]) -> float:
        set_a = set(edges_a)
        set_b = set(edges_b)

        union = set_a | set_b
        if not len(union):
            return 0.0

        intersection = set_a & set_b

        return 1.0 - len(intersection) / len(union)

    def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Spatial features for each route
        spatial_rows = []

        # One OD at a time, ignore the key. Groupby doesn't reset the index
        for _, group in df.groupby(["origins", "destinations"], sort=False):
            group = group.copy()

            edge_lists = {
                idx: path_to_edges(path)
                for idx, path in group["path"].items()
            }

            shortest_idx = group["free_flow_time"].idxmin()
            shortest_route_edges = edge_lists[shortest_idx]

            for idx in group.index:
                current_route_edges = edge_lists[idx]

                dist_from_shortest = edge_jaccard_distance(current_route_edges, shortest_route_edges)
                other_dists = [
                    edge_jaccard_distance(current_route_edges, edge_lists[other_idx])
                    for other_idx in group.index
                    if other_idx != idx
                ]

                mean_dist_to_others = float(np.mean(other_dists)) if other_dists else 0.0
                max_dist_to_others = float(np.max(other_dists)) if other_dists else 0.0

                # Append spatial features for each route
                spatial_rows.append({
                    "index": idx,
                    "edge_dist_from_shortest": dist_from_shortest,
                    "mean_edge_dist_to_other_routes": mean_dist_to_others,
                    "max_edge_dist_to_other_routes": max_dist_to_others,
                })

        spatial_df = pd.DataFrame(spatial_rows).set_index("index") # index keys become the rows and the rest of the keys become columns
        df = df.join(spatial_df)

        # Relative rank within each OD: 1.0 = most spatially distinct candidate for that OD
        df["spatial_distinct_rank_for_OD"] = (
            df.groupby(["origins", "destinations"])["mean_edge_dist_to_other_routes"]
            .rank(pct=True)
        )

        return df

    def compute_features_for_path(route):
        edges = path_to_edges(route["path"])
        if not edges:
            raise ValueError(f"Route {route['route_id']} has no usable SUMO edges")

        # Only select the rows that make up the path (index is sumo_id)
        # df = edge_features.loc[edge_features.index.intersection(edges)]
        # reindex preserves route order
        path_df = edge_features.reindex(edges)
        if path_df.isna().all(axis=1).any():
            raise ValueError(f"Route {route['route_id']} has missing edge feature rows")

        total_len = path_df["length"].sum(skipna=True)
        if not np.isfinite(total_len) or total_len <= 0.0:
            raise ValueError(
                f"Route {route['route_id']} has invalid total length: {total_len}"
            )
        total_len_km = total_len / 1000.0

        feature_dict = {}
        # Basic features
        feature_dict["total_length"] = total_len
        # Speed
        feature_dict["mean_speed"] = path_df["speed"].mean(skipna=True)
        feature_dict["speed_std"] = path_df["speed"].std(skipna=True) if len(path_df) > 1 else 0.0
        feature_dict["speed_range"] = path_df["speed"].max(skipna=True) - path_df["speed"].min(skipna=True)
        feature_dict["pct_high_speed"] = safe_div(
            path_df.loc[path_df["speed"] > 50, "length"].sum(skipna=True),
            total_len
        )

        feature_dict["pct_high_capacity"] = safe_div(
            path_df.loc[path_df["type_clean"].isin(["motorway", "trunk", "primary"]), "length"].sum(skipna=True),
            total_len
        )

        feature_dict["pct_mid_capacity"] = safe_div(
            path_df.loc[path_df["type_clean"].isin(["secondary", "tertiary"]), "length"].sum(skipna=True),
            total_len
        )

        feature_dict["pct_local"] = safe_div(
            path_df.loc[path_df["type_clean"].isin(["unclassified", "residential"]), "length"].sum(skipna=True),
            total_len
        )

        road_type_shares = []
        for hv in ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"]:
            share = safe_div(
                path_df.loc[path_df["type_clean"] == hv, "length"].sum(skipna=True),
                total_len
            )
            if share > 0:
                road_type_shares.append(share)

        other_share = max(0.0, 1.0 - sum(road_type_shares))
        if other_share > 0:
            road_type_shares.append(other_share)

        feature_dict["road_type_entropy"] = -sum(p * math.log(p) for p in road_type_shares)

        # Lane/priority changes, yieldings
        feature_dict["lane_changes_per_km"] = safe_div(path_df["lanes"].diff().abs().sum(skipna=True), total_len_km)
        priority_diff = path_df["priority"].diff().dropna()
        feature_dict["priority_changes_per_km"] = safe_div((priority_diff != 0).sum(), total_len_km)
        feature_dict["yield_priority_changes_per_km"] = safe_div((priority_diff < 0).sum(), total_len_km)
        feature_dict["traffic_lights_per_km"] = safe_div(path_df["has_traffic_light"].sum(skipna=True), total_len_km)

        # Road geometry: shape, turns, ...
        feature_dict["bearing_std"] = circular_std_degrees(path_df["bearing"]) # bearing variance (high = winding route, low = straight)
        feature_dict["turns_per_km"] = safe_div(count_turns(path_df), total_len_km) # copy not needed, df isn't modified
        feature_dict["left_yield_turns_per_km"] = safe_div(count_yield_left_turns(path_df), total_len_km) # copy not needed, df isn't modified
        feature_dict["mean_circuity"] = calculate_circuity(path_df, total_len) # copy not needed, df isn't modified 
        # feature_dict["edge_length_std"] = path_df["length"].std(skipna=True) if len(path_df) > 1 else 0.0 # Urban centers tend to have many short edges (blocks). Highways have long, consistent edges. High variance might indicate a route that transitions between highway and city.
        feature_dict["edges_per_km"] = safe_div(len(path_df), total_len_km) # Urban centers tend to have many short edges (blocks). Highways have long, consistent edges. High variance might indicate a route that transitions between highway and city.

        return feature_dict

    # Enrich and save
    enriched = pd.concat(
        [routes, routes.apply(compute_features_for_path, axis=1).apply(pd.Series)],
        axis=1
    )

    enriched = add_spatial_features(enriched)
    
    # Fill NaNs with defaults
    fill_values = {
        'mean_circuity': 1.0, # ? why is that NaN sometimes
    }
    enriched = enriched.fillna(fill_values)

    enriched_path = run_dir / f"{name}_routes_enriched.csv"
    enriched.to_csv(enriched_path, index=False)
    print(f"Saved to {enriched_path}")

    # Ranking matrix for clustering - one file per network because otherwise too many files!
    missing_features = set(CLUSTER_FEATURES) - set(enriched.columns)
    if missing_features:
        raise ValueError(
            f"{name}: missing clustering features: {sorted(missing_features)}"
        )
    non_finite_features = ~np.isfinite(
        enriched[CLUSTER_FEATURES].apply(pd.to_numeric, errors="coerce")
    )
    if non_finite_features.any().any():
        invalid_columns = non_finite_features.any()
        raise ValueError(
            f"{name}: non-finite values in clustering features: "
            f"{invalid_columns[invalid_columns].index.tolist()}"
        )

    def rank_within_od(series: pd.Series) -> pd.Series:
        if len(series) <= 1 or series.nunique(dropna=False) <= 1:
            return pd.Series(0.5, index=series.index)
        return (series.rank(method="average") - 1.0) / (len(series) - 1.0)

    od_pairs = enriched.groupby(["origins", "destinations"], sort=False)

    ranking_matrix_agents = []
    for _, group in od_pairs:
        agent_matrix = group[
            ["route_id", "origins", "destinations", *CLUSTER_FEATURES]
        ].copy()
        for col in CLUSTER_FEATURES:
            agent_matrix[col] = rank_within_od(agent_matrix[col])

        ranking_matrix_agents.append(agent_matrix)

    ranking_matrix = pd.concat(ranking_matrix_agents, ignore_index=True)

    ranking_path = run_dir / f"{name}_ranking_matrix.csv"
    ranking_matrix.to_csv(ranking_path, index=False)
    print(f"Saved to {ranking_path}")
    return enriched_path, ranking_path

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline to generate and enrich route data from SUMO networks",
    )
    
    # Store true
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
        help="Generate merged SUMO edge feature files"
    )
    parser.add_argument(
        "--enrich", 
        action="store_true", 
        help="Enrich routes with calculated features"
    )

    # Names
    parser.add_argument(
        "--name", "-n",
        nargs="+",
        help="One or more network names to process (e.g. --name city another_city). If omitted, all networks in data/ are processed."
    )
    parser.add_argument(
        "--run-name", "-r",
        help="Name for this run's output folder inside results/ (e.g. ingolstadt_1)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing one subdirectory per SUMO network. Defaults to data/."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit output directory. Cannot be combined with --run-name."
    )

    # Config, generator
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="clustering-default",
        help="Explicit config path or bundled config name."
    )
    parser.add_argument(
        "--generator", "-g",
        type=str,
        default="clustering",
        choices=["clustering", "alternative"],
        help="Choose the route generator (clustering/alternative)"
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

    if args.output_dir is not None and args.run_name is not None:
        parser.error("--output-dir and --run-name cannot be used together")
    if args.output_dir is None and args.run_name is None:
        parser.error("one of --run-name or --output-dir is required")
    # Network data
    data_dir = (args.data_dir or repo_root / "data").resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Network data directory does not exist: {data_dir}")

    # Results
    run_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (repo_root / "results" / args.run_name).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    
    path_gen_kwargs = {}
    route_generator = None
    if args.routes:
        path_gen_kwargs, config_source = load_path_gen_kwargs(args.config)

        route_generator = get_route_generator(args.generator)

        print(f"Using config file: {config_source}")
        print(f"Using route generator: {args.generator}")

        with open(run_dir / f"{run_dir.name}_path_gen_kwargs.json", "w") as f:
            json.dump(path_gen_kwargs, f, indent=2)
        with open(run_dir / f"{run_dir.name}_generation_config.json", "w") as f:
            json.dump(
                {
                    "generator": args.generator,
                    "config": config_source,
                    "path_gen_kwargs": path_gen_kwargs,
                },
                f,
                indent=2,
            )

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
    available_names = {
        path.name.lower()
        for path in data_dir.iterdir()
        if path.is_dir()
    }
    if names:
        missing_names = names - available_names
        if missing_names:
            raise FileNotFoundError(
                f"Networks not found in {data_dir}: {sorted(missing_names)}"
            )

    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if names and name.lower() not in names:
            continue
        if args.routes:
            generate_csv_routes(
                name,
                data_dir,
                path_gen_kwargs,
                run_dir,
                route_generator,
            ) # generates routes
        if args.features: generate_feature_file(name, data_dir, run_dir) # generates merged_edges
        if args.enrich: enrich_routes(name, run_dir) # uses routes and merged_edges to generate enriched_routes

if __name__ == "__main__":
    main()

import pandas as pd
from lxml import etree
import osmnx as ox
import math

############################################
# SETUP
############################################

OSM_FILE = "../data/andorra/andorra.osm"
NET_FILE = "../data/andorra/andorra.net.xml"

############################################
# UTILS
############################################

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

############################################
# PART 1 — EXTRACT SUMO EDGES (.edg.xml)
############################################

print("\n=== Extracting SUMO edges ===")

df_sumo = load_sumo_edges_from_edg("../data/andorra/andorra.edg.xml")
df_sumo["osm_id"] = df_sumo["sumo_id"].apply(get_osm_id_from_sumo) # move
df_sumo.to_csv("sumo_edges.csv", index=False)
print("SUMO edges extracted to sumo_edges.csv")
print(df_sumo.head())

############################################
# PART 2 — LOAD OSM USING OSMNX
############################################

print("\n=== Loading OSM with OSMnx ===")

G = ox.graph_from_xml(OSM_FILE)
nodes_osm, edges_osm = ox.graph_to_gdfs(G)
# Some osmid values can be lists - convert to first element
edges_osm["osmid_clean"] = edges_osm["osmid"].apply(
    lambda x: x[0] if isinstance(x, list) else x
) #?
edges_osm.to_csv("osm_edges.csv", index=False)
print("OSM edges extracted to osm_edges.csv")
print(edges_osm[["osmid_clean", "highway", "lanes", "maxspeed"]].head())

############################################
# PART 3 — MERGE SUMO EDGES WITH OSM FEATURES
############################################

print("\n=== Merging SUMO edges with OSM features ===")

edges_osm = edges_osm.set_index("osmid_clean") # replace default row index (0,1,2,...) with osmid

merged_rows = []

traffic_light_nodes = get_traffic_light_nodes(NET_FILE)
df_sumo["has_traffic_light"] = df_sumo["to"].isin(traffic_light_nodes).astype(int)

for idx, row in df_sumo.iterrows():
    osm_id = row["osm_id"]

    # OSM features
    if osm_id in edges_osm.index:
        osm_row = edges_osm.loc[osm_id]
        if isinstance(osm_row, pd.DataFrame):
            osm_row = osm_row.iloc[0]
        highway = osm_row.get("highway")
        maxspeed = osm_row.get("maxspeed")
        lanes = osm_row.get("lanes")
    else:
        highway = None
        maxspeed = None
        lanes = None

    merged_rows.append({
        "sumo_id": row["sumo_id"],
        "from": row["from"],
        "to": row["to"],
        "shape": row["shape"],
        "priority": row["priority"],
        "speed": row["speed"],
        "num_lanes": row["lanes"],

        "has_traffic_light": int(row["has_traffic_light"]),

        "osm_id": osm_id,
        "highway": highway,
        "maxspeed": maxspeed,
        "lanes": lanes,
    })

df_merged = pd.DataFrame(merged_rows)
df_merged["length"] = [
    r["length"] if ("length" in df_merged.columns and pd.notna(r["length"]))
    else compute_length_from_shape(r["shape"])
    for _, r in df_merged.iterrows()
]

df_merged.to_csv("sumo_osm_merged.csv", index=False)

print("Merged SUMO + OSM features to sumo_osm_merged.csv")
print(df_merged[["sumo_id", "osm_id", "highway", "lanes", "maxspeed"]].head())

############################################
# DONE
############################################

print("\n=== SUCCESS ===")
print("All steps completed.")
print("Check these files:")
print("  - sumo_edges.csv")
print("  - osm_edges.csv")
print("  - sumo_osm_merged.csv")

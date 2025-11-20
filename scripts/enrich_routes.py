import pandas as pd

routes = pd.read_csv("../results/andorra_routes.csv")
edge_features = pd.read_csv("sumo_osm_merged.csv")

NUMERIC_FIELDS = ["length", "speed", "lanes"]
HIGHWAY_VALUES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"]

for col in NUMERIC_FIELDS:
    if col in edge_features.columns:
        edge_features[col] = pd.to_numeric(edge_features[col], errors="coerce")

edge_features = edge_features.set_index("sumo_id")

def clean_hwy(x):
    if isinstance(x, str):
        return x.split(".")[-1]
    return x

edge_features["highway_clean"] = edge_features["highway"].apply(clean_hwy)

def compute_features_for_path(path_str):
    edges = path_str.split(",")

    df = edge_features.loc[edge_features.index.intersection(edges)]

    feature_dict = {}

    feature_dict["num_edges"] = len(edges)
    feature_dict["total_length"] = df["length"].sum(skipna=True)
    feature_dict["mean_speed"] = df["speed"].mean(skipna=True)

    total_len = feature_dict["total_length"]
    for hv in HIGHWAY_VALUES:
        if total_len > 0:
            feature_dict[f"pct_{hv}"] = df.loc[df["highway_clean"] == hv, "length"].sum() / total_len
            # feature_dict["pct_motorway"] = df.loc[df["highway_clean"] == "motorway", "length"].sum() / total_len
            # feature_dict["pct_trunk"] = df.loc[df["highway_clean"] == "trunk", "length"].sum() / total_len
            # feature_dict["pct_primary"] = df.loc[df["highway_clean"] == "primary", "length"].sum() / total_len
            # feature_dict["pct_secondary"] = df.loc[df["highway_clean"] == "secondary", "length"].sum() / total_len
            # feature_dict["pct_tertiary"] = df.loc[df["highway_clean"] == "tertiary", "length"].sum() / total_len
            # feature_dict["pct_unclassified"] = df.loc[df["highway_clean"] == "unclassified", "length"].sum() / total_len
            # feature_dict["pct_residential"] = df.loc[df["highway_clean"] == "residential", "length"].sum() / total_len
        else:
            feature_dict[f"pct_{hv}"] = 0.0
            # feature_dict["pct_motorway"] = 0.0
            # feature_dict["pct_trunk"] = 0.0
            # feature_dict["pct_primary"] = 0.0
            # feature_dict["pct_secondary"] = 0.0
            # feature_dict["pct_tertiary"] = 0.0
            # feature_dict["pct_unclassified"] = 0.0
            # feature_dict["pct_residential"] = 0.0

    feature_dict["num_traffic_lights"] = df["has_traffic_light"].sum()

    return feature_dict

final_df = routes.copy()
final_df = pd.concat([final_df, routes["path"].apply(compute_features_for_path).apply(pd.Series)], axis=1)
final_df.to_csv("andorra_routes_enriched.csv", index=False)

print("Saved enriched CSV to andorra_routes_enriched.csv")

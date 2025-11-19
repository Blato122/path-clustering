import pandas as pd

routes = pd.read_csv("../results/andorra_routes.csv")
edge_features = pd.read_csv("sumo_osm_merged.csv")

NUMERIC_FIELDS = ["length", "speed", "lanes"]

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
    feature_dict["median_lanes"] = df["lanes"].median(skipna=True)

    feature_dict["pct_motorway"]    = (df["highway_clean"] == "motorway").mean()
    feature_dict["pct_residential"] = (df["highway_clean"] == "residential").mean()
    feature_dict["pct_tertiary"]    = (df["highway_clean"] == "tertiary").mean()

    return feature_dict

final_df = routes.copy()
final_df = pd.concat([final_df, routes["path"].apply(compute_features_for_path).apply(pd.Series)], axis=1)
final_df.to_csv("andorra_routes_enriched.csv", index=False)

print("Saved enriched CSV to andorra_routes_enriched.csv")

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import os

# Get this script's dir and go up one level to path-clustering root
script_dir = Path(__file__).parent
repo_root = script_dir.parent

# Paths relative to repo root
results_dir = repo_root / "results"
analysis_dir = repo_root / "analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)

route_files = {}
for d in results_dir.iterdir():
    if d.is_dir():
        files = set()
        files.update(d.glob("*_routes_enriched.csv"))
        files.update(d.glob("*_clusters_representants.csv"))
        route_files[d.name] = list(files)

if not route_files:
    print("No route files found!")
    exit()

results = {}
for dir_name, route_files in route_files.items():
    if len(route_files) == 0: 
        continue
    for routes in route_files:
        full_name = dir_name + ": " + ("clusters" if "clusters" in routes.name else "enriched")
        
        routes = pd.read_csv(routes)

        exclude = ["origins", "destinations", "path", "h3_sequence", "cluster"]
        cols = [col for col in routes.columns if col not in exclude]
        
        means = routes[cols].mean()
        means["od_count_mean"] = routes.groupby(["origins", "destinations"]).size().mean()

        stats = routes[["free_flow_time", "total_length"]].agg(["min", "max", "std"]).T.stack()
        stats.index = [f"{col}_{stat}" for col, stat in stats.index]

        results[full_name] = pd.concat([means, stats]) 

df = pd.DataFrame(results)
df = df.sort_index(axis=1) # sort columns by name
df.to_csv(analysis_dir / 'path_analysis.csv')
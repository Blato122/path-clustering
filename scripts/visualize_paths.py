import matplotlib
matplotlib.use('Agg') # use non-interactive backend

import pandas as pd
from pathlib import Path
import janux as jx

city_name = "bussy_saint_georges"

# Get this script's dir and go up one level to path-clustering root
script_dir = Path(__file__).parent
repo_root = script_dir.parent

# Paths relative to repo root
data_dir = repo_root / "data" / city_name
routes_file = repo_root / "results" / "routes" / f"{city_name}_routes.csv"
misc_dir = repo_root / "results" / "misc"

nod_file = data_dir / f"{city_name}.nod.xml"
edg_file = data_dir / f"{city_name}.edg.xml"

routes = pd.read_csv(routes_file)

# Pick a random OD pair
import random
# random.seed(42) # for reproducibility, comment for randomness
unique_ods = routes.groupby(['origins', 'destinations']).size().index.tolist()
selected_od = random.choice(unique_ods)
origin, destination = selected_od

print(f"Selected OD pair: {origin} -> {destination}")

# Get all paths for this OD pair
od_routes = routes[(routes['origins'] == origin) & (routes['destinations'] == destination)]
print(f"Total paths for this OD pair: {len(od_routes)}")

# Select up to 5 random paths
n_paths = min(5, len(od_routes))
random_paths = od_routes.sample(n=n_paths)#, random_state=42) # remove random_state for randomness

# Convert to list of edge lists
routes_to_visualize = []
for row in random_paths.itertuples():
    routes_to_visualize.append(row.path.split(','))

jx.show_multi_routes(
    nod_file_path=str(nod_file),
    edg_file_path=str(edg_file),
    paths=routes_to_visualize,
    origin=origin,
    destination=destination,
    autocrop=True,
    save_file_path=misc_dir/f"{city_name}_random_paths.png"
)

print(f"Saved visualization to {misc_dir/f"{city_name}_random_paths.png"}_random_paths.png")

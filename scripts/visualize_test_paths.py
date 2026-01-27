import matplotlib
matplotlib.use('Agg') # use non-interactive backend

import pandas as pd
from pathlib import Path
import janux as jx
import random

# Get this script's dir and go up one level to path-clustering root
script_dir = Path(__file__).parent
repo_root = script_dir.parent
routes_dir = repo_root / "results" / "test"
misc_dir = repo_root / "results" / "test" # repo_root / "results" / "misc"
data_dir = repo_root / "results" / "test" # repo_root / "data"

route_files = list(routes_dir.glob("*_routes.csv"))

for route_file in route_files:
    city_name = route_file.stem.replace("_routes", "")
    print(city_name)

    nod_file = data_dir / f"{city_name}.nod.xml" # data_dir / city_name / f"{city_name}.nod.xml"
    edg_file = data_dir / f"{city_name}.edg.xml" # data_dir / city_name / f"{city_name}.edg.xml"
    routes = pd.read_csv(route_file)

    # Pick a random OD pair
    # random.seed(42) # for reproducibility, comment for randomness
    min_routes = 1 # only choose from ODs with at least 3 routes
    path_counts = routes.groupby(['origins', 'destinations']).size()
    filtered_ods = path_counts[path_counts >= min_routes].index.tolist()

    if not filtered_ods:
        print(f"No OD pairs with >= {min_routes} routes in {city_name}; skipping.")
        continue

    selected_od = random.choice(filtered_ods)
    origin, destination = selected_od

    print(f"Selected OD pair: {origin} -> {destination}")
    # Get all paths for this OD pair
    od_routes = routes[(routes['origins'] == origin) & (routes['destinations'] == destination)]
    print(f"Total paths for this OD pair: {len(od_routes)}")

    n_paths = min(5, len(od_routes))
    random_paths = od_routes.sample(n=n_paths)

    # Convert to list of edge lists, visualize all paths
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
        save_file_path=misc_dir/f"{city_name}_random_paths.png",
    )

    print(f"Saved visualization to {city_name}_random_paths.png\n")

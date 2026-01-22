import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Get this script's dir and go up one level to path-clustering root
script_dir = Path(__file__).parent
repo_root = script_dir.parent

# Paths relative to repo root
routes_dir = repo_root / "results" / "routes"
misc_dir = repo_root / "results" / "misc"

route_files = list(routes_dir.glob("*_routes.csv"))

if not route_files:
    print("No route files found!")
    exit()

results = []
for route_file in route_files:
    city_name = route_file.stem.replace("_routes", "")
    
    routes = pd.read_csv(route_file)
    path_counts = routes.groupby(['origins', 'destinations']).size()
    
    for count in path_counts:
        results.append({
            'city': city_name,
            'paths_per_agent': count
        })

df = pd.DataFrame(results)

summary = df.groupby('city')['paths_per_agent'].agg(['mean', 'std', 'min', 'max', 'count'])
summary.columns = ['Mean paths', 'Std paths', 'Min paths', 'Max paths', 'Total agents']

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

sns.boxplot(data=df, y='city', x='paths_per_agent', ax=ax)
ax.set_xlabel('Number of paths per agent')
ax.set_ylabel('City')
ax.set_title('Path count distribution by city')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(misc_dir / 'path_count_analysis.png', dpi=300, bbox_inches='tight')

summary.to_csv(misc_dir / 'path_count_summary.csv')
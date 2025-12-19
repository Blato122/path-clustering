import pandas as pd
from pathlib import Path
import webbrowser

df = pd.read_csv("../results/clustered_routes/beynes_routes_clustered.csv", quotechar='"')

# Cechy ciągłe do normalizacji
continuous_features = [
    'free_flow_time',
    'total_length',
    'mean_speed',
    'speed_std',
    'speed_range',
    'lane_changes_per_km',
    'priority_changes_per_km',
    'yield_priority_changes_per_km',
    'traffic_lights_per_km',
    # 'bearing_std',
    'turns_per_km',
    'left_yield_turns_per_km',
    # 'mean_circuity',
    # 'edge_length_std',
    # 'edges_per_km'
]

# Procentowe cechy w skali 0–1 (nie wymagają normalizacji)
percentage_features = [
    # 'pct_high_speed',
    # 'pct_motorway',
    # 'pct_trunk',
    # 'pct_primary',
    # 'pct_secondary',
    # 'pct_tertiary',
    # 'pct_unclassified',
    # 'pct_residential'
]

# Wszystkie cechy używane w modelu
features = continuous_features + percentage_features


numeric_features = df[features]
clusters = df['cluster']

#GLOBAL
global_mean = numeric_features.mean()
global_var = numeric_features.var()
global_min = numeric_features.min()
global_max = numeric_features.max()

global_stats = pd.DataFrame({
    "feature": numeric_features.columns,
    "cluster": "global",
    "mean": global_mean.values,
    "variance": global_var.values,
    "min": global_min.values,
    "max": global_max.values
})

#each cluster
cluster_stats = []

for clust in sorted(df['cluster'].unique()):
    subset = numeric_features[clusters == clust]

    mean_vals = subset.mean()
    var_vals = subset.var()
    min_vals = subset.min()
    max_vals = subset.max()

    tmp = pd.DataFrame({
        "feature": mean_vals.index,
        "cluster": clust,
        "mean": mean_vals.values,
        "variance": var_vals.values,
        "min": min_vals.values,
        "max": max_vals.values
    })

    cluster_stats.append(tmp)

cluster_stats_df = pd.concat(cluster_stats, ignore_index=True)


#Combine global and clusters 
combined = pd.concat([global_stats, cluster_stats_df], ignore_index=True)

combined.to_csv("combined_stats.csv", index=False)
print("Zapisano combined_stats.csv")

import pandas as pd
from pathlib import Path
import webbrowser


combined = pd.read_csv("combined_stats.csv")

# === Function to convert DataFrame into interactive HTML table ===
def make_datatable(df, title):
    table_html = df.to_html(index=False, classes="display compact cell-border", border=0)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>

    <link rel="stylesheet" 
          href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">

    <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
    <script 
        src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>

    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 20px;
        }}
        h2 {{
            margin-bottom: 20px;
        }}

        table.dataTable {{
            border-collapse: collapse !important;
            width: 100%;
        }}

        table.dataTable th,
        table.dataTable td {{
            border: 1px solid #ddd !important;
            padding: 8px !important;
        }}

        table.dataTable tbody tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        table.dataTable thead th {{
            background-color: #f0f0f0;
        }}
    </style>
</head>

<body>

<h2>{title}</h2>

{table_html}

<script>
    $(document).ready(function() {{
        $('table').DataTable({{
            paging: true,
            searching: true,
            ordering: true,
            pageLength: 25
        }});
    }});
</script>

</body>
</html>
"""
    return html



output_file = "combined_stats.html"
Path(output_file).write_text(make_datatable(combined, "Combined Global + Cluster Statistics"))

print(f"Generated: {output_file}")

# Automatically open in default browser - does not work idk
# webbrowser.open(output_file)

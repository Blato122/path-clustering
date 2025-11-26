from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

routes = pd.read_csv("")
scaler = StandardScaler() # mean=0, std=1 for each feature
routes_scaled = scaler.fit_transform(routes)

# exclude: origins, destinations, path
# as well as some sparse features like pct_unpaved or num_tunnels etc.
# one-hot encode some?
feature_cols = [
    'num_edges', 'total_length', 'mean_speed',
    'pct_motorway', 'pct_trunk', 'pct_primary', 
    'pct_secondary', 'pct_tertiary', 'pct_unclassified',
    'pct_residential', 'num_traffic_lights'
]

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
routes_enriched['cluster'] = kmeans.fit_predict(routes_scaled)

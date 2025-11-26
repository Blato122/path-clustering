import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap
import matplotlib.pyplot as plt
import math


# Load data
df = pd.read_csv("../results/enriched_routes/beynes_routes_enriched.csv", quotechar='"')


# Drop the path column becauce it has strings
df = df.drop(columns=['path']) 


# Select numeric features for clustering
numeric_cols = [
    'free_flow_time','num_edges','total_length','mean_speed',
    'pct_paved','pct_unpaved','pct_motorway','pct_trunk','pct_primary',
    'pct_secondary','pct_tertiary','pct_unclassified','pct_residential','pct_lit',
    'bearing_std','num_turns','num_bridges','num_tunnels','speed_std',
    'speed_range','lane_changes','priority_changes','num_traffic_lights',
    'path_points'
]

# Keep only columns that exist in the CSV
features = [col for col in numeric_cols if col in df.columns]

# Fill missing values
X = df[features].fillna(0).values


# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=5,
    metric="euclidean"
)

labels = clusterer.fit_predict(X_scaled)
df["cluster"] = labels

print(df["cluster"].value_counts())


# Save clustered dataset
df.to_csv("../results/clustered_routes/beynes_routes_clustered.csv", index=False)


# UMAP visualization
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X_scaled)

plt.figure(figsize=(10,7))
plt.scatter(embedding[:,0], embedding[:,1], c=labels, s=3, cmap='tab20')
plt.title("UMAP + HDBSCAN Clustering of Beynes Routes")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.colorbar(label='Cluster')
plt.show()

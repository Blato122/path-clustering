from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

"""
1. RANKING MATRICES

Per-OD - calculate the rank of a path only relative to other paths for that specific OD.
Every single OD pair has a #1 fastest path and a #1 shortest path etc. Whether a path is 2km 
or 20km, there is always a "shortest" (0th percentile) and "longest" (100th percentile). It also
serves as a normalization step. But we lose the "absolute truth". We won't know if the "shortest" 
route is actually 50km long. But the RL agent ultimately, I think, cares about the trade-off
between choosing action A over action B, not their absolute features. When there are 3 routes
with travel times of 50, 60 and 70 minutes, we care that the first one is the fastest relative
to the rest; not that it's slower than some other routes for different ODs.

Per-city - calculate the rank of a path relative to all paths in a given city.
The result might be that all paths for a given OD are clustered as "slow" because they're
slow compared to other paths in that city, even though some of these "slow" routes are faster
than other.

Global (all cities) - extreme variations? Huge city traffic vs rural roads.

--> My suggestion - per-OD.

2. CLUSTERING

Per-OD - poorly suited for RL? Clusters might have a different meaning for each OD pair. Actions
for different ODs might mean something different making it impossible (?) for the neural network
to learn anything.

Per-city - tailors the action space to a particular city network. Will work but actions change
meaning if we move the agent to a different city (we don't?).

Global (all cities) - also a viable approach. Ensures actions are consistent across all cities.
But the average characteristics of clusters might be less precise for any single city.

--> My suggestion - per-city.

3. CLOSEST REPRESENTATIVES

I'm afraid that forcing that each path for a given OD be assigned to a different cluster might 
confuse the agent and lead to poor performance and inoptimal policy. Let's say there are 5 clusters
and an agent has 5 paths, 1 of which matches cluster 1 and the rest matches cluster 2. This approach 
would mean that we force three of these paths to match clusters 3, 4 and 5 even if they completely
don't match them. This might cause the agent to think that it's trying action 5 (which should be a route
from cluster 5) while in reality it gets the same result as action 2 - a path from cluster 2.

--> My suggestion - drop the closest representative logic.

4. ACTION MASKING

To solve the 2nd issue, I think we should drop the closest representative logic and try action masking
instead. If an agent doesn't have a path that naturally falls into a given cluster, the action corresponding
to choosing a path from that cluster gets masked (disabled) for that agent.

This could also neatly handle cases in which there are insufficient paths for a given OD - e.g. just 1. This
will, unfortunately, inevitably happen no matter how good the path generation algorithm because sometimes the
network simply has only one connection between two points, especially with all the constraints (no edge revisit,
no junction revisit, no U-turns). In such cases, an RL agent would see action K as true and the rest as false - e.g.
[0, 0, 1, 0, 0]. It would see it as a "trivial state" with no alternatives rather than something confusing.

Requires RouteRL modification.

--> My suggestion - introduce action masking.

5. EVALUATION

In a setting where RL experiments take ~40 hours (URB), we cannot rely on extrinsic evaluation (measures a system's 
quality based on its impact on the performance of a real-world task). 

Before touching the RL, we need to run robust intrinsic evaluation tests to filter out most of the bad configurations.
Use the elbow method or silhouette score analysis.
Also, analyze the average number of valid actions per agent.

--> My suggestion - focus on exhaustive intrinsic evaluation and only test the best settings running URB simulations.

"""

def choose_features():
    features = [
        'free_flow_time',
        'total_length',
        'mean_speed',
        'speed_std',
        'speed_range',
        'pct_high_speed',
        'pct_motorway',
        'pct_trunk',
        'pct_primary',
        'pct_secondary',
        'pct_tertiary',
        'pct_unclassified',
        'pct_residential',
        'lane_changes_per_km',
        'priority_changes_per_km',
        'yield_priority_changes_per_km',
        'traffic_lights_per_km',
        'bearing_std',
        'turns_per_km',
        'left_yield_turns_per_km',
        # 'mean_circuity',
        'edge_length_std',
        'edges_per_km'
    ]

    return features

def _find_single(run_dir, pattern: str) -> Path:
    run_dir = Path(run_dir)
    matches = sorted(run_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"{run_dir}: expected 1 match for '{pattern}', found {len(matches)}: {[p.name for p in matches]}")
    return matches[0]

def get_ranking_matrix_path(run_dir) -> Path:
    return _find_single(run_dir, "*_ranking_matrix.csv")

def get_enriched_routes_path(run_dir) -> Path:
    return _find_single(run_dir, "*_routes_enriched.csv")

def get_clustered_ranking_matrix_path(run_dir) -> Path:
    base = get_ranking_matrix_path(run_dir)
    return base.with_name(base.name.replace("_ranking_matrix.csv", "_ranking_matrix_clustered.csv"))

def get_clustered_enriched_routes_path(run_dir) -> Path:
    base = get_enriched_routes_path(run_dir)
    return base.with_name(base.name.replace("_routes_enriched.csv", "_routes_enriched_clustered.csv"))

def get_representants_path(run_dir) -> Path:
    base = get_ranking_matrix_path(run_dir)
    return base.with_name(base.name.replace("_ranking_matrix.csv", "_clusters_representants.csv"))

def get_clusters_summary_path(run_dir) -> Path:
    base = get_ranking_matrix_path(run_dir)
    return base.with_name(base.name.replace("_ranking_matrix.csv", "_clusters_summary.txt"))

def get_action_masks_path(run_dir) -> Path:
    base = get_ranking_matrix_path(run_dir)
    return base.with_name(base.name.replace("_ranking_matrix.csv", "_action_masks.csv"))

def get_clustering_config_path(run_dir) -> Path:
    base = get_ranking_matrix_path(run_dir)
    return base.with_name(base.name.replace("_ranking_matrix.csv", "_clustering_config.json"))

def get_correlation_plot_path(run_dir) -> Path:
    base = get_ranking_matrix_path(run_dir)
    return base.with_name(base.name.replace("_ranking_matrix.csv", "_correlation.png"))

def get_dataset_name(run_dir) -> str:
    base = get_ranking_matrix_path(run_dir)
    suffix = "_ranking_matrix.csv"
    if not base.name.endswith(suffix):
        raise RuntimeError(f"Unexpected ranking matrix filename: {base.name}")
    return base.name[: -len(suffix)]

def get_num_of_clusters(run_dir):
    df = pd.read_csv(get_clustered_ranking_matrix_path(run_dir))
    return df["cluster"].nunique()

def get_representant_for_cluster_and_agent(cluster_means, agent_paths):
    """
    For a numpy array (matrix) of paths for an agent and 1D numpy array with mean for ever feature the function returns the path best fitting into the cluster.
    """
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(agent_paths)
    _, indices = nn.kneighbors(cluster_means.reshape(1, -1))
    nearest_idx = indices[0][0]
    return agent_paths[nearest_idx], nearest_idx

# ============================================================================
# Main functions
# ============================================================================

def _get_active_features(df):
    return auto_select_features(df)

# OK
def cluster(run_dir, clustering_algorithm, n_clusters, use_pca=False):
    """
    Clusters data for a given ranking matrix using a chosen clustering algorithm. 
    Optionally, performs PCA before the clustering.
    
    :param clustering_algorithm: kmeans, agglomerative, dbscan, birch, hdbscan
    :return: (labels, model, X_clustered) where X_clustered is what the model actually saw
    """

    df = pd.read_csv(get_ranking_matrix_path(run_dir))
    features = _get_active_features(df)
    X = df[features]

    if use_pca:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca_model = PCA(n_components=0.95, random_state=42)
        X_pca = pca_model.fit_transform(X_scaled)
        X_clustered = X_pca
    else:
        X_clustered = X.to_numpy()

    # birch?
    if clustering_algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif clustering_algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif clustering_algorithm == "birch":
        model = Birch(n_clusters=n_clusters, threshold=0.7)
    else:
        return None, None, None

    labels = model.fit_predict(X_clustered)

    return labels, model, X_clustered

# OK
def save_clustering_results(run_dir, labels):
    df = pd.read_csv(get_ranking_matrix_path(run_dir))
    enriched_df = pd.read_csv(get_enriched_routes_path(run_dir))

    # Add a cluster for each path in a new column
    df['cluster'] = labels
    enriched_df['cluster'] = labels
    
    # Save clustering statistics
    with open(get_clusters_summary_path(run_dir), "w") as f:
        print(df['cluster'].value_counts(), file=f)

    # Save original dfs but with an extra cluster column
    df.to_csv(get_clustered_ranking_matrix_path(run_dir), index=False)
    enriched_df.to_csv(get_clustered_enriched_routes_path(run_dir), index=False)

#!!!
def get_masked_representants(run_dir):
    """
    For each agent, selects one representative per cluster only if they have 
    a path in that cluster. Generates a mask indicating valid actions.
    """

    # Load clustered data
    df_ranked = pd.read_csv(get_clustered_ranking_matrix_path(run_dir))
    df_enriched = pd.read_csv(get_clustered_enriched_routes_path(run_dir))
    features = _get_active_features(df_ranked)

    # Get cluster centroids
    centroids = df_ranked.groupby("cluster")[features].mean()
    num_clusters = len(centroids)

    all_representants = []
    masks = []

    # For each OD, find the best/most representative path per cluster
    for (origin, destination), group in df_ranked.groupby(["origins", "destinations"]):
        od_representants = {} # keyed by cluster_id
        agent_mask = [0] * num_clusters

        # Choose representants for clusters for each OD
        for cluster in range(num_clusters):
            paths_in_cluster_df = group[group["cluster"] == cluster]
            paths_in_cluster = paths_in_cluster_df[features].to_numpy() # numpy for NearestNeighbors
            centroid = centroids.loc[cluster].to_numpy()

            if not paths_in_cluster_df.empty:
                _, local_idx = get_representant_for_cluster_and_agent(centroid, paths_in_cluster)
                # paths_in_cluster contains only a part of df_ranked 
                # (only the paths for a given OD+cluster)
                # but it keeps the original ("global") row indices of df_ranked
                # get_representant... returns a local index of paths_in_cluster (e.g., 0 to 10 if there are 10 paths)
                # to extract the actual route using this index, we must look up which global index it corresponds to
                global_idx = paths_in_cluster_df.index[local_idx]
                
                representant = df_enriched.iloc[[global_idx]].copy()
                representant["cluster"] = cluster
                od_representants[cluster] = (global_idx, representant)

                agent_mask[cluster] = 1

        # ensure the shortest path for a given OD is always there in some cluster
        # even if it's not the closest representant for this cluster
        best_path_idx = group["free_flow_time"].idxmin() # index of the lowest fft route
        selected_indices = {gidx for gidx, _ in od_representants.values()}
        if best_path_idx not in selected_indices:
            best_path_cluster = int(group.loc[best_path_idx, "cluster"])
            best_path = df_enriched.iloc[[best_path_idx]].copy()
            best_path["cluster"] = best_path_cluster
            od_representants[best_path_cluster] = (best_path_idx, best_path)

        all_representants.extend(rep for _, rep in od_representants.values())
        masks.append([origin, destination] + agent_mask)

    all_representants_concat = pd.concat(all_representants, ignore_index=True)
    all_representants_concat.to_csv(get_representants_path(run_dir), index=False)

    mask_df = pd.DataFrame(masks, columns=['origins', 'destinations'] + [f'mask_{i}' for i in range(num_clusters)])
    mask_df.to_csv(get_action_masks_path(run_dir), index=False)
    mask_cols = [f'mask_{i}' for i in range(num_clusters)]
    print(f"Masking complete. Average actions per agent: {mask_df[mask_cols].sum(axis=1).mean():.2f}")

def cluster_save(run_dir, clustering_algorithm, n_clusters, use_pca=False):
    dataset_name = get_dataset_name(run_dir)
    labels, _, _ = cluster(run_dir, clustering_algorithm, n_clusters, use_pca)
    save_clustering_results(run_dir, labels)
    get_masked_representants(run_dir)

    df = pd.read_csv(get_clustered_ranking_matrix_path(run_dir))
    features_used = _get_active_features(df)

    summary = {
        "run_name": Path(run_dir).name,
        "city_name": dataset_name,
        "algorithm": clustering_algorithm,
        "n_clusters": n_clusters,
        "use_pca": use_pca,
        "features_used": features_used,
    }
    with open(get_clustering_config_path(run_dir), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCluster Statistics:")
    print(df['cluster'].value_counts())

def evaluate(run_dir):
    """
    Intrinsic evaluation sweep to find the best algorithm and number of clusters.
    """

    results = []
    for use_pca in [True, False]:#, False]:
        for n_clusters in range(2, 7):
            for alg in ["kmeans"]:#, "agglomerative"]:#, "birch"]:
                labels, model, X_clustered = cluster(run_dir, alg, n_clusters, use_pca)

                # measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation)
                # -1 to 1; the higher the better
                s_score = silhouette_score(X_clustered, labels)

                # ratio of between-cluster variance to within-cluster variance
                # 0 to inf; the higher the better
                ch_score = calinski_harabasz_score(X_clustered, labels)

                # only KMeans has inertia
                inertia = getattr(model, "inertia_", None)

                df_ranked = pd.read_csv(get_ranking_matrix_path(run_dir))                
                df_ranked['cluster'] = labels
                action_coverage = df_ranked.groupby(["origins", "destinations"])['cluster'].nunique().mean()

                results.append({
                    "use_pca": use_pca, "alg": alg, "k": n_clusters,
                    "silhouette": round(s_score, 4), "ch": round(ch_score, 1),
                    "avg_actions": round(action_coverage, 2), 
                    "inertia": round(inertia, 1) if inertia is not None else None
                })

    return pd.DataFrame(results).sort_values(by="silhouette", ascending=False)

def check_vif(run_dir):
    dataset_name = get_dataset_name(run_dir)
    df = pd.read_csv(get_ranking_matrix_path(run_dir))
    features = _get_active_features(df)
    X = df[features]
    
    X_std = StandardScaler().fit_transform(X)
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X_std, i) for i in range(X_std.shape[1])]
    
    print(f"\n--- VIF Analysis for {dataset_name} ---")
    print(vif_data.sort_values("VIF", ascending=False))
    return vif_data

def plot_correlation_matrix(run_dir):
    """Generates a heatmap of feature correlations to identify redundant features."""
    dataset_name = get_dataset_name(run_dir)
    df = pd.read_csv(get_enriched_routes_path(run_dir))
    features = choose_features()
    corr = df[features].corr()  # type: ignore
    

    # --- Console Output for Analysis ---
    print(f"\n--- Highly correlated features (> 0.75) for {dataset_name} ---")
    # Use a mask to only look at the upper triangle (avoiding duplicates)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # upper = np.asarray(np.triu(np.ones(corr.shape), k=1).astype(bool)).nonzero()
    
    high_corr_pairs = []
    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if abs(val) > 0.75:
                high_corr_pairs.append((row, col, val))
    
    # Sort by absolute correlation strength
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if not high_corr_pairs:
        print("No highly correlated pairs found.")
    else:
        for f1, f2, val in high_corr_pairs:
            print(f"{f1:35} <-> {f2:35} : {val: .3f}")


    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title(f"Feature Correlation Matrix - {dataset_name}", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = get_correlation_plot_path(run_dir)
    plt.savefig(output_path)
    plt.close()
    print(f"Correlation heatmap saved to: {output_path}")

def diagnose_features(run_dir):
    df = pd.read_csv(get_enriched_routes_path(run_dir))
    features = choose_features()
    
    stats = []
    for f in features:
        col = df[f]
        stats.append({
            "feature": f,
            "pct_zero": (col == 0).mean() * 100,
            "std": col.std(),
            "mean": col.mean(),
            "unique_vals": col.nunique()
        })
    
    diag_df = pd.DataFrame(stats).sort_values("pct_zero", ascending=False)
    print("\n--- Feature Diagnostics ---")
    print(diag_df.to_string(index=False))

def describe_clusters(run_dir):
    df = pd.read_csv(get_clustered_enriched_routes_path(run_dir))
    features = _get_active_features(df)

    centroids = df.groupby("cluster")[features].mean() # why no id?
    return centroids

def auto_select_features(df, vif_threshold=10.0, corr_threshold=0.85):
    """
    Automatically select features by removing:
    1. Zero-variance features
    2. One member of each highly correlated pair
    3. High-VIF features (iteratively)
    """
    
    all_features = choose_features()

    # 1 - Remove zero-variance features
    features = [f for f in all_features if f in df.columns and df[f].std() > 1e-6]

    # 2 - Remove one feature from each highly correlated pair
    # Keeping the feature that has lower mean correlation
    corr = df[features].corr().abs()
    to_drop = set()
    for i in range(len(features)):
        if features[i] in to_drop: continue
        for j in range(i+1, len(features)): # skip i first features to avoid duplication
            if features[j] in to_drop: continue
            if corr.iloc[i, j] > corr_threshold:
                mean_corr_i = corr.iloc[i].drop(features[i]).mean()
                mean_corr_j = corr.iloc[j].drop(features[j]).mean()
                if mean_corr_i > mean_corr_j:
                    to_drop.add(features[i])
                else:
                    to_drop.add(features[j])
    features = [f for f in features if f not in to_drop]

    # 3 - Iteratively remove high-VIF features
    while len(features) > 2:
        X_std = StandardScaler().fit_transform(df[features])
        vifs = [variance_inflation_factor(X_std, i) for i in range(len(features))]
        max_vif = max(vifs)
        if max_vif <= vif_threshold:
            break
        worst_idx = vifs.index(max_vif)
        removed = features.pop(worst_idx)

    return features

def main():
    parser = argparse.ArgumentParser(
        description="Cluster ranking matrices of enriched route data from SUMO networks",
    )
    parser.add_argument(
        "--alg", "-a",
        help="Choose the clustering algorithm: kmeans/agglomerative",
        default="kmeans"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        help="How many clusters?",
        default=4
    )
    parser.add_argument( # false by default
        "--pca",
        help="Use pca?",
        action="store_true"
    )
    parser.add_argument(
        "--run-name", "-r",
        nargs="+",
        required=True,
        help="Run folder name inside results/ (e.g. ingolstadt_k4). All files saved flat inside results/<run-name>/."
    )
    args = parser.parse_args()

    if not args.run_name:
        print("No run-name provided. Usage: clustering.py --run-name run-name")
        return
    
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]

    for run_name in args.run_name:
        results_dir = repo_root / "results" / run_name
        if not results_dir.is_dir():
            raise RuntimeError(f"Run directory does not exist: {results_dir}")

        cluster_save(results_dir, args.alg, args.n_clusters, args.pca)
        ###
        plot_correlation_matrix(results_dir)
        diagnose_features(results_dir)
        check_vif(results_dir)
        results = evaluate(results_dir)
        print(results)
        print(describe_clusters(results_dir))

if __name__ == "__main__":
    main()
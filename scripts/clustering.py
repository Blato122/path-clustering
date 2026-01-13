import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
import hdbscan

# ============================================================================
# The script generates clustered_enriched_routes.cs, clustered_ranking_matrix.csv, representants_per_cluster.csv and clusters_summary.txt for a city specified in main().
# This script depends on the premis that in all fives (enriched_routes, ranking_matrix, clustered_enriched_routes and clustered_ranking_matrix) under the same index we have the same path. 
# Also  it requires one (origin,destination) pair per agent and agent_id in ranking_matrix.
# ============================================================================

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
        'mean_circuity',
        'edge_length_std',
        'edges_per_km'
    ]
    return features

def get_ranking_matrix_path(city_name):
    return f"../results/ranking_matrices/{city_name}_ranking_matrix.csv"

def get_enriched_routes_path(city_name):
    return f"../results/enriched_routes/{city_name}_routes_enriched.csv"

def get_clustered_ranking_matrix_path(city_name):
    return f"../results/clustered_routes/{city_name}_ranking_matrix_clustered.csv"

def get_clustered_enriched_routes_path(city_name):
    return f"../results/clustered_routes/{city_name}_routes_enriched_clustered.csv"

def get_representants_path(city_name):
    return f"../results/clusters_representants/{city_name}_clusters_representants.csv"

def get_clusters_summary_path(city_name):
    return f"../results/clustering_summary/{city_name}_clusters_summary.txt"

def get_num_of_clusters(city_name):
    df = pd.read_csv(get_clustered_ranking_matrix_path(city_name))
    return df["cluster"].nunique()

# ============================================================================
# 
# ============================================================================


def agent_clustering_coverage(city_name, agent_nr):
    """
    Check how many paths an agent has per cluster. Returnas procentages for eg. (0.25, 0.4, 0.1, 0.25) means that an agent has 25 precent paths in cluster 1, 40 present in cluster 2, etc.
    
    """
    df = pd.read_csv(get_clustered_ranking_matrix_path(city_name))
    tab_paths_per_cluster = df[df["agent_id"] == agent_nr]["cluster"].value_counts()
    sum = 0

    for cluster in tab_paths_per_cluster.index:
        sum+= tab_paths_per_cluster[cluster]
        

    paths_per_cluster = []
    # tab_paths_per_cluster = tab_paths_per_cluster.sort_index()
    # #print(tab_paths_per_cluster)

    for cluster in range(get_num_of_clusters(city_name)):
        paths_per_cluster.append(tab_paths_per_cluster.get(cluster, 0)/sum)

    #print(paths_per_cluster)

    return paths_per_cluster

def get_representant_for_cluster_and_agent(cluster_means, agent_paths):
    """
    For a numpy array (matrix) of paths for an agent and 1D numpy array with mean for ever feature the function returns the path best fitting into the cluster.
    """
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(agent_paths)

    distances, indices = nn.kneighbors(cluster_means.reshape(1, -1))

    nearest_idx = indices[0][0]
    # print("Nearest index:", nearest_idx)
    # print("Distance:", distances[0][0])
    # print("Closest path:", agent_paths[nearest_idx])
    return agent_paths[nearest_idx], nearest_idx

def get_representants_for_agent(clusters_means, agent_paths):  
    # features only np.array and np.array
    ans = []
    for cluster in clusters_means:
        path, index = get_representant_for_cluster_and_agent(cluster, agent_paths)
        ans.append(index)
    return ans

# ============================================================================
# Main functions
# ============================================================================


def cluster(city_name, clustering_algorithm):
    """
    Clusters data for a given ranking matrix using a chosen clustering algorithm. Saves the results in folder results/clustered_routes.
    
    Args:
        city_name
        clustering_algorithm: kmeans, birch
    """

    df = pd.read_csv(get_ranking_matrix_path(city_name))

    enriched_df = pd.read_csv(get_enriched_routes_path(city_name))
    #df.info()
    #df.columns
    features = choose_features()
    X = df[features]

    if(clustering_algorithm == 'kmeans'):
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X)

        labels = kmeans.labels_
        df['cluster'] = labels
        enriched_df['cluster'] = labels
        with open(get_clusters_summary_path(city_name), "w") as f:
            print("Clustering algorithm: K-Means", file=f)
            print(df['cluster'].value_counts(), file=f)
        print(df['cluster'].value_counts())
        df.to_csv(get_clustered_ranking_matrix_path(city_name), index=False)
        enriched_df.to_csv(get_clustered_enriched_routes_path(city_name), index=False)


    # elif(clustering_algorithm == "DBSCAN"):
    #     dbscan = DBSCAN(eps=5, min_samples=100)
    #     labels = dbscan.fit_predict(X)
    #     df['cluster'] = labels
    #     enriched_df['cluster'] = labels
    #     with open(get_clusters_summary_path(city_name), "w") as f:
    #         print("Clustering algorithm: DBSCAN", file=f)
    #         print(df['cluster'].value_counts(), file=f)
    #     print(df['cluster'].value_counts())
    #     df.to_csv(get_clustered_ranking_matrix_path(city_name), index=False)
    #     enriched_df.to_csv(get_clustered_enriched_routes_path(city_name), index=False)

    elif(clustering_algorithm == "birch"):
        birch = Birch(n_clusters=5, threshold=0.7)
        labels = birch.fit_predict(X)
        df['cluster'] = labels
        enriched_df['cluster'] = labels
        with open(get_clusters_summary_path(city_name), "w") as f:
            print("Clustering algorithm: Birch", file=f)
            print(df['cluster'].value_counts(), file=f)
        print(df['cluster'].value_counts())
        df.to_csv(get_clustered_ranking_matrix_path(city_name), index=False)
        enriched_df.to_csv(get_clustered_enriched_routes_path(city_name), index=False)

    # elif(clustering_algorithm == "hdbscan"):   # DENSITY - BASED CLUSTERING WON'T WORK FOR OUR DATA!
    #     clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
    #     labels = clusterer.fit_predict(X)
    #     df['cluster'] = labels
    #     enriched_df['cluster'] = labels
    #     with open(get_clusters_summary_path(city_name), "w") as f:
    #         print("Clustering algorithm: HDBSCAN", file=f)
    #         print(df['cluster'].value_counts(), file=f)
    #     print(df['cluster'].value_counts())
    #     df.to_csv(get_clustered_ranking_matrix_path(city_name), index=False)
    #     enriched_df.to_csv(get_clustered_enriched_routes_path(city_name), index=False)
    else:
        print(f"There is no {clustering_algorithm} clustering algorithm.")

    return



def clustering_coverage(city_name):
    """
    Checks whether each agent has a path in each cluster.
    
    """
    df = pd.read_csv(get_clustered_ranking_matrix_path(city_name))
    #print(df["agent_id"].nunique())
    agent_count = df["agent_id"].nunique()
    tab = []
    for i in range(agent_count):
        tab.append(agent_clustering_coverage(city_name,i))

    arr = np.array(tab)
    column_mins = np.min(arr, axis=0)
    with open(get_clusters_summary_path(city_name), "a") as f:
        print("Number of agents:", agent_count, file=f)
        print("Minimum coverage for each cluster:", file=f)
        print(column_mins, file=f)
        mask = arr == 0
        counts = np.sum(mask, axis=0)
        print("How many agents have coverage < 0.02 for each cluster:", file=f)
        print(counts, file=f)

        column_max = np.max(arr,axis=0)
        print("Maximum coverage for each cluster:", file=f)
        print(column_max, file=f)

    return

def clusters_summary(city_name):
    """
    Gives characteristics for each cluster: mean for each feature.
    """

    df = pd.read_csv(get_clustered_ranking_matrix_path(city_name))

    features = choose_features()
    features.append("cluster")
    X = df[features]

    cluster_means = X.groupby("cluster").mean()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    #print(cluster_means)

    return cluster_means



def get_representants(city_name):
    """
    For each agent chooses one path best fitting into the cluster (when an agent does not have a path in cluster it chooses the closest path form other clusters). Best fitting paths are in ../results/best_paths_per_cluster/

    """
    cluster_means = clusters_summary(city_name)
    cluster_means_numpy = cluster_means.to_numpy()
    #print(cluster_means_numpy)

    df = pd.read_csv(get_clustered_ranking_matrix_path(city_name))
    tables_by_agent = {
        agent_id: group.copy()
        for agent_id, group in df.groupby("agent_id")
    }
    rows_per_agent = {
        agent_id: len(table)
        for agent_id, table in tables_by_agent.items()
    }
    prefix_sum = {}
    running_sum = 0

    for agent_id in sorted(rows_per_agent):
        running_sum += rows_per_agent[agent_id]
        prefix_sum[agent_id] = running_sum


    features = choose_features()

    df_enriched = pd.read_csv(get_clustered_enriched_routes_path(city_name))
    
    representants = []
    for agent_id in tables_by_agent.keys():
        # print(agent_id)
        tab_numpy = tables_by_agent[agent_id][features].to_numpy()
        # print(tables_by_agent[agent_id])
        # print(tab_numpy)
        ans = get_representants_for_agent(cluster_means_numpy,tab_numpy)
        c = 0
        for i in ans:
            pref = 0
            if agent_id-1 >=0: 
                pref = prefix_sum[agent_id-1]
            path = df_enriched.iloc[[pref+i]]
            path = path.copy()
            path["representant_of_cluster"] = c
            path.insert(0,"agent_id", agent_id)
            c +=1
            representants.append(path)    

    all_representants = pd.concat(representants, ignore_index=True)
    all_representants.to_csv(get_representants_path(city_name), index=False)

    return

# ============================================================================
# Main
# ============================================================================

def main():
    city_name = "saint_arnoult"
    cluster(city_name,"kmeans")
    clustering_coverage(city_name)
    cluster_means = clusters_summary(city_name)
    with open(get_clusters_summary_path(city_name), "a") as f:
        print(cluster_means, file=f)

    get_representants(city_name)


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .generation import CLUSTER_FEATURES, load_config


def discover_paths(run_dir: Path) -> dict[str, Path | str]:
    run_dir = Path(run_dir)
    ranking_files = sorted(run_dir.glob("*_ranking_matrix.csv"))
    if len(ranking_files) != 1:
        raise RuntimeError(
            f"{run_dir}: expected one '*_ranking_matrix.csv', "
            f"found {[path.name for path in ranking_files]}"
        )

    ranking = ranking_files[0]
    dataset_name = ranking.name.removesuffix("_ranking_matrix.csv")
    paths = {
        "dataset_name": dataset_name,
        "ranking": ranking,
        "enriched": run_dir / f"{dataset_name}_routes_enriched.csv",
        "representants": run_dir / f"{dataset_name}_clusters_representants.csv",
        "masks": run_dir / f"{dataset_name}_action_masks.csv",
        "config": run_dir / f"{dataset_name}_clustering_config.json",
        "route_set_config": run_dir / f"{dataset_name}_route_set_config.json",
        "diagnostics": run_dir / f"{dataset_name}_clustering_diagnostics.json",
    }
    if not paths["enriched"].is_file():
        raise FileNotFoundError(f"Enriched routes file not found: {paths['enriched']}")
    return paths


def load_inputs(paths: dict[str, Path | str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = pd.read_csv(paths["ranking"])
    enriched = pd.read_csv(paths["enriched"])

    for name, data in (("ranking matrix", ranked), ("enriched routes", enriched)):
        if "route_id" not in data.columns:
            raise ValueError(f"{name} is missing route_id")
        if data["route_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate route_id values")

    if set(ranked["route_id"]) != set(enriched["route_id"]):
        raise ValueError("Ranking matrix and enriched routes contain different route IDs")

    for name, data in (("ranking matrix", ranked), ("enriched routes", enriched)):
        missing_features = set(CLUSTER_FEATURES) - set(data.columns)
        if missing_features:
            raise ValueError(
                f"{name} is missing features: {sorted(missing_features)}"
            )
    return ranked, enriched


def select_features(
    df: pd.DataFrame,
    vif_threshold: float = 10.0,
    corr_threshold: float = 0.85,
) -> tuple[list[str], dict]:
    """
    Select features by removing:
    1. Zero-variance features
    2. One member of each highly correlated pair
    3. High-VIF features iteratively
    """
    zero_variance = [
        feature
        for feature in CLUSTER_FEATURES
        if df[feature].std() <= 1e-6
    ]
    features = [
        feature for feature in CLUSTER_FEATURES if feature not in zero_variance
    ]

    correlation_removals = []
    corr = df[features].corr().abs()
    to_drop = set()
    for i, feature_i in enumerate(features):
        if feature_i in to_drop:
            continue
        for feature_j in features[i + 1:]:
            if feature_j in to_drop:
                continue
            correlation = corr.loc[feature_i, feature_j]
            if correlation <= corr_threshold:
                continue

            mean_corr_i = corr.loc[feature_i].drop(feature_i).mean()
            mean_corr_j = corr.loc[feature_j].drop(feature_j).mean()
            removed = feature_i if mean_corr_i > mean_corr_j else feature_j
            kept = feature_j if removed == feature_i else feature_i
            to_drop.add(removed)
            correlation_removals.append({
                "removed": removed,
                "kept": kept,
                "correlation": float(correlation),
            })

    features = [feature for feature in features if feature not in to_drop]

    vif_removals = []
    while len(features) > 2:
        scaled = StandardScaler().fit_transform(df[features])
        vifs = [
            float(variance_inflation_factor(scaled, i))
            for i in range(len(features))
        ]
        comparison_vifs = [
            value if np.isfinite(value) else math.inf for value in vifs
        ]
        max_vif = max(comparison_vifs)
        if max_vif <= vif_threshold:
            break

        worst_idx = comparison_vifs.index(max_vif)
        vif_removals.append({
            "removed": features[worst_idx],
            "vif": vifs[worst_idx],
        })
        features.pop(worst_idx)

    if not features:
        raise ValueError("No usable clustering features remain after selection")

    details = {
        "zero_variance_removed": zero_variance,
        "correlation_removals": correlation_removals,
        "vif_removals": vif_removals,
    }
    return features, details


def fit_clustering(
    ranked: pd.DataFrame,
    features: list[str],
    algorithm: str,
    n_clusters: int,
) -> tuple[np.ndarray, object, np.ndarray]:
    if not 1 < n_clusters < len(ranked):
        raise ValueError(
            f"n_clusters must be between 2 and {len(ranked) - 1}, got {n_clusters}"
        )

    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    elif algorithm == "birch":
        model = Birch(n_clusters=n_clusters, threshold=0.7)
    else:
        raise ValueError(
            f"Unknown clustering algorithm '{algorithm}'. "
            "Choose kmeans, agglomerative, or birch."
        )

    values = ranked[features].to_numpy()
    labels = model.fit_predict(values)
    label_map = {
        old_label: new_label
        for new_label, old_label in enumerate(sorted(np.unique(labels)))
    }
    labels = np.array([label_map[label] for label in labels], dtype=int)
    return labels, model, values


def calculate_metrics(
    ranked: pd.DataFrame,
    labels: np.ndarray,
    values: np.ndarray,
) -> dict:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(values):
        silhouette = None
        ch_score = None
    else:
        silhouette = float(silhouette_score(values, labels))
        ch_score = float(calinski_harabasz_score(values, labels))

    clustered = ranked[["origins", "destinations"]].copy()
    clustered["cluster"] = labels
    avg_actions = float(
        clustered.groupby(["origins", "destinations"])["cluster"].nunique().mean()
    )
    cluster_sizes = {
        str(cluster): int(count)
        for cluster, count in pd.Series(labels).value_counts().sort_index().items()
    }
    return {
        "silhouette": silhouette,
        "calinski_harabasz": ch_score,
        "average_actions_per_od": avg_actions,
        "cluster_sizes": cluster_sizes,
    }


def select_representants(
    ranked: pd.DataFrame,
    enriched: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each OD, select one representative for every cluster containing one of
    its routes and generate a mask indicating the available clusters.
    """
    clustered = ranked.copy()
    clustered["cluster"] = labels
    enriched_by_id = enriched.set_index("route_id", drop=False)
    centroids = clustered.groupby("cluster")[features].mean()
    num_clusters = len(centroids)

    all_representants = []
    masks = []

    for (origin, destination), group in clustered.groupby(
        ["origins", "destinations"]
    ):
        od_representants = {}
        agent_mask = [0] * num_clusters

        for cluster in range(num_clusters):
            candidates = group[group["cluster"] == cluster]
            if candidates.empty:
                continue

            distances = np.linalg.norm(
                candidates[features].to_numpy() - centroids.loc[cluster].to_numpy(),
                axis=1,
            )
            route_id = candidates.iloc[int(np.argmin(distances))]["route_id"]
            representant = enriched_by_id.loc[[route_id]].copy()
            representant["cluster"] = cluster
            od_representants[cluster] = (route_id, representant)
            agent_mask[cluster] = 1

        # Always keep the shortest route, even if another route was closer to that cluster's centroid
        od_enriched = enriched_by_id.loc[group["route_id"]]
        shortest_route_id = od_enriched["free_flow_time"].idxmin()
        selected_route_ids = {
            route_id for route_id, _ in od_representants.values()
        }
        if shortest_route_id not in selected_route_ids:
            shortest_cluster = int(
                group.loc[group["route_id"] == shortest_route_id, "cluster"].iloc[0]
            )
            shortest_route = enriched_by_id.loc[[shortest_route_id]].copy()
            shortest_route["cluster"] = shortest_cluster
            od_representants[shortest_cluster] = (
                shortest_route_id,
                shortest_route,
            )

        all_representants.extend(
            representant for _, representant in od_representants.values()
        )
        masks.append([origin, destination, *agent_mask])

    representants = pd.concat(all_representants, ignore_index=True)
    masks_df = pd.DataFrame(
        masks,
        columns=[
            "origins",
            "destinations",
            *[f"mask_{cluster}" for cluster in range(num_clusters)],
        ],
    )
    return representants, masks_df


def calculate_vif(df: pd.DataFrame, features: list[str]) -> list[dict]:
    if len(features) < 2:
        return []
    scaled = StandardScaler().fit_transform(df[features])
    return [
        {
            "feature": feature,
            "vif": float(variance_inflation_factor(scaled, i)),
        }
        for i, feature in enumerate(features)
    ]


def build_diagnostics(
    ranked: pd.DataFrame,
    enriched: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
    feature_selection: dict,
    algorithm: str,
) -> dict:
    feature_stats = []
    for feature in CLUSTER_FEATURES:
        values = enriched[feature]
        feature_stats.append({
            "feature": feature,
            "percent_zero": float((values == 0).mean() * 100),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "unique_values": int(values.nunique()),
        })

    correlation = enriched[CLUSTER_FEATURES].corr()
    high_correlations = []
    for i, feature_i in enumerate(CLUSTER_FEATURES):
        for feature_j in CLUSTER_FEATURES[i + 1:]:
            value = correlation.loc[feature_i, feature_j]
            if abs(value) > 0.75:
                high_correlations.append({
                    "feature_1": feature_i,
                    "feature_2": feature_j,
                    "correlation": float(value),
                })
    high_correlations.sort(
        key=lambda item: abs(item["correlation"]),
        reverse=True,
    )

    evaluation = []
    max_clusters = min(6, len(ranked) - 1)
    for n_clusters in range(2, max_clusters + 1):
        sweep_labels, model, values = fit_clustering(
            ranked,
            features,
            algorithm,
            n_clusters,
        )
        metrics = calculate_metrics(ranked, sweep_labels, values)
        evaluation.append({
            "algorithm": algorithm,
            "n_clusters": n_clusters,
            **metrics,
            "inertia": (
                float(model.inertia_) if hasattr(model, "inertia_") else None
            ),
        })

    clustered_enriched = enriched.merge(
        pd.DataFrame({
            "route_id": ranked["route_id"],
            "cluster": labels,
        }),
        on="route_id",
        how="left",
        validate="one_to_one",
    )
    cluster_means = (
        clustered_enriched.groupby("cluster")[CLUSTER_FEATURES]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "feature_selection": {
            "selected": features,
            **feature_selection,
        },
        "feature_statistics": feature_stats,
        "correlation_matrix": correlation.to_dict(),
        "high_correlations": high_correlations,
        "vif": calculate_vif(ranked, features),
        "evaluation": evaluation,
        "cluster_feature_means": cluster_means,
    }


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_feature_pruning(feature_selection: dict) -> list[dict]:
    summary = []
    for feature in feature_selection.get("zero_variance_removed", []):
        summary.append({
            "feature": feature,
            "reason": "zero_variance",
        })
    for removal in feature_selection.get("correlation_removals", []):
        summary.append({
            "feature": removal["removed"],
            "reason": "correlation",
            "kept": removal["kept"],
            "correlation": removal["correlation"],
        })
    for removal in feature_selection.get("vif_removals", []):
        summary.append({
            "feature": removal["removed"],
            "reason": "vif",
            "vif": removal["vif"],
        })
    return summary


def run_clustering(
    run_dir: Path,
    algorithm: str,
    n_clusters: int,
    diagnostics: bool = False,
    route_set_config: dict | None = None,
) -> tuple[Path, Path, Path, Path | None]:
    paths = discover_paths(run_dir)
    ranked, enriched = load_inputs(paths)
    features, feature_selection = select_features(ranked)
    labels, model, values = fit_clustering(
        ranked,
        features,
        algorithm,
        n_clusters,
    )
    metrics = calculate_metrics(ranked, labels, values)
    representants, masks = select_representants(
        ranked,
        enriched,
        labels,
        features,
    )

    representants.to_csv(paths["representants"], index=False)
    masks.to_csv(paths["masks"], index=False)

    config = {
        "run_name": Path(run_dir).name,
        "city_name": paths["dataset_name"],
        "algorithm": algorithm,
        "n_clusters": n_clusters,
        "features_used": features,
        "feature_pruning": summarize_feature_pruning(feature_selection),
        **metrics,
        "inertia": float(model.inertia_) if hasattr(model, "inertia_") else None,
    }
    with open(paths["config"], "w") as file:
        json.dump(make_json_safe(config), file, indent=2, allow_nan=False)

    if route_set_config is not None:
        with open(paths["route_set_config"], "w") as file:
            json.dump(
                make_json_safe(route_set_config),
                file,
                indent=2,
                allow_nan=False,
            )

    if diagnostics:
        diagnostic_data = build_diagnostics(
            ranked,
            enriched,
            labels,
            features,
            feature_selection,
            algorithm,
        )
        with open(paths["diagnostics"], "w") as file:
            json.dump(
                make_json_safe(diagnostic_data),
                file,
                indent=2,
                allow_nan=False,
            )

    sizes = ", ".join(
        f"{cluster}: {count}"
        for cluster, count in metrics["cluster_sizes"].items()
    )
    silhouette = metrics["silhouette"]
    ch_score = metrics["calinski_harabasz"]
    silhouette_text = f"{silhouette:.4f}" if silhouette is not None else "N/A"
    ch_text = f"{ch_score:.1f}" if ch_score is not None else "N/A"
    print(
        f"Clustered {len(ranked)} routes into {len(metrics['cluster_sizes'])} clusters.\n"
        f"Cluster sizes: {sizes}\n"
        f"Silhouette: {silhouette_text}\n"
        f"Calinski-Harabasz: {ch_text}\n"
        f"Average valid actions per OD: {metrics['average_actions_per_od']:.2f}"
    )
    return (
        Path(paths["representants"]),
        Path(paths["masks"]),
        Path(paths["config"]),
        Path(paths["diagnostics"]) if diagnostics else None,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cluster ranking matrices of enriched route data from SUMO networks",
    )
    parser.add_argument(
        "--alg", "-a",
        choices=["kmeans", "agglomerative", "birch"],
        default=None,
        help="Clustering algorithm.",
    )
    parser.add_argument(
        "--n_clusters", "-n",
        type=int,
        default=None,
        help="Number of clusters.",
    )
    parser.add_argument(
        "--diag", "-d",
        action="store_true",
        default=None,
        help="Save additional clustering diagnostics as JSON.",
    )
    parser.add_argument(
        "--config", "-c",
        default="cluster-default",
        help="Explicit config path or bundled config name.",
    )
    parser.add_argument(
        "--run-name", "-r",
        nargs="+",
        help="Run folder names inside --results-dir.",
    )
    parser.add_argument(
        "--run-dir",
        nargs="+",
        type=Path,
        help="One or more explicit run directories.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Base directory used with --run-name.",
    )
    args = parser.parse_args()

    run_dirs = [path.resolve() for path in (args.run_dir or [])]
    if args.run_name and args.results_dir is None:
        parser.error("--results-dir is required when using --run-name")

    results_root = args.results_dir.resolve() if args.results_dir else None
    run_dirs.extend(results_root / name for name in (args.run_name or []))
    if not run_dirs:
        parser.error("at least one --run-name or --run-dir is required")

    config, config_source = load_config(args.config)
    clustering_settings = config.get("clustering", config)
    if not isinstance(clustering_settings, dict):
        raise ValueError(
            f"Clustering config must be a JSON object: {config_source}"
        )

    algorithm = args.alg or clustering_settings.get("algorithm", "kmeans")
    n_clusters = args.n_clusters or clustering_settings.get("n_clusters", 4)
    diagnostics = (
        args.diag
        if args.diag is not None
        else clustering_settings.get("diagnostics", False)
    )

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise RuntimeError(f"Run directory does not exist: {run_dir}")
        run_clustering(
            run_dir,
            algorithm,
            n_clusters,
            diagnostics=diagnostics,
        )


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from pathlib import Path
import argparse

from .generation import (
    enrich_routes,
    generate_csv_routes,
    generate_feature_file,
    get_route_generator,
    load_config,
)
from .clustering import run_clustering


@dataclass(frozen=True)
class ClusteredRouteOutputs:
    """
    This dataclass represents the contract for the path generation
    and clustering output files, later used by URB scripts.
    """
    representants: Path
    action_masks: Path
    clustering_config: Path
    diagnostics: Path | None = None


def _next_available_route_set_dir(network_folder: Path, route_set: str) -> Path:
    if Path(route_set).name != route_set:
        raise ValueError("route_set must be a directory name, not a path")

    clustered_routes_dir = network_folder / "clustered_routes"
    output_dir = clustered_routes_dir / route_set

    if not output_dir.exists() or not any(output_dir.iterdir()):
        return output_dir

    suffix = 2
    while True:
        candidate = clustered_routes_dir / f"{route_set}-{suffix}"
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        suffix += 1


# Do not call main() functions (intended for CLI use)
def generate_clustered_routes(
    network_name: str,
    network_root: Path,
    output_dir: Path,
    config: str | Path = "run-default",
) -> ClusteredRouteOutputs:
    network_root = Path(network_root).resolve()
    output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config, config_source = load_config(config)
    generator = pipeline_config.get("generator", "clustering")
    path_gen_kwargs = pipeline_config.get("path_gen_kwargs")
    clustering_settings = pipeline_config.get("clustering", {})

    if not isinstance(path_gen_kwargs, dict):
        raise ValueError(
            f"Pipeline config must contain 'path_gen_kwargs': {config_source}"
        )
    if not isinstance(clustering_settings, dict):
        raise ValueError(
            f"Pipeline config 'clustering' must be an object: {config_source}"
        )

    algorithm = clustering_settings.get("algorithm", "kmeans")
    n_clusters = clustering_settings.get("n_clusters", 4)
    diagnostics = clustering_settings.get("diagnostics", False)

    if not isinstance(generator, str):
        raise ValueError(
            f"Pipeline config 'generator' must be a string: {config_source}"
        )
    if not isinstance(algorithm, str):
        raise ValueError(
            f"Pipeline config clustering algorithm must be a string: {config_source}"
        )
    if not isinstance(n_clusters, int):
        raise ValueError(
            f"Pipeline config n_clusters must be an integer: {config_source}"
        )
    if not isinstance(diagnostics, bool):
        raise ValueError(
            f"Pipeline config diagnostics must be a boolean: {config_source}"
        )

    route_generator = get_route_generator(generator)

    generate_csv_routes(
        name=network_name,
        net_dir=network_root,
        path_gen_kwargs=path_gen_kwargs,
        run_dir=output_dir,
        route_generator=route_generator,
    )

    generate_feature_file(
        name=network_name,
        net_dir=network_root,
        run_dir=output_dir,
    )

    enrich_routes(
        name=network_name,
        run_dir=output_dir,
    )

    representants, action_masks, config_path, diagnostics_path = run_clustering(
        run_dir=output_dir,
        algorithm=algorithm,
        n_clusters=n_clusters,
        diagnostics=diagnostics,
        route_set_config={
            "config_source": config_source,
            **pipeline_config,
        },
    )

    return ClusteredRouteOutputs(
        representants=representants,
        action_masks=action_masks,
        clustering_config=config_path,
        diagnostics=diagnostics_path,
    )


def generate_named_route_set(
    network_folder: Path,
    route_set: str,
    config: str | Path = "run-default",
) -> ClusteredRouteOutputs:
    """
    Generate a named URB route set under:
      <network_folder>/clustered_routes/<route_set>

    If that directory already exists and contains files, a numeric suffix is
    appended to avoid overwriting previous experimental outputs.
    """
    network_folder = Path(network_folder).resolve()
    output_dir = _next_available_route_set_dir(network_folder, route_set)

    return generate_clustered_routes(
        network_name=network_folder.name,
        network_root=network_folder.parent,
        output_dir=output_dir,
        config=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network-folder",
        type=Path,
        required=True,
        help="URB network folder, e.g. URB/networks/ingolstadt_custom.",
    )
    parser.add_argument(
        "--route-set",
        required=True,
        help="Name written under <network-folder>/clustered_routes/.",
    )
    parser.add_argument(
        "--config",
        default="run-default",
        help="Explicit config path or bundled config name.",
    )
    args = parser.parse_args()

    outputs = generate_named_route_set(
        network_folder=args.network_folder,
        route_set=args.route_set,
        config=args.config,
    )

    print(f"Route set directory: {outputs.representants.parent}")
    print(f"Representants: {outputs.representants}")
    print(f"Action masks: {outputs.action_masks}")
    print(f"Clustering config: {outputs.clustering_config}")
    if outputs.diagnostics is not None:
        print(f"Diagnostics: {outputs.diagnostics}")


if __name__ == "__main__":
    main()

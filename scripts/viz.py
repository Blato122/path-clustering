import matplotlib
matplotlib.use("Agg")

import argparse
import json
import random
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import janux as jx
from PIL import Image, ImageDraw, ImageFont
from janux.visualizers.visualization_utils import create_graph, get_colors, parse_network_files, shift_edge_by_offset


def parse_path_cell(value) -> list[str]:
    s = "" if value is None else str(value)
    s = s.strip()
    for ch in ["[", "]", "(", ")", "{", "}", "'"]:
        s = s.replace(ch, "")
    s = s.replace('"', "")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def safe_od_tag(origin: str, destination: str, max_len: int = 40) -> str:
    tag = f"{origin}->{destination}"
    if len(tag) <= max_len:
        return tag
    return f"{str(origin)[:max_len//2]}…->{str(destination)[-max_len//2:]}"


def city_from_routes_file(p: Path) -> str:
    # Handles:
    # - "<city>_routes.csv", "<city>_routes_<suffix>.csv"
    # - "<city>_clusters_representants.csv"
    stem = p.stem
    if "_clusters_representants" in stem:
        return stem.split("_clusters_representants", 1)[0]
    if "_routes" in stem:
        return stem.split("_routes", 1)[0]
    return stem


def label_from_routes_file(p: Path) -> str:
    # label shown in headers / selection logic
    stem = p.stem
    if "_clusters_representants" in stem:
        # keep it explicit in the header
        return "clusters_representants"
    if "_routes_" in stem:
        return stem.split("_routes_", 1)[1]
    if stem.endswith("_routes"):
        return "routes"
    return stem


def read_od_pairs(od_json_path: Path) -> list[tuple[str, str]]:
    with od_json_path.open("r") as f:
        data = json.load(f)
    origins = [str(x) for x in data.get("origins", [])]
    destinations = [str(x) for x in data.get("destinations", [])]
    return list(product(origins, destinations))


def is_results_run_dir(p: Path) -> bool:
    return p.is_dir() and not p.name.startswith(".") and p.name != "visualizations"


def pick_routes_file(files: list[Path], only_label_contains: str | None) -> Path | None:
    candidates = []
    for f in files:
        if only_label_contains and (only_label_contains not in f.stem):
            continue
        candidates.append(f)

    if not candidates:
        return None

    def score(f: Path) -> tuple[int, str]:
        label = label_from_routes_file(f)
        s = 0
        # Prefer "most processed" variants if present
        if "clusters_representants" in f.stem:
            s += 1000
        if "enriched_clustered" in label:
            s += 300
        if "clustered" in label:
            s += 200
        if "enriched" in label:
            s += 100
        if label == "routes":
            s += 10
        return (s, label)

    # Highest score wins; tie-break by label then filename
    candidates_sorted = sorted(candidates, key=lambda f: (score(f)[0], score(f)[1], f.name), reverse=True)
    return candidates_sorted[0]


def visualize_paths_with_labels(
    graph: nx.DiGraph,
    paths: list[list[str]],
    origin_edge: str,
    destination_edge: str,
    path_labels: list[str] | None = None,
    show: bool = True,
    save_file_path: str | None = None,
    title: str = "Path Visualization",
    cmap_names: list[str] = ['Reds', 'Blues', 'Greens', 'Oranges', 'Greys', 'Purples', 'copper', 'pink'],
    offsets: list | None = None,
    fig_size: tuple[int, int] = (12, 8),
    autocrop: bool = True,
    autocrop_margin: float = 10,
    xcrop: tuple[float, float] | None = None,
    ycrop: tuple[float, float] | None = None,
    node_size: int = 10,
    node_color: str = 'lightblue',
    path_width: int = 4,
) -> None:
    if offsets is None:
        margin_between_paths = path_width / 5
        offsets = [((path_width / 2) + margin_between_paths) + ((path_width + margin_between_paths) * i) for i in range(len(paths))]

    plt.figure(figsize=fig_size)

    node_positions = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, node_positions, node_size=node_size, node_color=node_color, style='--', edge_color='gray', arrows=False)

    origin_coords, dest_coords = None, None
    for source_node, target_node, edge_id in graph.edges(data=True):
        if edge_id['edge_id'] == origin_edge:
            origin_coords = (source_node, target_node)
            if dest_coords is not None:
                break
        elif edge_id['edge_id'] == destination_edge:
            dest_coords = (source_node, target_node)
            if origin_coords is not None:
                break

    assert origin_coords is not None, f"Origin {origin_edge} is not found in the network."
    assert dest_coords is not None, f"Destination {destination_edge} is not found in the network."

    x_max, x_min, y_max, y_min = float('-inf'), float('inf'), float('-inf'), float('inf')
    artists = []

    for path_idx, path_edges in enumerate(paths):
        path_edges_graph = {
            data_dict['edge_id']: (source, target)
            for source, target, data_dict in graph.edges(data=True)
            if data_dict['edge_id'] in path_edges
        }
        cmap_name = cmap_names[path_idx % len(cmap_names)]
        colors = get_colors(len(path_edges), cmap_name)
        label = path_labels[path_idx] if path_labels and path_idx < len(path_labels) else f"Path {path_idx}"

        for edge_id, (source_node, target_node) in path_edges_graph.items():
            new_pos = shift_edge_by_offset(node_positions, source_node, target_node, offsets[path_idx])
            color = colors[path_edges.index(edge_id)]
            nx.draw_networkx_edges(graph, new_pos, edgelist=[(source_node, target_node)], edge_color=[color], width=path_width)

            if autocrop:
                x_max = max(x_max, new_pos[source_node][0], new_pos[target_node][0])
                x_min = min(x_min, new_pos[source_node][0], new_pos[target_node][0])
                y_max = max(y_max, new_pos[source_node][1], new_pos[target_node][1])
                y_min = min(y_min, new_pos[source_node][1], new_pos[target_node][1])

        artist = plt.Line2D([0], [0], color=colors[len(colors) // 2], lw=path_width, label=label)
        artists.append(artist)

    if autocrop:
        x_range_length = x_max - x_min
        y_range_length = y_max - y_min
        cropped_aspect_ratio = y_range_length / x_range_length

        fig_width, fig_height = fig_size
        fig_aspect_ratio = fig_height / fig_width
        if cropped_aspect_ratio > fig_aspect_ratio:
            x_range_length_new = (cropped_aspect_ratio / fig_aspect_ratio) * x_range_length
            difference = x_range_length_new - x_range_length
            x_max += difference / 2
            x_min -= difference / 2
        else:
            y_range_length_new = (fig_aspect_ratio / cropped_aspect_ratio) * y_range_length
            difference = y_range_length_new - y_range_length
            y_max += difference / 2
            y_min -= difference / 2

        plt.xlim(x_min - autocrop_margin, x_max + autocrop_margin)
        plt.ylim(y_min - autocrop_margin, y_max + autocrop_margin)
    else:
        if xcrop is not None:
            plt.xlim(xcrop)
        if ycrop is not None:
            plt.ylim(ycrop)

    plt.title(title)
    plt.legend(handles=artists)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(title)

    if save_file_path is not None:
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    plt.close()


def show_multi_routes_with_labels(
    nod_file_path: str,
    edg_file_path: str,
    paths: list[list[str]],
    origin: str,
    destination: str,
    path_labels: list[str] | None = None,
    **kwargs,
):
    nodes, edges = parse_network_files(nod_file_path, edg_file_path)
    graph = create_graph(nodes, edges)
    visualize_paths_with_labels(graph, paths, origin, destination, path_labels=path_labels, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-ods", type=int, default=2, help="Number of OD pairs plotted per grid.")
    ap.add_argument("--seed", type=int, default=0, help="Seed for reproducible OD sampling (default: 0).")
    ap.add_argument("--cities", type=str, nargs="*", default=None, help="Optional city filter.")
    ap.add_argument(
        "--name",
        type=str,
        nargs="*",
        default=None,
        help="Optional results run folder filter (e.g. --name ingolstadt_1 ingolstadt_2).",
    )
    ap.add_argument(
        "--only-label-contains",
        type=str,
        default=None,
        help="Optional filter for choosing the routes CSV inside a run (matches label part after _routes_).",
    )
    args = ap.parse_args()

    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]  # path-clustering/
    results_dir = repo_root / "results"
    viz_root = results_dir / "visualizations"
    data_dir = repo_root / "data"

    viz_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    allowed_names = set(args.name) if args.name else None
    allowed_cities = set(args.cities) if args.cities else None

    run_dirs = [p for p in sorted(results_dir.iterdir()) if is_results_run_dir(p)]
    if allowed_names is not None:
        run_dirs = [p for p in run_dirs if p.name in allowed_names]

    if not run_dirs:
        print(f"No results run folders found in {results_dir}")
        return

    for run_dir in run_dirs:
        run_name = run_dir.name
        route_files = sorted(
            list(run_dir.glob("*_clusters_representants*.csv")) +
            list(run_dir.glob("*_routes*.csv"))
        )
        if not route_files:
            print(f"\nRun {run_name}: no '*_routes*.csv' or '*_clusters_representants*.csv' found, skipping")
            continue

        by_city: dict[str, list[Path]] = {}
        for f in route_files:
            city = city_from_routes_file(f)
            if allowed_cities and city not in allowed_cities:
                continue
            by_city.setdefault(city, []).append(f)

        if not by_city:
            print(f"\nRun {run_name}: no route files after city filtering, skipping")
            continue

        print(f"\n=== Run: {run_name} ===")

        for city, files in sorted(by_city.items()):
            od_json = data_dir / city / f"od_{city}.json"
            nod_file = data_dir / city / f"{city}.nod.xml"
            edg_file = data_dir / city / f"{city}.edg.xml"

            if not od_json.exists():
                print(f"City {city}: missing {od_json}, skipping")
                continue
            if not nod_file.exists() or not edg_file.exists():
                print(f"City {city}: missing {nod_file} or {edg_file}, skipping")
                continue

            routes_file = pick_routes_file(files, args.only_label_contains)
            if routes_file is None:
                print(f"City {city}: no routes CSV matched selection, skipping")
                continue

            label = label_from_routes_file(routes_file)

            try:
                df = pd.read_csv(
                    routes_file,
                    usecols=lambda c: c in {"origins", "destinations", "path", "cluster"},
                    dtype=str,
                    low_memory=False,
                )
            except Exception as e:
                print(f"City {city}: failed reading {routes_file.name}: {e}")
                continue

            if not {"origins", "destinations", "path"}.issubset(df.columns):
                print(f"City {city}: {routes_file.name} missing required columns (origins, destinations, path), skipping")
                continue

            od_pairs_all = read_od_pairs(od_json)
            if not od_pairs_all:
                print(f"City {city}: od json empty, skipping")
                continue

            present = set(zip(df["origins"].astype(str), df["destinations"].astype(str)))
            candidates = [od for od in od_pairs_all if (str(od[0]), str(od[1])) in present]
            if not candidates:
                print(f"City {city}: none of ODs appear in {routes_file.name}, skipping")
                continue

            k = min(args.num_ods, len(candidates))
            sampled_ods = rng.sample(candidates, k=k)

            city_out = viz_root / run_name
            city_out.mkdir(parents=True, exist_ok=True)

            print(f"\nCity: {city}")
            print(f"  Using: {routes_file.name} (label={label})")
            print(f"  Sampled ODs: {len(sampled_ods)} (seed={args.seed})")

            images: list[tuple[int, Path]] = []
            for od_idx, (origin, destination) in enumerate(sampled_ods):
                od_df = df[
                    (df["origins"].astype(str) == str(origin)) &
                    (df["destinations"].astype(str) == str(destination))
                ]
                if od_df.empty:
                    continue

                paths_with_cluster: list[tuple[int, list[str]]] = []
                paths_plain: list[list[str]] = []
                path_labels: list[str] = []

                has_cluster = "cluster" in od_df.columns

                for row in od_df.itertuples(index=False):
                    parsed = parse_path_cell(getattr(row, "path", None))
                    if not parsed:
                        continue

                    if has_cluster:
                        try:
                            cl = int(getattr(row, "cluster"))
                        except Exception:
                            cl = 10**9
                        paths_with_cluster.append((cl, parsed))
                    else:
                        paths_plain.append(parsed)

                if has_cluster:
                    ordered = sorted(paths_with_cluster, key=lambda t: t[0])
                    paths = [p for _, p in ordered]
                    path_labels = [f"cluster {cl}" for cl, _ in ordered]
                else:
                    paths = paths_plain
                    path_labels = [f"Path {i}" for i in range(len(paths))]

                if not paths:
                    continue

                out_png = city_out / f"{city}_{run_name}_od{od_idx}.png" # no extra intermediate /city dir after run dir
                title = f"{run_name} | {city} | {label} | OD{od_idx}: {safe_od_tag(origin, destination)} | n={len(paths)}"

                try:
                    show_multi_routes_with_labels(
                        nod_file_path=str(nod_file),
                        edg_file_path=str(edg_file),
                        paths=paths,  # ALL routes
                        origin=str(origin),
                        destination=str(destination),
                        path_labels=path_labels,
                        autocrop=True,
                        title=title,
                        save_file_path=str(out_png),
                        show=False,
                    )
                    images.append((od_idx, out_png))
                    print(f"  Saved {out_png.relative_to(results_dir)}")
                except Exception as e:
                    print(f"  Visualize failed: OD{od_idx}: {e}")

            if not images:
                print(f"  No images rendered for {city} in run {run_name}; skipping grid.")
                continue


if __name__ == "__main__":
    main()
"""Quick cluster statistics for a results run.

Usage:
	python3 scripts/analyze_clusters.py <run_name>

The script prefers ``*_clusters_representants.csv`` because those files already
carry the path, cluster label, and route features needed for a quick cluster
check. If that file is missing, it falls back to any CSV in the run folder that
contains the required columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FEATURES = [
	"total_length",
	"mean_speed",
	"speed_std",
	"pct_high_speed",
	"traffic_lights_per_km",
]

REQUIRED_COLUMNS = {"origins", "destinations", "path", "cluster", "free_flow_time"}


def parse_path_cell(value) -> list[str]:
	text = "" if value is None or pd.isna(value) else str(value)
	text = text.strip()
	for char in ["[", "]", "(", ")", "{", "}", "'"]:
		text = text.replace(char, "")
	text = text.replace('"', "")
	return [part.strip() for part in text.split(",") if part.strip()]


def path_to_edges(path_nodes: list[str]) -> set[tuple[str, str]]:
	if len(path_nodes) < 2:
		return set()
	return set(zip(path_nodes, path_nodes[1:]))


def jaccard_similarity(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> float:
	union = left | right
	if not union:
		return 1.0
	return len(left & right) / len(union)


def has_required_columns(csv_path: Path) -> bool:
	try:
		columns = pd.read_csv(csv_path, nrows=0).columns
	except Exception:
		return False
	return REQUIRED_COLUMNS.issubset(columns)


def find_cluster_file(run_dir: Path) -> Path:
	preferred_patterns = [
		"*_clusters_representants*.csv",
		"*_routes_enriched_clustered*.csv",
		"*_routes_clustered*.csv",
	]

	for pattern in preferred_patterns:
		for csv_path in sorted(run_dir.glob(pattern)):
			if has_required_columns(csv_path):
				return csv_path

	for csv_path in sorted(run_dir.glob("*.csv")):
		if has_required_columns(csv_path):
			return csv_path

	raise FileNotFoundError(
		f"No CSV with the required columns found in {run_dir}. "
		f"Needed columns: {sorted(REQUIRED_COLUMNS)}"
	)


def flatten_columns(columns: pd.Index) -> list[str]:
	flat: list[str] = []
	for column in columns:
		if isinstance(column, tuple):
			flat.append("_".join(str(part) for part in column if part))
		else:
			flat.append(str(column))
	return flat


def load_cluster_frame(csv_path: Path) -> pd.DataFrame:
	frame = pd.read_csv(csv_path, low_memory=False)
	missing = REQUIRED_COLUMNS - set(frame.columns)
	if missing:
		raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

	frame = frame.copy()
	frame["cluster"] = pd.to_numeric(frame["cluster"], errors="coerce")
	frame = frame.dropna(subset=["cluster"])
	frame["cluster"] = frame["cluster"].astype(int)
	frame["od"] = frame["origins"].astype(str) + " -> " + frame["destinations"].astype(str)
	return frame


def normalize_features(frame: pd.DataFrame, requested: list[str]) -> list[str]:
	available = [feature for feature in requested if feature in frame.columns]
	missing = [feature for feature in requested if feature not in frame.columns]
	if missing:
		print(f"Warning: missing feature columns will be skipped: {', '.join(missing)}")
	if not available:
		raise ValueError("None of the requested feature columns exist in the CSV.")
	return available


def build_cluster_summary(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
	grouped = frame.groupby("cluster", sort=True)

	summary = grouped.size().to_frame("n_routes")
	summary["n_ods"] = grouped["od"].nunique()

	fft_stats = grouped["free_flow_time"].agg(["min", "max", "mean", "std", "median"])
	fft_stats.columns = [f"free_flow_time_{name}" for name in fft_stats.columns]
	summary = summary.join(fft_stats)

	feature_stats = grouped[features].agg(["min", "max", "mean", "std", "median"])
	feature_stats.columns = flatten_columns(feature_stats.columns)
	summary = summary.join(feature_stats)

	return summary.reset_index()


def build_overlap_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
	records: list[dict[str, object]] = []

	for (origin, destination), od_frame in frame.groupby(["origins", "destinations"], sort=False):
		paths: list[tuple[int, set[tuple[str, str]]]] = []

		for row in od_frame.itertuples(index=False):
			path_nodes = parse_path_cell(getattr(row, "path", None))
			edge_set = path_to_edges(path_nodes)
			if not edge_set:
				continue

			paths.append((int(getattr(row, "cluster")), edge_set))

		for index, (cluster_a, edges_a) in enumerate(paths):
			for cluster_b, edges_b in paths[index + 1 :]:
				if cluster_a == cluster_b:
					continue

				left, right = sorted((cluster_a, cluster_b))
				records.append(
					{
						"cluster_a": left,
						"cluster_b": right,
						"od": f"{origin} -> {destination}",
						"overlap": jaccard_similarity(edges_a, edges_b),
					}
				)

	if not records:
		empty_summary = pd.DataFrame(columns=["cluster_a", "cluster_b", "n_route_pairs", "n_ods", "mean_overlap"])
		empty_matrix = pd.DataFrame()
		return empty_summary, empty_matrix, float("nan")

	pair_frame = pd.DataFrame(records)
	summary = (
		pair_frame.groupby(["cluster_a", "cluster_b"], sort=True)
		.agg(
			n_route_pairs=("overlap", "size"),
			n_ods=("od", "nunique"),
			mean_overlap=("overlap", "mean"),
			std_overlap=("overlap", "std"),
			min_overlap=("overlap", "min"),
			max_overlap=("overlap", "max"),
		)
		.reset_index()
	)

	clusters = sorted(frame["cluster"].unique().tolist())
	matrix = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
	for row in summary.itertuples(index=False):
		matrix.loc[row.cluster_a, row.cluster_b] = row.mean_overlap
		matrix.loc[row.cluster_b, row.cluster_a] = row.mean_overlap

	overall_mean = float(pair_frame["overlap"].mean())
	return summary, matrix, overall_mean


def build_od_summary(frame: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []

	for (origin, destination), od_frame in frame.groupby(["origins", "destinations"], sort=False):
		path_rows: list[tuple[int, float, float, set[tuple[str, str]]]] = []

		for row in od_frame.itertuples(index=False):
			path_nodes = parse_path_cell(getattr(row, "path", None))
			edge_set = path_to_edges(path_nodes)
			path_rows.append(
				(
					int(getattr(row, "cluster")),
					float(getattr(row, "free_flow_time")),
					float(getattr(row, "total_length")),
					edge_set,
				)
			)

		if not path_rows:
			continue

		fft_values = [item[1] for item in path_rows]
		length_values = [item[2] for item in path_rows]
		clusters = sorted({item[0] for item in path_rows})
		max_overlap = 0.0

		for index, current in enumerate(path_rows):
			for other in path_rows[index + 1 :]:
				max_overlap = max(max_overlap, jaccard_similarity(current[3], other[3]))

		rows.append(
			{
				"origins": origin,
				"destinations": destination,
				"n_routes": len(path_rows),
				"n_clusters": len(clusters),
				"clusters": ",".join(str(cluster) for cluster in clusters),
				"n_zero_fft_routes": sum(1 for item in path_rows if item[1] == 0.0),
				"free_flow_time_min": min(fft_values),
				"free_flow_time_max": max(fft_values),
				"free_flow_time_spread": max(fft_values) - min(fft_values),
				"total_length_min": min(length_values),
				"total_length_max": max(length_values),
				"total_length_spread": max(length_values) - min(length_values),
				"max_pairwise_overlap": max_overlap,
			}
		)

	return pd.DataFrame(rows).sort_values(
		["free_flow_time_spread", "total_length_spread", "max_pairwise_overlap"],
		ascending=[False, False, False],
	)


def print_table(title: str, frame: pd.DataFrame) -> None:
	print(f"\n{title}")
	if frame.empty:
		print("  <no data>")
		return
	print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
	parser = argparse.ArgumentParser(description="Calculate quick per-cluster statistics for a results run.")
	parser.add_argument("run_name", help="Results subfolder name inside path-clustering/results/")
	parser.add_argument(
		"--features",
		nargs="*",
		default=DEFAULT_FEATURES,
		help="Five feature columns to summarize per cluster.",
	)
	parser.add_argument(
		"--results-dir",
		default=None,
		help="Optional path to the results directory. Defaults to ../results relative to this script.",
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Optional directory for CSV outputs. Defaults to ../analysis relative to this script.",
	)
	args = parser.parse_args()

	script_dir = Path(__file__).resolve().parent
	repo_root = script_dir.parent
	results_dir = Path(args.results_dir) if args.results_dir else repo_root / "results"
	output_dir = Path(args.output_dir) if args.output_dir else repo_root / "analysis"

	run_dir = results_dir / args.run_name
	if not run_dir.exists():
		raise FileNotFoundError(f"Run folder not found: {run_dir}")

	cluster_file = find_cluster_file(run_dir)
	frame = load_cluster_frame(cluster_file)
	features = normalize_features(frame, args.features)

	cluster_summary = build_cluster_summary(frame, features)
	overlap_summary, overlap_matrix, overall_overlap = build_overlap_summary(frame)
	od_summary = build_od_summary(frame)

	fft_cluster_means = frame.groupby("cluster")["free_flow_time"].mean()
	fft_spread = float(fft_cluster_means.max() - fft_cluster_means.min()) if not fft_cluster_means.empty else float("nan")

	output_dir.mkdir(parents=True, exist_ok=True)
	cluster_summary_path = output_dir / f"{args.run_name}_cluster_stats.csv"
	overlap_summary_path = output_dir / f"{args.run_name}_cluster_overlap.csv"
	overlap_matrix_path = output_dir / f"{args.run_name}_cluster_overlap_matrix.csv"
	od_summary_path = output_dir / f"{args.run_name}_od_spread.csv"

	cluster_summary.to_csv(cluster_summary_path, index=False)
	overlap_summary.to_csv(overlap_summary_path, index=False)
	overlap_matrix.to_csv(overlap_matrix_path)
	od_summary.to_csv(od_summary_path, index=False)

	print(f"Using {cluster_file.relative_to(repo_root)}")
	print(f"Rows: {len(frame)} | ODs: {frame['od'].nunique()} | Clusters: {frame['cluster'].nunique()}")
	print(f"Saved: {cluster_summary_path.relative_to(repo_root)}")
	print(f"Saved: {overlap_summary_path.relative_to(repo_root)}")
	print(f"Saved: {overlap_matrix_path.relative_to(repo_root)}")
	print(f"Saved: {od_summary_path.relative_to(repo_root)}")
	print(f"Free-flow-time spread across cluster means: {fft_spread:.4f}")
	print(f"Overall cross-cluster edge overlap: {overall_overlap:.4f}")

	print_table("Per-cluster summary", cluster_summary)
	print_table("Cross-cluster edge overlap", overlap_summary)
	print_table("Cross-cluster edge overlap matrix", overlap_matrix.reset_index().rename(columns={"index": "cluster"}))
	print_table("Per-OD spread summary", od_summary)


if __name__ == "__main__":
	main()

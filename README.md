# path-clustering

Route generation, feature enrichment, and clustering utilities for URB route sets.

Note: the alternative generator is not supported with JanuX 1.1.0.

## Individual Commands

These commands are available mostly for debugging or running individual pipeline steps manually.

Generate routes and route-level feature inputs:

```bash
path-clustering-generate \
  --all \
  --name ingolstadt_custom \
  --data-dir /path/to/URB/networks \
  --output-dir /path/to/output/run \
  --config generate-default
```

Cluster an already generated/enriched route file:

```bash
path-clustering-cluster \
  --run-dir /path/to/output/run \
  --config cluster-default
```

For normal usage, prefer the full pipeline command below.

## Full Pipeline

Use this command to generate routes, calculate route features, cluster them, and save the files needed by URB:

```bash
path-clustering-run \
  --network-folder /path/to/URB/networks/ingolstadt_custom \
  --route-set clustering-kmeans-4 \
  --config run-default
```

Arguments:

- `--network-folder`: URB network directory, for example `URB/networks/ingolstadt_custom`.
- `--route-set`: output subdirectory name under `<network-folder>/clustered_routes/`.
- `--config`: full pipeline config path, or a bundled package config name.

The command writes:

```text
<network-folder>/clustered_routes/<route-set>/
  <network>_clusters_representants.csv
  <network>_action_masks.csv
  <network>_clustering_config.json
```

If `<route-set>` already exists and contains files, a numeric suffix is appended, for example `<route-set>-2`.

## URB Wrapper

For normal URB usage, prefer the wrapper inside URB:

```bash
cd /path/to/URB
python scripts/generate_clustered_routes.py \
  --net ingolstadt_custom \
  --route-set clustering-kmeans-4 \
  --config example
```

This resolves paths using the URB folder layout:

- network: `networks/<net>`
- config: `config/clustering_config/<config>.json`
- output: `networks/<net>/clustered_routes/<route-set>`

Then URB training scripts can consume the route set by passing `ROUTE_SET=<route-set>`.

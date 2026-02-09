import matplotlib
matplotlib.use("Agg")

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import janux as jx
from PIL import Image, ImageDraw, ImageFont


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
    # Handles: "<city>_routes.csv" and "<city>_routes_<suffix>.csv"
    stem = p.stem
    if "_routes" in stem:
        return stem.split("_routes", 1)[0]
    return stem


def label_from_routes_file(p: Path) -> str:
    # label shown in comparison headers
    stem = p.stem
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
    return list(zip(origins, destinations))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-ods", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None, help="Set for reproducible OD sampling.")
    ap.add_argument("--max-paths", type=int, default=10, help="Max routes drawn per OD per file.")
    ap.add_argument("--cities", type=str, nargs="*", default=None, help="Optional city filter.")
    ap.add_argument("--only-label-contains", type=str, default=None,
                    help="Optional filter: only include route files whose label contains this substring (e.g. 'extended3').")
    args = ap.parse_args()

    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]  # path-clustering/
    results_dir = repo_root / "results"
    routes_dir = results_dir / "routes"
    viz_root = results_dir / "visualizations"
    data_dir = repo_root / "data"

    viz_root.mkdir(parents=True, exist_ok=True)

    route_files = sorted(routes_dir.glob("*.csv"))
    if not route_files:
        print(f"No route CSVs found in {routes_dir}")
        return

    if args.seed is not None:
        rng = random.Random(args.seed)
    else:
        rng = random.Random()

    by_city: dict[str, list[Path]] = {}
    for f in route_files:
        city = city_from_routes_file(f)
        if args.cities and city not in set(args.cities):
            continue
        by_city.setdefault(city, []).append(f)

    for city, files in sorted(by_city.items()):
        od_json = data_dir / city / f"od_{city}.json"
        nod_file = data_dir / city / f"{city}.nod.xml"
        edg_file = data_dir / city / f"{city}.edg.xml"

        if not od_json.exists():
            print(f"\nCity {city}: missing {od_json}, skipping")
            continue
        if not nod_file.exists() or not edg_file.exists():
            print(f"\nCity {city}: missing {nod_file} or {edg_file}, skipping")
            continue

        # Load ODs
        od_pairs_all = read_od_pairs(od_json)
        if not od_pairs_all:
            print(f"\nCity {city}: od json empty, skipping")
            continue

        # Load route files and keep only those that have the required columns
        dfs: dict[str, pd.DataFrame] = {}
        for f in files:
            label = label_from_routes_file(f)
            if args.only_label_contains and (args.only_label_contains not in label):
                continue
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"\nCity {city}: failed reading {f.name}: {e}")
                continue
            if not {"origins", "destinations", "path"}.issubset(df.columns):
                print(f"\nCity {city}: {f.name} missing required columns, skipping")
                continue
            dfs[label] = df

        if not dfs:
            print(f"\nCity {city}: no usable route files after filtering, skipping")
            continue

        # Only pick ODs that are actually present in at least one df
        present = set()
        for df in dfs.values():
            present |= set(zip(df["origins"].astype(str), df["destinations"].astype(str)))

        candidates = [od for od in od_pairs_all if (od[0], od[1]) in present]
        if not candidates:
            print(f"\nCity {city}: none of ODs appear in route CSVs, skipping")
            continue

        k = min(args.num_ods, len(candidates))
        sampled_ods = rng.sample(candidates, k=k)

        ################################
        if city == "saint_arnoult":
            sampled_ods = [
                ("-101609498#5", "282689983#1"),
                ("336863934", "100475365#1"),
                ("-282689981#0", "-352797377"),
            ]
        ################################

        # Output folders
        city_out = viz_root / city
        city_out.mkdir(parents=True, exist_ok=True)

        print(f"\nCity: {city}")
        print(f"  Route files: {len(dfs)}")
        print(f"  Sampled ODs: {len(sampled_ods)}")

        # Render per-cell images
        images = []  # (label, od_idx, origin, destination, path_to_png)
        for label, df in dfs.items():
            for od_idx, (origin, destination) in enumerate(sampled_ods):
                od_df = df[
                    (df["origins"].astype(str) == str(origin)) &
                    (df["destinations"].astype(str) == str(destination))
                ]
                if od_df.empty:
                    continue

                n = min(args.max_paths, len(od_df))
                od_df = od_df.sample(n=n, random_state=args.seed) if len(od_df) > n else od_df

                paths = []
                for row in od_df.itertuples(index=False):
                    parsed = parse_path_cell(getattr(row, "path", None))
                    if parsed:
                        paths.append(parsed)
                if not paths:
                    continue

                out_png = city_out / f"{city}_routes_{label}_od{od_idx}.png"
                title = f"{city} | {label} | OD{od_idx}: {safe_od_tag(origin, destination)} | n={len(paths)}"

                try:
                    jx.show_multi_routes(
                        nod_file_path=str(nod_file),
                        edg_file_path=str(edg_file),
                        paths=paths,
                        origin=str(origin),
                        destination=str(destination),
                        autocrop=True,
                        title=title,
                        save_file_path=str(out_png),
                        show=False,
                    )
                    images.append((label, od_idx, origin, destination, out_png))
                    print(f"  Saved {out_png.relative_to(results_dir)}")
                except Exception as e:
                    print(f"  Visualize failed: {label} OD{od_idx}: {e}")

        if not images:
            print(f"  No images rendered for {city}; skipping comparison grid.")
            continue

        # ---- Build comparison grid (columns = label × OD) ----
        labels = sorted({lab for lab, *_ in images})
        ods = list(range(len(sampled_ods)))

        # Load images to determine cell size
        loaded = []
        for _, _, _, _, p in images:
            try:
                loaded.append(Image.open(p))
            except Exception:
                pass
        if not loaded:
            print(f"  Could not load rendered images for {city}; skipping grid.")
            continue

        cell_w = max(im.width for im in loaded)
        cell_h = max(im.height for im in loaded)
        for im in loaded:
            im.close()

        pad = 8
        header_h = 70
        label_w = 20  # no row labels (single row)

        n_cols = len(labels) * len(ods)
        n_rows = 1

        total_w = label_w + n_cols * (cell_w + pad) + pad
        total_h = header_h + n_rows * (cell_h + pad) + pad

        composite = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(composite)
        try:
            font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
        except Exception:
            font_small = ImageFont.load_default()

        # Map for lookup
        grid = {(lab, oi): None for lab in labels for oi in ods}
        for lab, od_idx, origin, destination, p in images:
            grid[(lab, od_idx)] = (p, origin, destination)

        # Header
        col = 0
        for lab in labels:
            for od_idx in ods:
                x = label_w + pad + col * (cell_w + pad)
                y = 6
                origin, destination = sampled_ods[od_idx]
                header_text = f"{lab}\nOD{od_idx}: {safe_od_tag(origin, destination, max_len=30)}"
                draw.multiline_text(
                    (x + cell_w // 2, y),
                    header_text,
                    fill=(0, 0, 0),
                    anchor="mm",
                    font=font_small,
                    align="center",
                )
                col += 1

        # Cells
        y0 = header_h
        col = 0
        for lab in labels:
            for od_idx in ods:
                x0 = label_w + pad + col * (cell_w + pad)
                cell = grid.get((lab, od_idx))
                if cell and cell[0].exists():
                    im = Image.open(cell[0])
                    im.thumbnail((cell_w, cell_h), Image.LANCZOS)
                    composite.paste(im, (x0 + (cell_w - im.width) // 2, y0 + (cell_h - im.height) // 2))
                    im.close()
                else:
                    draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(200, 200, 200))
                col += 1

        out_grid = city_out / f"{city}_comparison_grid.png"
        composite.save(out_grid)
        print(f"  Saved grid: {out_grid.relative_to(results_dir)}")


if __name__ == "__main__":
    main()
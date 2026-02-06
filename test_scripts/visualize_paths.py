import matplotlib
matplotlib.use("Agg")

import pandas as pd
from pathlib import Path
import janux as jx
import random
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# Config / Paths
# -----------------------------
random.seed(42)

script_dir = Path(__file__).parent
repo_root = script_dir.parent

routes_dir = repo_root / "test_results"
viz_dir = repo_root / "test_results" / "visualizations"
data_dir = repo_root / "test_data"

viz_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def parse_path_cell(value) -> list[str]:
    """
    Try to parse a path cell robustly.
    Supports formats like:
      - "e1,e2,e3"
      - "('e1', 'e2', 'e3')"
      - "['e1', 'e2']"
    """
    s = "" if value is None else str(value)
    s = s.strip()

    # remove common wrappers/brackets/quotes
    for ch in ["[", "]", "(", ")", "{", "}", "'"]:
        s = s.replace(ch, "")
    s = s.replace('"', "")

    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    return parts


def safe_od_tag(origin: str, destination: str, max_len: int = 40) -> str:
    """
    Short OD tag for filenames/titles.
    """
    o = str(origin)
    d = str(destination)
    tag = f"{o}->{d}"
    if len(tag) <= max_len:
        return tag
    return f"{o[:max_len//2]}…->{d[-max_len//2:]}"


def split_gen_and_beta(gen_name: str) -> tuple[str, str]:
    """
    Parse a filename suffix into (generator_base_name, beta_string).
    Expected patterns:
      - "extended3_-0.1" -> ("extended3", "-0.1")
      - "extended_uturn" -> ("extended_uturn", "")
    """
    parts = gen_name.rsplit("_", 1)
    if len(parts) == 2:
        base, maybe_beta = parts[0], parts[1]
        try:
            float(maybe_beta)
            return base, maybe_beta
        except Exception:
            return gen_name, ""
    return gen_name, ""


# -----------------------------
# Collect route files
# -----------------------------
route_files = list(routes_dir.rglob("*_routes_*.csv"))
groups: dict[tuple[str, str], list[Path]] = {}
for f in route_files:
    rel = f.relative_to(routes_dir)
    parts = rel.parts
    # expected: test_results/<city>/<N>_paths/<files>
    if len(parts) < 2:
        continue
    city = parts[0]
    n_paths_folder = parts[1]
    groups.setdefault((city, n_paths_folder), []).append(f)

if not groups:
    print("No route files found in", routes_dir)
    raise SystemExit(0)


# -----------------------------
# Main loop
# -----------------------------
for (city, n_paths_folder), files in groups.items():
    print(f"Network: {city}")

    # Load all generator outputs for this network
    dfs: dict[str, pd.DataFrame] = {}
    for f in files:
        gen = f.stem.split("_routes_")[-1]
        try:
            dfs[gen] = pd.read_csv(f)
        except Exception as e:
            print(f"  Failed to read {f}: {e}")

    if not dfs:
        print(f"  No readable route files for {city}, skipping.")
        continue

    # Determine ODs to visualize (all ODs present; you said it's ~2 anyway)
    all_ods = set()
    for df in dfs.values():
        if "origins" in df.columns and "destinations" in df.columns:
            all_ods |= set(zip(df["origins"].astype(str), df["destinations"].astype(str)))

    if not all_ods:
        print(f"  No OD columns found for {city}, skipping.")
        continue

    # Stable order: sort by (origin, dest)
    ods = sorted(list(all_ods), key=lambda x: (x[0], x[1]))
    print(f"  Found {len(ods)} OD pairs")

    # Find network files
    nod_file = data_dir / city / f"{city}.nod.xml"
    if not nod_file.exists():
        nod_file = data_dir / city / f"{city}.con.xml"
    edg_file = data_dir / city / f"{city}.edg.xml"

    if not nod_file.exists() or not edg_file.exists():
        print(f"  Missing network files for {city}: {nod_file}, {edg_file}. Skipping.")
        continue

    city_dir = viz_dir / city / n_paths_folder
    city_dir.mkdir(parents=True, exist_ok=True)

    # Create PNGs for ALL ODs and ALL generators
    # Collect: (base_gen, beta, od_idx, image_path)
    images = []

    for gen_name, df in dfs.items():
        if "path" not in df.columns:
            print(f"  Generator {gen_name}: missing 'path' column; skipping generator.")
            continue
        if "origins" not in df.columns or "destinations" not in df.columns:
            print(f"  Generator {gen_name}: missing OD columns; skipping generator.")
            continue

        base_gen, beta = split_gen_and_beta(gen_name)

        for od_idx, (origin, destination) in enumerate(ods):
            od_routes = df[
                (df["origins"].astype(str) == origin) &
                (df["destinations"].astype(str) == destination)
            ]
            if od_routes.empty:
                continue

            n_paths = min(100, len(od_routes))
            sample = od_routes.sample(n=n_paths, random_state=42) if len(od_routes) > n_paths else od_routes

            routes_to_visualize = []
            for row in sample.itertuples(index=False):
                # row.path exists because we checked columns
                parsed = parse_path_cell(getattr(row, "path", None))
                if parsed:
                    routes_to_visualize.append(parsed)

            if not routes_to_visualize:
                continue

            od_tag = safe_od_tag(origin, destination)
            save_path = city_dir / f"{city}_routes_{gen_name}_od{od_idx}.png"
            title = f"{city} - {gen_name} - OD{od_idx}: {od_tag} ({len(routes_to_visualize)} paths)"

            try:
                jx.show_multi_routes(
                    nod_file_path=str(nod_file),
                    edg_file_path=str(edg_file),
                    paths=routes_to_visualize,
                    origin=origin,
                    destination=destination,
                    autocrop=True,
                    title=title,
                    save_file_path=str(save_path),
                    show=False,
                )
                images.append((base_gen, beta, od_idx, origin, destination, save_path))
                print(f"  Saved {gen_name} OD{od_idx} image: {save_path.name}")
            except AssertionError as e:
                print(f"  Visualization failed for {gen_name} OD{od_idx}: {e}")
            except Exception as e:
                print(f"  Visualizer error for {gen_name} OD{od_idx}: {e}")

    if not images:
        print(f"  No images created for {city}; skipping composite.")
        continue

    # -----------------------------
    # Build comparison grid:
    # columns = gen1_OD0, gen1_OD1, gen2_OD0, gen2_OD1, ...
    # rows = beta values (if present), else single row "beta="
    # -----------------------------
    base_gens = sorted({g for g, _, _, _, _, _ in images})
    preferred_order = ["extended", "extended_uturn", "extended2", "extended2_lookahead", "extended3"]
    present_preferred = [g for g in preferred_order if g in set(base_gens)]
    other_gens = sorted(list(set(base_gens) - set(present_preferred)))
    gens = present_preferred + other_gens

    betas = sorted({b for _, b, _, _, _, _ in images}, key=lambda x: float(x) if x else 0.0)
    if not betas:
        betas = [""]

    # grid[beta][(gen, od_idx)] = image_path
    grid = {b: {(g, od_idx): None for g in gens for od_idx in range(len(ods))} for b in betas}
    od_lookup = {idx: ods[idx] for idx in range(len(ods))}

    for g, b, od_idx, origin, destination, p in images:
        if b not in grid:
            grid[b] = {(gg, oi): None for gg in gens for oi in range(len(ods))}
        grid[b][(g, od_idx)] = p

    # Determine cell size
    loaded = []
    for _, _, _, _, _, p in images:
        try:
            loaded.append(Image.open(p))
        except Exception:
            pass

    if not loaded:
        print(f"  Could not load any images for {city}; skipping composite.")
        continue

    cell_w = max(im.width for im in loaded)
    cell_h = max(im.height for im in loaded)
    for im in loaded:
        im.close()

    # Layout
    pad = 8
    header_h = 60
    label_w = 160

    n_cols = len(gens) * len(ods)
    n_rows = len(betas)

    total_w = label_w + n_cols * (cell_w + pad) + pad
    total_h = header_h + n_rows * (cell_h + pad) + pad

    composite = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # Column headers: gen + OD
    col = 0
    for gen in gens:
        for od_idx in range(len(ods)):
            origin, destination = od_lookup[od_idx]
            od_tag = safe_od_tag(origin, destination, max_len=30)

            x = label_w + pad + col * (cell_w + pad)
            y = 6

            header_text = f"{gen}\nOD{od_idx}: {od_tag}"
            draw.multiline_text(
                (x + cell_w // 2, y),
                header_text,
                fill=(0, 0, 0),
                anchor="mm",
                font=font_small,
                align="center",
            )
            col += 1

    # Fill grid cells
    for row_idx, beta in enumerate(betas):
        y = header_h + row_idx * (cell_h + pad)

        # row label at left
        label_x = 10
        label_y = y + cell_h // 2
        beta_label = f"beta={beta}" if beta else "beta="
        draw.text((label_x, label_y), beta_label, fill=(0, 0, 0), anchor="lm", font=font)

        col = 0
        for gen in gens:
            for od_idx in range(len(ods)):
                x = label_w + pad + col * (cell_w + pad)
                p = grid.get(beta, {}).get((gen, od_idx))

                if p and p.exists():
                    im = Image.open(p)
                    im.thumbnail((cell_w, cell_h), Image.LANCZOS)
                    paste_x = x + (cell_w - im.width) // 2
                    paste_y = y + (cell_h - im.height) // 2
                    composite.paste(im, (paste_x, paste_y))
                    im.close()
                else:
                    draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(200, 200, 200))
                col += 1

    out_path = city_dir / f"{city}_comparison_grid_all_ods.png"
    composite.save(out_path)
    print(f"  Saved grid comparison image: {out_path.name}")
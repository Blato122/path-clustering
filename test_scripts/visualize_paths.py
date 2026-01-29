import matplotlib
matplotlib.use('Agg')

import pandas as pd
from pathlib import Path
import janux as jx
import random
from PIL import Image, ImageDraw, ImageFont

# Paths
script_dir = Path(__file__).parent
repo_root = script_dir.parent
routes_dir = repo_root / "test_results"
viz_dir = repo_root / "test_results" / "visualizations"
data_dir = repo_root / "test_data"

viz_dir.mkdir(parents=True, exist_ok=True)

# Collect route files recursively and group by city name
route_files = list(routes_dir.rglob("*_routes_*.csv"))
groups = {}
for f in route_files:
    if f.parent != routes_dir:
        city = f.parent.name
    else:
        stem = f.stem
        city = stem.split("_routes_")[0] if "_routes_" in stem else stem.split("_routes")[0]
    groups.setdefault(city, []).append(f)

if not groups:
    print("No route files found in", routes_dir)
    raise SystemExit(0)

for city, files in groups.items():
    print(f"Network: {city}")
    dfs = {}
    for f in files:
        # generator label is everything after the first "_routes_"
        gen = f.stem.split("_routes_")[-1]
        try:
            dfs[gen] = pd.read_csv(f)
        except Exception as e:
            print(f"  Failed to read {f}: {e}")

    if not dfs:
        print(f"  No readable route files for {city}, skipping.")
        continue

    # find OD pairs present in all generators (intersection), else use one generator's ODs
    od_sets = []
    for df in dfs.values():
        if 'origins' in df.columns and 'destinations' in df.columns:
            od_sets.append(set(zip(df['origins'].astype(str), df['destinations'].astype(str))))
    common_ods = set.intersection(*od_sets) if od_sets else set()

    if common_ods:
        selected_od = random.choice(list(common_ods))
    else:
        # fall back: pick an OD from the first generator that has at least one route
        first_df = next(iter(dfs.values()))
        if first_df.empty:
            print(f"  No routes in {city}, skipping.")
            continue
        selected_row = first_df.sample(n=1).iloc[0]
        selected_od = (str(selected_row['origins']), str(selected_row['destinations']))

    origin, destination = selected_od
    print(f"  Selected OD: {origin} -> {destination}")

    # find node/edge files: prefer .nod.xml, fallback to .con.xml for nod
    nod_file = data_dir / city / f"{city}.nod.xml"
    if not nod_file.exists():
        nod_file = data_dir / city / f"{city}.con.xml"
    edg_file = data_dir / city / f"{city}.edg.xml"

    city_dir = viz_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)

    if not nod_file.exists() or not edg_file.exists():
        print(f"  Missing network files for {city}: {nod_file}, {edg_file}. Skipping visualization.")
        continue

    temp_images = []
    for gen_name, df in dfs.items():
        od_routes = df[(df['origins'].astype(str) == origin) & (df['destinations'].astype(str) == destination)]
        if od_routes.empty:
            print(f"  Generator {gen_name}: no routes for selected OD; skipping this generator.")
            continue

        # 10
        n_paths = min(10, len(od_routes))
        sample = od_routes.sample(n=n_paths) if len(od_routes) > n_paths else od_routes

        routes_to_visualize = []
        for row in sample.itertuples():
            try:
                routes_to_visualize.append(str(row.path).split(','))
            except Exception:
                # try alternative column names or skip
                if 'path' in df.columns:
                    routes_to_visualize.append(str(row.path).split(','))
                else:
                    print(f"  Row missing 'path' column for {gen_name}; skipping row.")
        if not routes_to_visualize:
            print(f"  No valid paths for {gen_name}; skipping.")
            continue

        save_path = city_dir / f"{city}_routes_{gen_name}.png"
        title = f"{city} - {gen_name}  ({len(routes_to_visualize)} paths)"
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
                show=False
            )
            temp_images.append((gen_name, save_path))
            print(f"  Saved {gen_name} image: {save_path.name}")
        except AssertionError as e:
            print(f"  Visualization failed for {gen_name}: {e}")
        except Exception as e:
            print(f"  Visualizer error for {gen_name}: {e}")

    if not temp_images:
        print(f"  No images created for {city}; skipping composite.")
        continue
    
    # temp_images: list of (gen_name, save_path) as before
    # parse gen_name into (gen, beta) expecting format "{name}_{beta}" or "{name}" if no beta
    parsed = []
    for gen_name, p in temp_images:
        parts = gen_name.rsplit('_', 1)
        if len(parts) == 2:
            gen, beta = parts[0], parts[1]
        else:
            gen, beta = gen_name, ""
        parsed.append((gen, beta, p))
    
    gens = sorted({g for g, b, p in parsed})
    betas = sorted({b for g, b, p in parsed}, key=lambda x: float(x) if x else 0.0)
    if not betas:
        betas = [""]
    
    # build mapping beta -> gen -> path
    grid = {b: {g: None for g in gens} for b in betas}
    for g, b, p in parsed:
        grid[b][g] = p
    
    # determine cell size from existing images (use median or max)
    loaded = [Image.open(p) for _, _, p in parsed]
    cell_w = max(im.width for im in loaded)
    cell_h = max(im.height for im in loaded)
    for im in loaded:
        im.close()
    
    # label/spacing sizes
    pad = 8
    header_h = 40
    label_w = 140
    
    total_w = label_w + len(gens) * (cell_w + pad) + pad
    total_h = header_h + len(betas) * (cell_h + pad) + pad
    
    composite = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Draw column headers
    for col_idx, gen in enumerate(gens):
        x = label_w + pad + col_idx * (cell_w + pad)
        y = 4
        draw.text((x + cell_w//2, y), gen, fill=(0,0,0), anchor="mm", font=font)
    
    # Fill grid cells
    for row_idx, beta in enumerate(betas):
        y = header_h + row_idx * (cell_h + pad)
        # row label at left
        label_x = 8
        label_y = y + cell_h//2
        draw.text((label_x, label_y), f"beta={beta}", fill=(0,0,0), anchor="lm", font=font)
    
        for col_idx, gen in enumerate(gens):
            x = label_w + pad + col_idx * (cell_w + pad)
            p = grid[beta].get(gen)
            if p and p.exists():
                im = Image.open(p)
                im.thumbnail((cell_w, cell_h), Image.LANCZOS)
                # center in cell
                paste_x = x + (cell_w - im.width)//2
                paste_y = y + (cell_h - im.height)//2
                composite.paste(im, (paste_x, paste_y))
                im.close()
            else:
                # draw placeholder rectangle
                draw.rectangle([x, y, x+cell_w, y+cell_h], outline=(200,200,200))
    
    out_path = city_dir / f"{city}_comparison_grid.png"
    composite.save(out_path)
    print(f"  Saved grid comparison image: {out_path.name}")
from pathlib import Path
import janux as jx
import json
import ast

def find_networks(base_dir: Path) -> dict[str, Path]:
    """
    Look for network folders under base_dir. 
    
    A network directory is valid if it has:
    - <name>.con.xml
    - <name>.edg.xml
    - <name>.rou.xml
    - od_<name>.json
    """
    out = {}
    if not base_dir.exists():
        return out
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        con = d / f"{name}.con.xml"
        edg = d / f"{name}.edg.xml"
        rou = d / f"{name}.rou.xml"
        od = d / f"od_{name}.txt" # TXT
        if con.exists() and edg.exists() and rou.exists() and od.exists():
            out[name] = d
    return out

def load_od_file(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    data = ast.literal_eval(text)
    origins = data["origins"]
    destinations = data["destinations"]
    return origins, destinations

def main():
    this_file = Path(__file__).resolve()
    # .../path-clustering/scripts/ods_to_json.py -> .../path-clustering
    repo_root = this_file.parents[1]
    data_dir = repo_root / "data"

    network_dict: dict[str, Path] = find_networks(data_dir)
    if not network_dict:
        print("No networks found. Check the directory structure.")
        return

    for name, dir in network_dict.items():
        print(f"Processing network: {name}")
        origins, destinations = load_od_file(f"{dir}/od_{name}.txt") # TXT

        od_dict = {
            "origins": origins,
            "destinations": destinations
        }
        with open(f"{dir}/od_{name}.json", 'w') as f: # JSON
            json.dump(od_dict, f)

        # ods = jx.utils.read_json(f"{dir}/od_{name}.json")
        # origins = ods["origins"]
        # destinations = ods["destinations"]
        # print(origins, destinations)

if __name__ == "__main__":
    main()

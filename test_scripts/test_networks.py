import networkx as nx
from collections import defaultdict
import janux as jx
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import combinations
from lxml import etree

"""
1)
===========================
Huge difference in extended_generator:
if pot_next < current_potential: # <= pot_cand
# if pot_next < pot_cand: # less strict, huge difference!

The second option allows for much more freedom - 50 paths
generated for a 3x3 grid (and possibly more but I set the cap to 50).
But way lower quality - loops started appearing.

The first option is stricter which is why it only allowed generating
3 paths for a 3x3 grid. While there were no loops, which is good, not
all options were exhausted, for example, the generated paths never
reached the top-left and bottom-right corners (same for a 5x5 grid). 
I think this is because the neighbor check only allow for one inoptimal
move before doing an optimal one. 

===========================
Notation: p(x) = potential of node x (shortest-path distance to the destination). 
Let current node = c, candidate = a, neighbor-of-candidate = b (the lookahead).

Stricter condition (A): 
Accept candidate 'a' via lookahead iff ∃b ∈ nbr(a): p(b) < p(c). 
Only accept 'a' when 'a' has a neighbor 'b' that is better than the current node 'c' - 
that guarantees there exists a direct-downhill neighbor relative to 'a' that is 
also downhill relative to 'c'. This limits allowed detours and prevents long lookahead chains.

Weaker condition (B): 
Accept candidate 'a' via lookahead iff ∃b ∈ nbr(a): p(b) < p(a).
Accept 'a' when 'a' has a neighbor 'b' downhill relative to 'a' (not necessarily relative to 'c'). 
B allows more detours because 'b' can still be higher than 'c', so moving c->a->... may not 
progress relative to 'c' and chains of lookahead-accepted nodes can occur.

If A holds then B also holds (because p(c) <= p(a) when a was not direct-downhill), 
so A ⇒ B. Therefore A is strictly stronger (more conservative) than B; 
B accepts strictly more candidates than A.

===========================
Doesn't matter whether the beta is set to -0.5, -0.1 or -0.000001 - this is not the limiting factor.
Same with the number of samples - in general, the more the better but because of the strict node
potential check it simply isn't possible to generate more unique paths.   
"""

this_file = Path(__file__).resolve()
repo_root = this_file.parents[1]
test_results_dir = repo_root / "test_results"
test_data_dir = repo_root / "test_data"

def jaccard_similarity(path_a, path_b):
    sa, sb = set(path_a), set(path_b)
    if not sa and not sb:
        return 1.0
    # size of intersection / size of union
    return len(sa & sb) / len (sa | sb)

def summarize(routes):
    print(f"Total selected routes: {len(routes.index)}")
    unique_routes = routes['path'].unique()
    print(f"Unique routes: {len(unique_routes)}")
    lengths = [len(p.split(",")) for p in routes['path'].unique()]
    print(f"Route lengths (min/mean/max): {min(lengths)}/{np.mean(lengths):.2f}/{max(lengths)}")
    
    route_edges = [r.split(",") for r in unique_routes]
    total_j = 0.0
    pairs = 0
    # Produces unordered pairs without replacement (no aa or bb; ab=ba)
    for a, b in combinations(route_edges, 2):
        total_j += jaccard_similarity(a, b)
        pairs += 1
    print(f"Mean pairwise Jaccard overlap: {total_j/pairs:.3f}")

if __name__ == "__main__":
    path_gen_kwargs = {
        "verbose": True,
        "number_of_paths": 10,
        "beta": -0.10,
        "weight": "time",
        "num_samples": 300,
        "max_path_length": None,
        "allow_loops": False,

        "adaptive": True,
        "tolerate_num_iterations": 20,
        "shift_parameters_by": 5,
        "params_to_shift": "both",

        "random_seed": 42,
        "max_resample_iterations": 50
    }

    ods_file = test_data_dir / "default_ods.json"
    if not ods_file.exists():
        print(f"Exiting: missing f{ods_file} file")
        exit(1)
    ods = jx.utils.read_json(ods_file)

    test_networks = ["grid", "csomor", "ingolstadt"] # simple, medium, large
    generators = ["extended", "extended_uturn", "extended2", "extended2_lookahead"]
    betas = (-np.logspace(np.log10(0.1), np.log10(3), num=5)).tolist() # negative

    for beta in betas:
        path_gen_kwargs['beta'] = beta
        for gen in generators:
            path_gen_kwargs['version'] = gen
            for name in test_networks:
                required_files = [
                    test_data_dir / name / f"{name}.con.xml",
                    test_data_dir / name / f"{name}.edg.xml",
                    test_data_dir / name / f"{name}.rou.xml",
                ]
                
                if not all(f.exists() for f in required_files):
                    print(f"Skipping {name}: missing required files for route generation")
                    continue

                con_file, edg_file, rou_file= required_files

                origins = ods[name]["origins"]
                destinations = ods[name]["destinations"]
                network = jx.build_digraph(str(con_file), str(edg_file), str(rou_file))

                all_routes = []
                for o_id, d_id in zip(origins, destinations):
                    try:
                        routes = jx.extended_generator(
                            network, 
                            [o_id], 
                            [d_id],
                            as_df=True,
                            calc_free_flow=True,
                            **path_gen_kwargs
                        )
                        all_routes.append(routes)
                    except AssertionError as e:
                        print(f"Skipping network {name}: {e}")
                        continue

                # Save the routes to a CSV file    
                if all_routes:
                    # summarize(all_routes)
                    all_routes_merged = pd.concat(all_routes)
                    
                    csv_save_dir = test_results_dir / name
                    csv_save_dir.mkdir(parents=True, exist_ok=True)

                    csv_save_path = csv_save_dir / f"{name}_routes_{gen}_{round(beta, 3)}.csv"
                    all_routes_merged.to_csv(csv_save_path, index=False)
                    print(f"Saved routes to: {csv_save_path}")
                else:
                    print(f"No routes generated for {name}")
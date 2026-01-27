import networkx as nx
from collections import defaultdict
import janux as jx
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import combinations
from lxml import etree

"""
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
test_results_dir = repo_root / "results" / "test"

def write_nod_xml(nodes, path):
    root = etree.Element("nodes")
    for nid, (x,y) in nodes.items():
        n = etree.SubElement(root, "node", id=nid, x=str(x*100), y=str(y*100))
    tree = etree.ElementTree(root)
    tree.write(path, pretty_print=True, xml_declaration=True, encoding="utf-8")

def write_edg_xml(segments, path):
    root = etree.Element("edges")
    for eid, (u, v) in segments.items():
        e = etree.SubElement(root, "edge", id=eid)
        e.set("from", u)
        e.set("to", v)
        lane = etree.SubElement(e, "lane", id=f"{eid}_0", speed="13.9", length="100.0")
    tree = etree.ElementTree(root)
    tree.write(path, pretty_print=True, xml_declaration=True, encoding="utf-8")

def build_nxn_graph(n=3) -> tuple[nx.DiGraph, list]:
    intersections = [(i, j) for i in range(n) for j in range(n)] # (0,0), ..., (n-1,n-1)
    intersection_id = lambda n: f"n{n[0]}{n[1]}"

    segments = {} # edge_id -> (start, end)
    nodes_coords = {}
    for i, j in intersections:
        start = intersection_id((i, j))
        nodes_coords[start] = (float(i), float(j))
        # Offset from an intersection to its 4 neighbors
        for ofs_i, ofs_j in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nbr_i, nbr_j = i + ofs_i, j + ofs_j
            if 0 <= nbr_i < n and 0 <= nbr_j < n:
                end = intersection_id((nbr_i, nbr_j))
                sid = f"e_{start}_to_{end}"
                # Add a connection between two intersections
                segments[sid] = (start, end)

    # Build DiGraph where nodes are segment ids and edges represent allowed edge->edge transitions
    G = nx.DiGraph()
    for sid in segments:
        G.add_node(sid)

    # Allow all transitions between segments (fully connected segment graph)
    for e1, (u1, v1) in segments.items():
        for e2, (u2, v2) in segments.items():
            if v1 == u2:
                G.add_edge(e1, e2, time=1.0)

    # Save SUMO-style xml files:
    test_results_dir.mkdir(parents=True, exist_ok=True)
    edg_path = test_results_dir / "simple.edg.xml"
    nod_path = test_results_dir / "simple.nod.xml"
    write_edg_xml(segments, edg_path)
    write_nod_xml(nodes_coords, nod_path)

    return G, list(segments.keys())

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
        "verbose" : True,
        "number_of_paths" : 50,
        "beta" : -0.10,
        "weight" : "time",
        "num_samples" : 300,
        "max_path_length" : 1000,
        "allow_loops" : False,

        "adaptive" : True,
        "tolerate_num_iterations" : 20,
        "shift_parameters_by" : 5,
        "params_to_shift" : "both",

        "random_seed": 42,
        "max_resample_iterations": 50
    }

    network, edges = build_nxn_graph(n=5)
    origin = edges[0]
    destination = edges[-1]
    all_routes = []

    routes = jx.extended_generator(
        network, 
        [origin], 
        [destination],
        as_df=True,
        calc_free_flow=True,
        **path_gen_kwargs
    )
    summarize(routes)

    # Save the routes to a CSV file    
    csv_save_path = test_results_dir / f"simple_routes.csv"
    routes.to_csv(csv_save_path, index=False)
    print(f"Saved routes to: {csv_save_path}")
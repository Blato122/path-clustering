"""
Test script to investigate why OD (82, 56) produces 0 routes with extended3.
Run from: path-clustering/scripts/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'JanuX')))

import ast
import janux as jx

# --- Network files ---
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'saint_arnoult')
con_file = os.path.join(data_dir, 'saint_arnoult.con.xml')
edg_file = os.path.join(data_dir, 'saint_arnoult.edg.xml')
rou_file = os.path.join(data_dir, 'saint_arnoult.rou.xml')
nod_file = os.path.join(data_dir, 'saint_arnoult.nod.xml')

# --- Load ODs ---
od_file = os.path.join(data_dir, 'od_saint_arnoult.txt')
with open(od_file, 'r') as f:
    ods = ast.literal_eval(f.read())

origins = ods["origins"]
destinations = ods["destinations"]

o_idx, d_idx = 82, 56
print(f"Origin {o_idx}: {origins[o_idx]}")
print(f"Destination {d_idx}: {destinations[d_idx]}")

# --- Build network ---
network = jx.build_digraph(con_file, edg_file, rou_file)

# --- Common kwargs ---
common_kwargs = dict(
    number_of_paths=10,
    beta=-0.5,
    weight="time",
    num_samples=100,        # small for quick debugging
    max_path_length=1000,
    allow_loops=False,
    adaptive=False,
    random_seed=42,
    max_resample_iterations=50,
    verbose=True,
)

# --- Test 1: extended3 (the failing version) ---
print("\n" + "="*60)
print("TEST 1: extended3 + diverse_selection=True")
print("="*60)
routes_v3 = jx.extended_generator(
    network,
    [origins[o_idx]],
    [destinations[d_idx]],
    as_df=False,
    calc_free_flow=False,
    version="extended3",
    diverse_selection=True,
    **common_kwargs,
)
print(f"Result: {len(routes_v3[(0, 0)])} routes")

# --- Test 2: extended3 with forbid_abs_reuse=False ---
print("\n" + "="*60)
print("TEST 2: extended3 + forbid_abs_reuse=False")
print("="*60)
routes_v3_no_reuse = jx.extended_generator(
    network,
    [origins[o_idx]],
    [destinations[d_idx]],
    as_df=False,
    calc_free_flow=False,
    version="extended3",
    diverse_selection=True,
    forbid_abs_reuse=False,
    **common_kwargs,
)
print(f"Result: {len(routes_v3_no_reuse[(0, 0)])} routes")

# --- Test 3: extended3 with forbid_junction_revisit=False ---
print("\n" + "="*60)
print("TEST 3: extended3 + forbid_junction_revisit=False")
print("="*60)
routes_v3_no_junc = jx.extended_generator(
    network,
    [origins[o_idx]],
    [destinations[d_idx]],
    as_df=False,
    calc_free_flow=False,
    version="extended3",
    diverse_selection=True,
    forbid_junction_revisit=False,
    **common_kwargs,
)
print(f"Result: {len(routes_v3_no_junc[(0, 0)])} routes")

# --- Test 4: extended3 with BOTH relaxed ---
print("\n" + "="*60)
print("TEST 4: extended3 + both relaxed")
print("="*60)
routes_v3_relaxed = jx.extended_generator(
    network,
    [origins[o_idx]],
    [destinations[d_idx]],
    as_df=False,
    calc_free_flow=False,
    version="extended3",
    diverse_selection=True,
    forbid_abs_reuse=False,
    forbid_junction_revisit=False,
    **common_kwargs,
)
print(f"Result: {len(routes_v3_relaxed[(0, 0)])} routes")

# --- Test 5: extended3 with BOTH relaxed and diverse_selection=False ---
print("\n" + "="*60)
print("TEST 5: extended3 + both relaxed + diverse_selection=False")
print("="*60)
routes_v3_relaxed_no_div = jx.extended_generator(
    network,
    [origins[o_idx]],
    [destinations[d_idx]],
    as_df=False,
    calc_free_flow=False,
    version="extended3",
    diverse_selection=True,
    forbid_abs_reuse=False,
    forbid_junction_revisit=False,
    **common_kwargs,
)
print(f"Result: {len(routes_v3_relaxed_no_div[(0, 0)])} routes")

# --- Test 6: original extended (should succeed) ---
print("\n" + "="*60)
print("TEST 6: extended (original)")
print("="*60)
routes_ext = jx.extended_generator(
    network,
    [origins[o_idx]],
    [destinations[d_idx]],
    as_df=False,
    calc_free_flow=False,
    version="extended",
    diverse_selection=False,
    **common_kwargs,
)
print(f"Result: {len(routes_ext[(0, 0)])} routes")

jx.show_multi_routes(
    nod_file_path=str(nod_file),
    edg_file_path=str(edg_file),
    paths=routes_v3_no_junc[(0, 0)],
    origin=str(origins[o_idx]),
    destination=str(destinations[d_idx]),
    autocrop=True,
    title="test",
    save_file_path="test.png",
    show=False,
)

# --- Summary ---
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"extended3 (default)                 : {len(routes_v3[(0, 0)])} routes")
print(f"extended3 (no abs_reuse)            : {len(routes_v3_no_reuse[(0, 0)])} routes")
print(f"extended3 (no junction_revisit)     : {len(routes_v3_no_junc[(0, 0)])} routes")
print(f"extended3 (both relaxed)            : {len(routes_v3_relaxed[(0, 0)])} routes")
print(f"extended3 (both relaxed no diverse) : {len(routes_v3_relaxed_no_div[(0, 0)])} routes")
print(f"extended  (original)                : {len(routes_ext[(0, 0)])} routes")
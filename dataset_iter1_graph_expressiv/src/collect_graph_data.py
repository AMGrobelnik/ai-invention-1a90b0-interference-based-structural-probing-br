"""
Graph Expressiveness Benchmark Data Collection for ISP Testing.

Collects and prepares graph expressiveness benchmark datasets:
1. BREC benchmark subset (from GitHub)
2. CSL graphs (from HuggingFace)
3. Strongly regular graph pairs (programmatic)
4. Molecular graph pairs (programmatic)
5. Classic 1-WL failure pairs (programmatic)

All stored as standardized JSON with edge lists, node degrees, and ground truth annotations.
"""

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
TEMP_DIR = WORKSPACE / "temp" / "datasets"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = TEMP_DIR / "data_out.json"
MINI_FILE = TEMP_DIR / "data_out_mini.json"
PREVIEW_FILE = TEMP_DIR / "data_out_preview.json"


# ── Utility Functions ──────────────────────────────────────────────────────────

def graph_to_dict(G: nx.Graph) -> dict:
    """Convert a networkx graph to our standardized dict format."""
    # Ensure integer node labels starting from 0
    G = nx.convert_node_labels_to_integers(G, first_label=0)

    edge_list = sorted([[int(u), int(v)] for u, v in G.edges()])
    degrees = [int(G.degree(n)) for n in sorted(G.nodes())]

    return {
        "num_nodes": int(G.number_of_nodes()),
        "edge_list": edge_list,
        "node_features": {"degree": degrees},
        "num_edges": int(G.number_of_edges()),
    }


def make_pair(
    pair_id: int,
    category: str,
    source: str,
    G_a: nx.Graph,
    G_b: nx.Graph,
    notes: str = "",
    wl_level: str = "1-WL",
    same_wl_color: bool = True,
    is_isomorphic: bool = False,
) -> dict:
    """Create a standardized pair dict."""
    return {
        "pair_id": pair_id,
        "category": category,
        "source": source,
        "same_wl_color": same_wl_color,
        "is_isomorphic": is_isomorphic,
        "wl_level": wl_level,
        "graph_a": graph_to_dict(G_a),
        "graph_b": graph_to_dict(G_b),
        "notes": notes,
    }


# ── STEP 1: BREC Benchmark Subset ─────────────────────────────────────────────

def collect_brec_pairs() -> list[dict]:
    """Download and parse BREC benchmark graphs, extract representative subset."""
    print("=" * 60)
    print("STEP 1: Collecting BREC Benchmark Subset")
    print("=" * 60)

    brec_dir = TEMP_DIR / "brec_tmp"
    brec_dir.mkdir(parents=True, exist_ok=True)

    # Try downloading BREC_data_all.zip from GitHub
    zip_path = brec_dir / "BREC_data_all.zip"
    npy_path = None

    # Method 1: Download zip directly
    try:
        print("Downloading BREC_data_all.zip from GitHub...")
        result = subprocess.run(
            [
                "wget", "-q", "--timeout=30",
                "https://github.com/GraphPKU/BREC/raw/Release/BREC_data_all.zip",
                "-O", str(zip_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and zip_path.exists() and zip_path.stat().st_size > 1000:
            print(f"Downloaded zip: {zip_path.stat().st_size} bytes")
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(brec_dir)
                print(f"Extracted files: {zf.namelist()}")

            # Look for brec_v3.npy
            for candidate in [
                brec_dir / "brec_v3.npy",
                brec_dir / "Data" / "raw" / "brec_v3.npy",
                brec_dir / "raw" / "brec_v3.npy",
            ]:
                if candidate.exists():
                    npy_path = candidate
                    break

            # Also check for individual category files
            if npy_path is None:
                # Search recursively
                for p in brec_dir.rglob("*.npy"):
                    print(f"  Found npy: {p}")
                    if "brec_v3" in p.name:
                        npy_path = p
                        break
        else:
            print(f"Download failed or empty file. Return code: {result.returncode}")
    except Exception as e:
        print(f"Download method 1 failed: {e}")

    # Method 2: Clone repo (shallow)
    if npy_path is None:
        try:
            print("Trying git clone (shallow)...")
            clone_dir = brec_dir / "BREC_repo"
            result = subprocess.run(
                [
                    "git", "clone", "--branch", "Release", "--depth", "1",
                    "https://github.com/GraphPKU/BREC.git",
                    str(clone_dir),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                # Find the zip in the cloned repo
                repo_zip = clone_dir / "BREC_data_all.zip"
                if repo_zip.exists():
                    with zipfile.ZipFile(repo_zip, 'r') as zf:
                        zf.extractall(brec_dir)
                    for p in brec_dir.rglob("brec_v3.npy"):
                        npy_path = p
                        break

                # Also look for individual category npy files
                if npy_path is None:
                    for p in clone_dir.rglob("*.npy"):
                        print(f"  Found npy in repo: {p}")
                        if "brec_v3" in p.name:
                            npy_path = p
                            break
        except Exception as e:
            print(f"Git clone method failed: {e}")

    # Method 3: Try individual category files from the repo
    category_files = {}
    if npy_path is None:
        print("Looking for individual category .npy files...")
        for p in brec_dir.rglob("*.npy"):
            name = p.stem.lower()
            print(f"  Found: {p.name}")
            category_files[name] = p

    # Parse BREC data
    pairs = []
    pair_id_offset = 0

    if npy_path is not None:
        print(f"\nLoading BREC from: {npy_path}")
        raw = np.load(str(npy_path), allow_pickle=True)
        print(f"Loaded {len(raw)} graph entries ({len(raw)//2} pairs)")

        # Parse all graphs from graph6
        graphs = []
        for i, g_bytes in enumerate(raw):
            try:
                if isinstance(g_bytes, str):
                    g_bytes = g_bytes.encode()
                elif isinstance(g_bytes, np.bytes_):
                    g_bytes = bytes(g_bytes)
                G = nx.from_graph6_bytes(g_bytes)
                graphs.append(G)
            except Exception as e:
                print(f"  Warning: Failed to parse graph {i}: {e}")
                graphs.append(None)

        print(f"Successfully parsed {sum(1 for g in graphs if g is not None)}/{len(graphs)} graphs")

        # Category index ranges (pairs):
        # Basic: pairs 0–59 (indices 0–119)
        # Regular: pairs 60–159 (indices 120–319)
        # Extension: pairs 160–259 (indices 320–519)
        # CFI: pairs 260–359 (indices 520–719)
        # 4-Vertex Condition: pairs 360–379 (indices 720–759)
        # Distance Regular: pairs 380–399 (indices 760–799)

        category_ranges = {
            "brec_basic": (0, 10),        # pairs 0-9
            "brec_regular": (60, 75),     # pairs 60-74
            "brec_extension": (160, 170), # pairs 160-169
            "brec_cfi": (260, 270),       # pairs 260-269
            "brec_distance_regular": (380, 385),  # pairs 380-384
        }

        for cat_name, (start_pair, end_pair) in category_ranges.items():
            cat_count = 0
            for pair_idx in range(start_pair, end_pair):
                idx_a = 2 * pair_idx
                idx_b = 2 * pair_idx + 1

                if idx_a >= len(graphs) or idx_b >= len(graphs):
                    print(f"  Warning: Pair {pair_idx} out of range (total={len(graphs)})")
                    break

                G_a = graphs[idx_a]
                G_b = graphs[idx_b]

                if G_a is None or G_b is None:
                    print(f"  Warning: Skipping pair {pair_idx} (parse failure)")
                    continue

                # Verify non-isomorphism
                is_iso = nx.is_isomorphic(G_a, G_b)

                pair = make_pair(
                    pair_id=pair_id_offset + len(pairs),
                    category=cat_name,
                    source="BREC",
                    G_a=G_a,
                    G_b=G_b,
                    notes=f"BREC pair index {pair_idx}, {cat_name}",
                    wl_level="1-WL" if cat_name != "brec_cfi" else "3-WL",
                    same_wl_color=True,
                    is_isomorphic=is_iso,
                )
                pairs.append(pair)
                cat_count += 1

            print(f"  {cat_name}: extracted {cat_count} pairs")

    elif category_files:
        # Fallback: use individual category files
        print("\nUsing individual category files...")
        cat_file_map = {
            "brec_basic": "basic",
            "brec_regular": "regular",
            "brec_extension": "extension",
            "brec_cfi": "cfi",
            "brec_distance_regular": "dr",
        }
        cat_limits = {
            "brec_basic": 10,
            "brec_regular": 15,
            "brec_extension": 10,
            "brec_cfi": 10,
            "brec_distance_regular": 5,
        }

        for cat_name, file_key in cat_file_map.items():
            if file_key in category_files:
                raw = np.load(str(category_files[file_key]), allow_pickle=True)
                limit = cat_limits[cat_name]

                for pair_idx in range(min(limit, len(raw) // 2)):
                    try:
                        g_a_bytes = raw[2 * pair_idx]
                        g_b_bytes = raw[2 * pair_idx + 1]
                        if isinstance(g_a_bytes, str):
                            g_a_bytes = g_a_bytes.encode()
                        if isinstance(g_b_bytes, str):
                            g_b_bytes = g_b_bytes.encode()

                        G_a = nx.from_graph6_bytes(bytes(g_a_bytes))
                        G_b = nx.from_graph6_bytes(bytes(g_b_bytes))

                        pair = make_pair(
                            pair_id=pair_id_offset + len(pairs),
                            category=cat_name,
                            source="BREC",
                            G_a=G_a,
                            G_b=G_b,
                            notes=f"BREC {cat_name} pair {pair_idx}",
                            wl_level="1-WL" if cat_name != "brec_cfi" else "3-WL",
                            same_wl_color=True,
                            is_isomorphic=nx.is_isomorphic(G_a, G_b),
                        )
                        pairs.append(pair)
                    except Exception as e:
                        print(f"  Warning: Failed pair {pair_idx} in {cat_name}: {e}")

                print(f"  {cat_name}: extracted pairs from {file_key}.npy")

    else:
        print("\nWARNING: Could not download BREC. Generating fallback strongly regular pairs.")
        # Generate additional SR pairs as fallback
        pairs.extend(generate_fallback_brec_pairs(pair_id_offset))

    print(f"\nTotal BREC pairs collected: {len(pairs)}")
    return pairs


def generate_fallback_brec_pairs(pair_id_offset: int) -> list[dict]:
    """Generate strongly regular graph pairs as fallback if BREC download fails."""
    pairs = []

    # Generate several pairs of regular graphs with same parameters
    # Petersen graph (srg(10, 3, 0, 1)) variants
    G_petersen = nx.petersen_graph()

    # Create another 3-regular graph on 10 nodes that is NOT isomorphic to Petersen
    # The Petersen graph complement is also srg(10, 6, 3, 4)
    G_petersen_comp = nx.complement(G_petersen)

    # Note: Petersen and its complement are NOT the same parameters, so this is
    # just for structural diversity, not a true 1-WL failure pair

    # Instead, generate cycle pairs
    for n in range(6, 16, 2):
        # C_n vs n/2 copies of C_2 (both 2-regular)
        # Actually C_n vs disjoint union of smaller cycles
        if n == 6:
            G_a = nx.cycle_graph(6)
            G_b = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
        elif n == 8:
            G_a = nx.cycle_graph(8)
            G_b = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))
        elif n == 10:
            G_a = nx.cycle_graph(10)
            G_b = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
        elif n == 12:
            G_a = nx.cycle_graph(12)
            G_b = nx.disjoint_union(nx.cycle_graph(4), nx.disjoint_union(
                nx.cycle_graph(4), nx.cycle_graph(4)
            ))
        elif n == 14:
            G_a = nx.cycle_graph(14)
            G_b = nx.disjoint_union(nx.cycle_graph(7), nx.cycle_graph(7))
        else:
            continue

        pair = make_pair(
            pair_id=pair_id_offset + len(pairs),
            category="brec_basic_fallback",
            source="programmatic (BREC fallback)",
            G_a=G_a,
            G_b=G_b,
            notes=f"Regular graph pair: C_{n} vs disjoint cycles, both 2-regular",
            wl_level="1-WL",
            same_wl_color=True,
            is_isomorphic=False,
        )
        pairs.append(pair)

    return pairs


# ── STEP 2: CSL (Circular Skip Link) Graphs ───────────────────────────────────

def csl_graph(n: int, skip: int) -> nx.Graph:
    """Generate a Circular Skip Link graph: cycle of n nodes with skip-link connections."""
    G = nx.cycle_graph(n)
    for i in range(n):
        G.add_edge(i, (i + skip) % n)
    return G


def collect_csl_pairs(pair_id_offset: int) -> list[dict]:
    """Generate CSL graph pairs for ISP testing."""
    print("\n" + "=" * 60)
    print("STEP 2: Collecting CSL (Circular Skip Link) Graphs")
    print("=" * 60)

    n = 41  # Standard CSL size
    skip_values = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]

    # Try loading from HuggingFace first
    hf_graphs = None
    try:
        print("Attempting to load CSL from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("graphs-datasets/CSL", split="train")

        # Group by label
        label_groups = {}
        for idx, example in enumerate(ds):
            label = tuple(example["y"]) if isinstance(example["y"], list) else (example["y"],)
            label_val = label[0]
            if label_val not in label_groups:
                label_groups[label_val] = []

            # Parse edge_index to networkx graph
            edge_index = example["edge_index"]
            num_nodes = example["num_nodes"]
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))

            # edge_index is [[src1, src2, ...], [dst1, dst2, ...]]
            if len(edge_index) == 2:
                for src, dst in zip(edge_index[0], edge_index[1]):
                    if src < dst:  # Avoid duplicate edges
                        G.add_edge(int(src), int(dst))

            label_groups[label_val].append((idx, G))

        print(f"  Loaded {len(ds)} CSL graphs, {len(label_groups)} isomorphism classes")
        hf_graphs = label_groups

    except Exception as e:
        print(f"  HuggingFace load failed: {e}")
        print("  Falling back to programmatic CSL generation")

    pairs = []

    if hf_graphs is not None and len(hf_graphs) >= 2:
        # Create pairs across different isomorphism classes
        labels = sorted(hf_graphs.keys())
        pair_count = 0

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if pair_count >= 20:
                    break

                label_a = labels[i]
                label_b = labels[j]

                # Pick one graph from each class
                _, G_a = hf_graphs[label_a][0]
                _, G_b = hf_graphs[label_b][0]

                pair = make_pair(
                    pair_id=pair_id_offset + len(pairs),
                    category="csl",
                    source="HuggingFace graphs-datasets/CSL",
                    G_a=G_a,
                    G_b=G_b,
                    notes=f"CSL pair: class {label_a} vs class {label_b}, "
                          f"both 41-node 4-regular, 1-WL indistinguishable",
                    wl_level="1-WL",
                    same_wl_color=True,
                    is_isomorphic=False,  # Different isomorphism classes
                )
                pairs.append(pair)
                pair_count += 1

            if pair_count >= 20:
                break

    else:
        # Programmatic fallback: generate CSL graphs
        print("  Generating CSL graphs programmatically...")
        csl_graphs = {}
        for skip in skip_values:
            G = csl_graph(n, skip)
            csl_graphs[skip] = G
            print(f"    CSL(41, {skip}): {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges, "
                  f"degrees={sorted(set(dict(G.degree()).values()))}")

        # Create pairs between graphs with different skip values
        pair_count = 0
        skip_list = list(csl_graphs.keys())
        for i in range(len(skip_list)):
            for j in range(i + 1, len(skip_list)):
                if pair_count >= 20:
                    break

                skip_a = skip_list[i]
                skip_b = skip_list[j]
                G_a = csl_graphs[skip_a]
                G_b = csl_graphs[skip_b]

                is_iso = nx.is_isomorphic(G_a, G_b)

                if not is_iso:
                    pair = make_pair(
                        pair_id=pair_id_offset + len(pairs),
                        category="csl",
                        source="programmatic (CSL)",
                        G_a=G_a,
                        G_b=G_b,
                        notes=f"CSL(41, {skip_a}) vs CSL(41, {skip_b}), "
                              f"4-regular, 1-WL indistinguishable",
                        wl_level="1-WL",
                        same_wl_color=True,
                        is_isomorphic=False,
                    )
                    pairs.append(pair)
                    pair_count += 1

            if pair_count >= 20:
                break

    print(f"  Total CSL pairs collected: {len(pairs)}")
    return pairs


# ── STEP 3: Strongly Regular Graph Pairs ───────────────────────────────────────

def collect_strongly_regular_pairs(pair_id_offset: int) -> list[dict]:
    """Generate classic strongly regular graph pairs."""
    print("\n" + "=" * 60)
    print("STEP 3: Collecting Strongly Regular Graph Pairs")
    print("=" * 60)

    pairs = []

    # ── Pair A: Rook 4×4 vs Shrikhande — srg(16, 6, 2, 2) ──
    print("\n  Generating Rook 4x4 graph...")
    G_rook = nx.Graph()
    for i in range(4):
        for j in range(4):
            G_rook.add_node((i, j))
    for i1 in range(4):
        for j1 in range(4):
            for i2 in range(4):
                for j2 in range(4):
                    if (i1 == i2 and j1 != j2) or (j1 == j2 and i1 != i2):
                        G_rook.add_edge((i1, j1), (i2, j2))
    G_rook = nx.convert_node_labels_to_integers(G_rook)

    print("  Generating Shrikhande graph...")
    G_shrikhande = nx.Graph()
    for i in range(4):
        for j in range(4):
            G_shrikhande.add_node((i, j))

    connection_set = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    for a in range(4):
        for b in range(4):
            for (da, db) in connection_set:
                c, d = (a + da) % 4, (b + db) % 4
                G_shrikhande.add_edge((a, b), (c, d))
    G_shrikhande = nx.convert_node_labels_to_integers(G_shrikhande)

    # Verify properties
    assert G_rook.number_of_nodes() == 16, f"Rook should have 16 nodes, got {G_rook.number_of_nodes()}"
    assert G_shrikhande.number_of_nodes() == 16, f"Shrikhande should have 16 nodes, got {G_shrikhande.number_of_nodes()}"
    assert G_rook.number_of_edges() == 48, f"Rook should have 48 edges, got {G_rook.number_of_edges()}"
    assert G_shrikhande.number_of_edges() == 48, f"Shrikhande should have 48 edges, got {G_shrikhande.number_of_edges()}"

    rook_degrees = sorted(set(dict(G_rook.degree()).values()))
    shrik_degrees = sorted(set(dict(G_shrikhande.degree()).values()))
    assert rook_degrees == [6], f"Rook should be 6-regular, got {rook_degrees}"
    assert shrik_degrees == [6], f"Shrikhande should be 6-regular, got {shrik_degrees}"

    is_iso = nx.is_isomorphic(G_rook, G_shrikhande)
    assert not is_iso, "Rook 4x4 and Shrikhande should NOT be isomorphic!"

    print(f"  Rook 4x4: {G_rook.number_of_nodes()} nodes, {G_rook.number_of_edges()} edges, 6-regular ✓")
    print(f"  Shrikhande: {G_shrikhande.number_of_nodes()} nodes, {G_shrikhande.number_of_edges()} edges, 6-regular ✓")
    print(f"  Non-isomorphic: ✓")

    pair_a = make_pair(
        pair_id=pair_id_offset,
        category="strongly_regular",
        source="programmatic",
        G_a=G_rook,
        G_b=G_shrikhande,
        notes="Rook 4x4 vs Shrikhande graph, srg(16, 6, 2, 2). "
              "Classic 1-WL (and 3-WL) indistinguishable pair. "
              "Requires higher-order WL or subgraph-based methods to distinguish.",
        wl_level="3-WL",
        same_wl_color=True,
        is_isomorphic=False,
    )
    pairs.append(pair_a)

    # ── Pair B: K3,3 vs Triangular Prism — both 3-regular on 6 nodes ──
    print("\n  Generating K3,3 (complete bipartite) graph...")
    G_k33 = nx.complete_bipartite_graph(3, 3)
    G_k33 = nx.convert_node_labels_to_integers(G_k33)

    print("  Generating Triangular Prism (= circular ladder C3) graph...")
    G_triprism = nx.circular_ladder_graph(3)
    G_triprism = nx.convert_node_labels_to_integers(G_triprism)

    # Verify
    assert G_k33.number_of_nodes() == 6, f"K3,3 should have 6 nodes, got {G_k33.number_of_nodes()}"
    assert G_triprism.number_of_nodes() == 6, f"Tri Prism should have 6 nodes, got {G_triprism.number_of_nodes()}"

    k33_degrees = sorted(set(dict(G_k33.degree()).values()))
    triprism_degrees = sorted(set(dict(G_triprism.degree()).values()))
    assert k33_degrees == [3], f"K3,3 should be 3-regular, got {k33_degrees}"
    assert triprism_degrees == [3], f"Tri Prism should be 3-regular, got {triprism_degrees}"

    is_iso_b = nx.is_isomorphic(G_k33, G_triprism)
    assert not is_iso_b, "K3,3 and Triangular Prism should NOT be isomorphic!"

    print(f"  K3,3: {G_k33.number_of_nodes()} nodes, {G_k33.number_of_edges()} edges, 3-regular ✓")
    print(f"  Tri Prism: {G_triprism.number_of_nodes()} nodes, {G_triprism.number_of_edges()} edges, 3-regular ✓")
    print(f"  Non-isomorphic: ✓ (K3,3 is bipartite with girth 4, Tri Prism has triangles)")

    pair_b = make_pair(
        pair_id=pair_id_offset + 1,
        category="strongly_regular",
        source="programmatic",
        G_a=G_k33,
        G_b=G_triprism,
        notes="K3,3 (complete bipartite) vs Triangular Prism, both 3-regular on 6 nodes. "
              "1-WL indistinguishable (same degree sequence). K3,3 is bipartite (girth 4, "
              "no triangles), Tri Prism has triangles (girth 3).",
        wl_level="1-WL",
        same_wl_color=True,
        is_isomorphic=False,
    )
    pairs.append(pair_b)

    # ── Pair C: Petersen graph vs Pentagonal Prism (Y5) — both 3-regular on 10 nodes ──
    print("\n  Generating Petersen graph...")
    G_petersen = nx.petersen_graph()

    print("  Generating Pentagonal Prism (Y5) graph...")
    G_prism5 = nx.circular_ladder_graph(5)

    assert G_petersen.number_of_nodes() == 10
    assert G_prism5.number_of_nodes() == 10
    pet_deg = sorted(set(dict(G_petersen.degree()).values()))
    prism_deg = sorted(set(dict(G_prism5.degree()).values()))
    assert pet_deg == [3], f"Petersen should be 3-regular, got {pet_deg}"
    assert prism_deg == [3], f"Prism should be 3-regular, got {prism_deg}"

    is_iso_c = nx.is_isomorphic(G_petersen, G_prism5)
    assert not is_iso_c, "Petersen and Prism Y5 should NOT be isomorphic!"

    print(f"  Petersen: {G_petersen.number_of_nodes()} nodes, {G_petersen.number_of_edges()} edges, 3-regular ✓")
    print(f"  Prism Y5: {G_prism5.number_of_nodes()} nodes, {G_prism5.number_of_edges()} edges, 3-regular ✓")
    print(f"  Non-isomorphic: ✓ (Petersen has girth 5, Prism has girth 4)")

    pair_c = make_pair(
        pair_id=pair_id_offset + 2,
        category="strongly_regular",
        source="programmatic",
        G_a=G_petersen,
        G_b=G_prism5,
        notes="Petersen graph (srg(10, 3, 0, 1), girth 5) vs Pentagonal Prism (Y5, girth 4), "
              "both 3-regular on 10 nodes. "
              "1-WL indistinguishable (same degree sequence: all degree 3).",
        wl_level="1-WL",
        same_wl_color=True,
        is_isomorphic=False,
    )
    pairs.append(pair_c)

    print(f"\n  Total strongly regular pairs: {len(pairs)}")
    return pairs


# ── STEP 4: Decalin vs Bicyclopentyl ──────────────────────────────────────────

def collect_molecular_pairs(pair_id_offset: int) -> list[dict]:
    """Generate classic molecular graph pairs (1-WL indistinguishable)."""
    print("\n" + "=" * 60)
    print("STEP 4: Collecting Molecular Graph Pairs")
    print("=" * 60)

    pairs = []

    # ── Decalin (bicyclo[4.4.0]decane) ──
    print("\n  Generating Decalin graph...")
    G_decalin = nx.Graph()
    G_decalin.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # first 6-ring
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 4),            # second 6-ring fused at 4-5
    ])

    # ── Bicyclopentyl (two 5-cycles connected by single bond) ──
    print("  Generating Bicyclopentyl graph...")
    G_bicyclopentyl = nx.Graph()
    G_bicyclopentyl.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # first 5-ring
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # second 5-ring
        (0, 5),                                     # connecting bond
    ])

    # Verify
    assert G_decalin.number_of_nodes() == 10, f"Decalin: expected 10 nodes, got {G_decalin.number_of_nodes()}"
    assert G_bicyclopentyl.number_of_nodes() == 10, f"Bicyclopentyl: expected 10 nodes, got {G_bicyclopentyl.number_of_nodes()}"
    assert G_decalin.number_of_edges() == 11, f"Decalin: expected 11 edges, got {G_decalin.number_of_edges()}"
    assert G_bicyclopentyl.number_of_edges() == 11, f"Bicyclopentyl: expected 11 edges, got {G_bicyclopentyl.number_of_edges()}"

    # Verify degree sequences match (both should have same multiset of degrees)
    dec_degs = sorted(dict(G_decalin.degree()).values())
    bic_degs = sorted(dict(G_bicyclopentyl.degree()).values())
    assert dec_degs == bic_degs, f"Degree sequences should match: {dec_degs} vs {bic_degs}"

    # Verify non-isomorphism
    is_iso = nx.is_isomorphic(G_decalin, G_bicyclopentyl)
    assert not is_iso, "Decalin and Bicyclopentyl should NOT be isomorphic!"

    print(f"  Decalin: {G_decalin.number_of_nodes()} nodes, {G_decalin.number_of_edges()} edges")
    print(f"  Degree sequence: {dec_degs}")
    print(f"  Bicyclopentyl: {G_bicyclopentyl.number_of_nodes()} nodes, {G_bicyclopentyl.number_of_edges()} edges")
    print(f"  Degree sequence: {bic_degs}")
    print(f"  Non-isomorphic: ✓")
    print(f"  Same degree sequence (1-WL indistinguishable): ✓")

    pair = make_pair(
        pair_id=pair_id_offset,
        category="molecular",
        source="programmatic",
        G_a=G_decalin,
        G_b=G_bicyclopentyl,
        notes="Decalin (two fused 6-cycles) vs Bicyclopentyl (two 5-cycles connected by bond). "
              "Classic 1-WL indistinguishable molecular pair. Same degree sequence "
              "(eight degree-2 nodes, two degree-3 nodes) but different cycle structures.",
        wl_level="1-WL",
        same_wl_color=True,
        is_isomorphic=False,
    )
    pairs.append(pair)

    print(f"\n  Total molecular pairs: {len(pairs)}")
    return pairs


# ── STEP 5: Classic 1-WL Failure Pairs ────────────────────────────────────────

def collect_classic_wl_pairs(pair_id_offset: int) -> list[dict]:
    """Generate classic 1-WL failure pairs."""
    print("\n" + "=" * 60)
    print("STEP 5: Collecting Classic 1-WL Failure Pairs")
    print("=" * 60)

    pairs = []

    # ── Pair 1: Hexagon (C₆) vs Two Triangles (2×C₃) ──
    print("\n  Generating C6 vs 2×C3 pair...")
    G_hex = nx.cycle_graph(6)
    G_2tri = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))

    # Verify
    assert G_hex.number_of_nodes() == 6
    assert G_2tri.number_of_nodes() == 6
    assert G_hex.number_of_edges() == 6
    assert G_2tri.number_of_edges() == 6

    hex_degs = sorted(dict(G_hex.degree()).values())
    tri_degs = sorted(dict(G_2tri.degree()).values())
    assert hex_degs == tri_degs, f"Degree sequences should match: {hex_degs} vs {tri_degs}"
    assert not nx.is_isomorphic(G_hex, G_2tri)

    print(f"  C6: {G_hex.number_of_nodes()} nodes, {G_hex.number_of_edges()} edges, 2-regular ✓")
    print(f"  2×C3: {G_2tri.number_of_nodes()} nodes, {G_2tri.number_of_edges()} edges, 2-regular ✓")
    print(f"  Non-isomorphic: ✓")

    pair1 = make_pair(
        pair_id=pair_id_offset,
        category="classic_wl_failures",
        source="programmatic",
        G_a=G_hex,
        G_b=G_2tri,
        notes="Hexagon (C6) vs two disjoint triangles (2×C3). "
              "Canonical 1-WL failure pair. Both 2-regular with 6 nodes and 6 edges.",
        wl_level="1-WL",
        same_wl_color=True,
        is_isomorphic=False,
    )
    pairs.append(pair1)

    # ── Pair 2: C8 vs 2×C4 ──
    print("\n  Generating C8 vs 2×C4 pair...")
    G_c8 = nx.cycle_graph(8)
    G_2c4 = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))

    assert G_c8.number_of_nodes() == 8
    assert G_2c4.number_of_nodes() == 8
    assert not nx.is_isomorphic(G_c8, G_2c4)

    print(f"  C8: {G_c8.number_of_nodes()} nodes, {G_c8.number_of_edges()} edges, 2-regular ✓")
    print(f"  2×C4: {G_2c4.number_of_nodes()} nodes, {G_2c4.number_of_edges()} edges, 2-regular ✓")
    print(f"  Non-isomorphic: ✓")

    pair2 = make_pair(
        pair_id=pair_id_offset + 1,
        category="classic_wl_failures",
        source="programmatic",
        G_a=G_c8,
        G_b=G_2c4,
        notes="C8 vs two disjoint C4 (squares). Both 2-regular with 8 nodes and 8 edges.",
        wl_level="1-WL",
        same_wl_color=True,
        is_isomorphic=False,
    )
    pairs.append(pair2)

    # ── Pair 3: C10 vs 2×C5 ──
    print("\n  Generating C10 vs 2×C5 pair...")
    G_c10 = nx.cycle_graph(10)
    G_2c5 = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))

    assert G_c10.number_of_nodes() == 10
    assert G_2c5.number_of_nodes() == 10
    assert not nx.is_isomorphic(G_c10, G_2c5)

    pair3 = make_pair(
        pair_id=pair_id_offset + 2,
        category="classic_wl_failures",
        source="programmatic",
        G_a=G_c10,
        G_b=G_2c5,
        notes="C10 vs two disjoint C5 (pentagons). Both 2-regular with 10 nodes and 10 edges.",
        wl_level="1-WL",
        same_wl_color=True,
        is_isomorphic=False,
    )
    pairs.append(pair3)

    # ── Pair 4: C12 vs 3×C4 ──
    print("\n  Generating C12 vs 3×C4 pair...")
    G_c12 = nx.cycle_graph(12)
    G_3c4 = nx.disjoint_union(nx.cycle_graph(4), nx.disjoint_union(
        nx.cycle_graph(4), nx.cycle_graph(4)
    ))

    assert G_c12.number_of_nodes() == 12
    assert G_3c4.number_of_nodes() == 12
    assert not nx.is_isomorphic(G_c12, G_3c4)

    pair4 = make_pair(
        pair_id=pair_id_offset + 3,
        category="classic_wl_failures",
        source="programmatic",
        G_a=G_c12,
        G_b=G_3c4,
        notes="C12 vs three disjoint C4 (squares). Both 2-regular with 12 nodes.",
        wl_level="1-WL",
        same_wl_color=True,
        is_isomorphic=False,
    )
    pairs.append(pair4)

    # ── Pair 5: C12 vs 2×C6 ──
    print("\n  Generating C12 vs 2×C6 pair...")
    G_2c6 = nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))

    assert G_2c6.number_of_nodes() == 12
    assert not nx.is_isomorphic(G_c12, G_2c6)

    pair5 = make_pair(
        pair_id=pair_id_offset + 4,
        category="classic_wl_failures",
        source="programmatic",
        G_a=G_c12,
        G_b=G_2c6,
        notes="C12 vs two disjoint C6 (hexagons). Both 2-regular with 12 nodes.",
        wl_level="1-WL",
        same_wl_color=True,
        is_isomorphic=False,
    )
    pairs.append(pair5)

    print(f"\n  Total classic WL failure pairs: {len(pairs)}")
    return pairs


# ── STEP 6: Assemble and Validate ─────────────────────────────────────────────

def validate_pair(pair: dict) -> list[str]:
    """Validate a single pair entry. Returns list of issues found."""
    issues = []

    for graph_key in ["graph_a", "graph_b"]:
        g = pair[graph_key]

        # Check num_nodes consistency
        actual_nodes = set()
        for u, v in g["edge_list"]:
            actual_nodes.add(u)
            actual_nodes.add(v)

        if len(actual_nodes) > 0 and max(actual_nodes) >= g["num_nodes"]:
            issues.append(f"{graph_key}: max node id {max(actual_nodes)} >= num_nodes {g['num_nodes']}")

        # Check num_edges
        if len(g["edge_list"]) != g["num_edges"]:
            issues.append(f"{graph_key}: edge_list length {len(g['edge_list'])} != num_edges {g['num_edges']}")

        # Check degree array length
        if len(g["node_features"]["degree"]) != g["num_nodes"]:
            issues.append(f"{graph_key}: degree array length {len(g['node_features']['degree'])} != num_nodes {g['num_nodes']}")

        # Check degree consistency
        degree_count = [0] * g["num_nodes"]
        for u, v in g["edge_list"]:
            if u < g["num_nodes"]:
                degree_count[u] += 1
            if v < g["num_nodes"]:
                degree_count[v] += 1

        for node_id in range(g["num_nodes"]):
            if node_id < len(g["node_features"]["degree"]):
                if degree_count[node_id] != g["node_features"]["degree"][node_id]:
                    issues.append(
                        f"{graph_key}: node {node_id} degree mismatch: "
                        f"computed={degree_count[node_id]} vs stored={g['node_features']['degree'][node_id]}"
                    )

    return issues


def main():
    """Main entry point: collect all graph pairs and assemble dataset."""
    print("=" * 70)
    print("Graph Expressiveness Benchmark Data Collection for ISP Testing")
    print("=" * 70)

    all_pairs = []

    # Step 1: BREC
    brec_pairs = collect_brec_pairs()
    all_pairs.extend(brec_pairs)

    # Step 2: CSL
    csl_pairs = collect_csl_pairs(pair_id_offset=len(all_pairs))
    all_pairs.extend(csl_pairs)

    # Step 3: Strongly Regular
    sr_pairs = collect_strongly_regular_pairs(pair_id_offset=len(all_pairs))
    all_pairs.extend(sr_pairs)

    # Step 4: Molecular
    mol_pairs = collect_molecular_pairs(pair_id_offset=len(all_pairs))
    all_pairs.extend(mol_pairs)

    # Step 5: Classic WL failures
    wl_pairs = collect_classic_wl_pairs(pair_id_offset=len(all_pairs))
    all_pairs.extend(wl_pairs)

    # Re-number pair IDs sequentially
    for i, pair in enumerate(all_pairs):
        pair["pair_id"] = i

    # ── Step 6: Validate ──
    print("\n" + "=" * 60)
    print("STEP 6: Validation")
    print("=" * 60)

    total_issues = 0
    for pair in all_pairs:
        issues = validate_pair(pair)
        if issues:
            print(f"  Pair {pair['pair_id']} ({pair['category']}): {len(issues)} issue(s)")
            for issue in issues:
                print(f"    - {issue}")
            total_issues += len(issues)

    if total_issues == 0:
        print("  All pairs validated successfully! ✓")
    else:
        print(f"\n  WARNING: {total_issues} total issues found!")

    # ── Category statistics ──
    print("\n" + "=" * 60)
    print("STEP 7: Summary Statistics")
    print("=" * 60)

    category_counts = {}
    category_node_ranges = {}
    category_edge_ranges = {}
    category_sources = {}

    for pair in all_pairs:
        cat = pair["category"]
        if cat not in category_counts:
            category_counts[cat] = 0
            category_node_ranges[cat] = [float('inf'), 0]
            category_edge_ranges[cat] = [float('inf'), 0]
            category_sources[cat] = pair["source"]

        category_counts[cat] += 1

        for gk in ["graph_a", "graph_b"]:
            nn = pair[gk]["num_nodes"]
            ne = pair[gk]["num_edges"]
            category_node_ranges[cat][0] = min(category_node_ranges[cat][0], nn)
            category_node_ranges[cat][1] = max(category_node_ranges[cat][1], nn)
            category_edge_ranges[cat][0] = min(category_edge_ranges[cat][0], ne)
            category_edge_ranges[cat][1] = max(category_edge_ranges[cat][1], ne)

    for cat in sorted(category_counts.keys()):
        nr = category_node_ranges[cat]
        er = category_edge_ranges[cat]
        print(f"  {cat}: {category_counts[cat]} pairs, "
              f"nodes=[{nr[0]}-{nr[1]}], edges=[{er[0]}-{er[1]}], "
              f"source={category_sources[cat]}")

    print(f"\n  Total pairs: {len(all_pairs)}")

    # Check if we have enough pairs (target: at least 70)
    if len(all_pairs) < 70:
        print(f"  WARNING: Only {len(all_pairs)} pairs (target: ≥70). Consider adding more.")
    else:
        print(f"  Target of ≥70 pairs met: ✓")

    # ── Assemble final dataset ──
    categories_meta = {}
    for cat in sorted(category_counts.keys()):
        categories_meta[cat] = {
            "count": category_counts[cat],
            "source": category_sources[cat],
            "node_range": category_node_ranges[cat],
            "edge_range": category_edge_ranges[cat],
        }

    dataset = {
        "metadata": {
            "description": "Graph expressiveness benchmark for ISP testing",
            "total_pairs": len(all_pairs),
            "categories": categories_meta,
        },
        "pairs": all_pairs,
    }

    # ── Step 8: Save outputs ──
    print("\n" + "=" * 60)
    print("STEP 8: Saving Output Files")
    print("=" * 60)

    # Full dataset
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  Full dataset: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size:,} bytes)")

    # Mini dataset: 10 pairs (2 BREC, 2 CSL, 2 SR, 2 molecular, 2 classic WL)
    mini_pairs = []
    mini_cats = {
        "brec": 2,
        "csl": 2,
        "strongly_regular": 2,
        "molecular": 2,
        "classic_wl_failures": 2,
    }

    for cat_prefix, limit in mini_cats.items():
        count = 0
        for pair in all_pairs:
            if pair["category"].startswith(cat_prefix) and count < limit:
                mini_pairs.append(pair)
                count += 1

    # Re-number
    for i, pair in enumerate(mini_pairs):
        pair_copy = dict(pair)
        pair_copy["pair_id"] = i
        mini_pairs[i] = pair_copy

    mini_categories = {}
    for pair in mini_pairs:
        cat = pair["category"]
        if cat not in mini_categories:
            mini_categories[cat] = {"count": 0, "source": pair["source"]}
        mini_categories[cat]["count"] += 1

    mini_dataset = {
        "metadata": {
            "description": "Graph expressiveness benchmark for ISP testing (mini version)",
            "total_pairs": len(mini_pairs),
            "categories": mini_categories,
        },
        "pairs": mini_pairs,
    }

    with open(MINI_FILE, "w") as f:
        json.dump(mini_dataset, f, indent=2)
    print(f"  Mini dataset: {MINI_FILE} ({MINI_FILE.stat().st_size:,} bytes)")

    # Preview dataset: 3 pairs (1 BREC, 1 SR, 1 molecular)
    preview_pairs = []
    preview_cats = {
        "brec": 1,
        "strongly_regular": 1,
        "molecular": 1,
    }

    for cat_prefix, limit in preview_cats.items():
        count = 0
        for pair in all_pairs:
            if pair["category"].startswith(cat_prefix) and count < limit:
                pair_copy = dict(pair)
                pair_copy["pair_id"] = len(preview_pairs)
                preview_pairs.append(pair_copy)
                count += 1

    preview_categories = {}
    for pair in preview_pairs:
        cat = pair["category"]
        if cat not in preview_categories:
            preview_categories[cat] = {"count": 0, "source": pair["source"]}
        preview_categories[cat]["count"] += 1

    preview_dataset = {
        "metadata": {
            "description": "Graph expressiveness benchmark for ISP testing (preview version)",
            "total_pairs": len(preview_pairs),
            "categories": preview_categories,
        },
        "pairs": preview_pairs,
    }

    with open(PREVIEW_FILE, "w") as f:
        json.dump(preview_dataset, f, indent=2)
    print(f"  Preview dataset: {PREVIEW_FILE} ({PREVIEW_FILE.stat().st_size:,} bytes)")

    print("\n" + "=" * 60)
    print("DONE! All data collected and saved.")
    print("=" * 60)

    return dataset


if __name__ == "__main__":
    main()

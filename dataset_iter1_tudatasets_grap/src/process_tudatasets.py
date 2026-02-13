"""
Process TUDataset graph classification benchmarks (MUTAG, PROTEINS, IMDB-BINARY, PTC_MR)
into standardized JSON format with 10-fold CV splits and published baseline accuracies.

Outputs:
  - data_out.json (full, all 2645 graphs)
  - data_out_mini.json (~10% stratified sample)
  - data_out_preview.json (5 graphs per dataset)
"""

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Step 1: Parse TUDataset raw files
# ============================================================

def parse_tu_dataset(ds_name: str, base_dir: str = "/tmp/tudatasets") -> dict:
    """Parse a TUDataset from its raw text files."""
    ds_dir = os.path.join(base_dir, ds_name)

    # 1. Read edges (1-indexed pairs)
    edges = []
    with open(f"{ds_dir}/{ds_name}_A.txt") as f:
        for line in f:
            parts = line.strip().split(",")
            src, dst = int(parts[0].strip()), int(parts[1].strip())
            edges.append((src, dst))

    # 2. Read graph indicators (which graph each node belongs to)
    graph_indicator = []
    with open(f"{ds_dir}/{ds_name}_graph_indicator.txt") as f:
        for line in f:
            graph_indicator.append(int(line.strip()))

    # 3. Read graph labels
    graph_labels = []
    with open(f"{ds_dir}/{ds_name}_graph_labels.txt") as f:
        for line in f:
            graph_labels.append(int(line.strip()))

    # 4. Read node labels (if available)
    node_labels = None
    nl_path = f"{ds_dir}/{ds_name}_node_labels.txt"
    if os.path.exists(nl_path):
        node_labels = []
        with open(nl_path) as f:
            for line in f:
                node_labels.append(int(line.strip()))

    # 5. Read node attributes (if available, e.g., PROTEINS)
    node_attrs = None
    na_path = f"{ds_dir}/{ds_name}_node_attributes.txt"
    if os.path.exists(na_path):
        node_attrs = []
        with open(na_path) as f:
            for line in f:
                vals = [float(x.strip()) for x in line.strip().split(",")]
                node_attrs.append(vals)

    return {
        "edges": edges,
        "graph_indicator": graph_indicator,
        "graph_labels": graph_labels,
        "node_labels": node_labels,
        "node_attrs": node_attrs,
    }


# ============================================================
# Step 2: Build per-graph records with 0-indexed node IDs
# ============================================================

def build_graphs(parsed: dict, ds_name: str) -> list:
    """Convert parsed TUDataset data into per-graph records."""
    edges = parsed["edges"]
    graph_indicator = parsed["graph_indicator"]
    graph_labels = parsed["graph_labels"]
    node_labels = parsed["node_labels"]
    node_attrs = parsed["node_attrs"]

    # Group nodes by graph (1-indexed graph IDs)
    graph_nodes = defaultdict(list)
    for node_id_1indexed, g_id in enumerate(graph_indicator, 1):
        graph_nodes[g_id].append(node_id_1indexed)

    num_graphs = len(graph_labels)

    # Determine global max node label for consistent one-hot dimensionality
    global_max_label = None
    if node_labels is not None:
        global_max_label = max(node_labels)

    # Print label info for verification
    unique_raw_labels = sorted(set(graph_labels))
    print(f"  {ds_name}: {num_graphs} graphs, raw labels = {unique_raw_labels}")

    records = []

    for g_id in range(1, num_graphs + 1):
        nodes = sorted(graph_nodes[g_id])
        node_set = set(nodes)
        # Map global 1-indexed node IDs to local 0-indexed
        global_to_local = {gn: i for i, gn in enumerate(nodes)}

        # Extract edges for this graph (keep only one direction: src < dst for undirected)
        edge_list = []
        seen = set()
        for src, dst in edges:
            if src in node_set and dst in node_set:
                a, b = global_to_local[src], global_to_local[dst]
                edge_key = (min(a, b), max(a, b))
                if edge_key not in seen:
                    seen.add(edge_key)
                    edge_list.append([edge_key[0], edge_key[1]])

        # Build node features
        if node_labels is not None and node_attrs is not None:
            # Both labels and attributes: concatenate one-hot label + attributes
            # (e.g., PROTEINS has both node_labels and node_attributes)
            features = []
            for gn in nodes:
                lbl = node_labels[gn - 1]
                one_hot = [0] * (global_max_label + 1)
                one_hot[lbl] = 1
                attr = node_attrs[gn - 1]
                features.append(one_hot + attr)
        elif node_labels is not None:
            # One-hot encode discrete labels
            features = []
            for gn in nodes:
                lbl = node_labels[gn - 1]
                one_hot = [0] * (global_max_label + 1)
                one_hot[lbl] = 1
                features.append(one_hot)
        elif node_attrs is not None:
            features = [node_attrs[gn - 1] for gn in nodes]
        else:
            # No features: use node degree as single feature
            deg_count = Counter()
            for src, dst in edges:
                if src in node_set:
                    deg_count[global_to_local[src]] += 1
            features = [[deg_count.get(i, 0)] for i in range(len(nodes))]

        # Map labels to 0/1
        raw_label = graph_labels[g_id - 1]
        if ds_name == "PROTEINS":
            # PROTEINS uses 1/2 convention
            label = raw_label - 1
        elif ds_name in ("MUTAG", "PTC_MR"):
            # MUTAG and PTC_MR use -1/1 convention
            label = 1 if raw_label == 1 else 0
        else:
            # IMDB-BINARY uses 0/1 already
            label = raw_label

        records.append({
            "graph_id": g_id - 1,
            "dataset": ds_name,
            "edge_list": edge_list,
            "node_features": features,
            "label": label,
            "num_nodes": len(nodes),
            "num_edges": len(edge_list),
            "fold": -1,
        })

    return records


# ============================================================
# Step 3: Assign 10-fold stratified CV splits
# ============================================================

def assign_folds(records: list, seed: int = 42) -> list:
    """Assign stratified 10-fold CV splits to graph records."""
    # Manual stratified splitting (no sklearn dependency)
    random.seed(seed)

    labels = [r["label"] for r in records]
    indices_by_class = defaultdict(list)
    for i, lbl in enumerate(labels):
        indices_by_class[lbl].append(i)

    # Shuffle within each class
    for lbl in indices_by_class:
        random.shuffle(indices_by_class[lbl])

    # Assign folds round-robin within each class
    fold_assignments = [0] * len(records)
    for lbl in sorted(indices_by_class.keys()):
        indices = indices_by_class[lbl]
        for rank, idx in enumerate(indices):
            fold_assignments[idx] = rank % 10

    for i, fold in enumerate(fold_assignments):
        records[i]["fold"] = fold

    return records


# ============================================================
# Step 4: Published baseline accuracies
# ============================================================

def get_baselines() -> dict:
    """Return published baseline accuracies from GIN and GSN papers."""
    return {
        "source_papers": [
            {
                "name": "GIN",
                "ref": "Xu et al., ICLR 2019",
                "arxiv": "1810.00826",
                "title": "How Powerful are Graph Neural Networks?",
            },
            {
                "name": "GSN",
                "ref": "Bouritsas et al., TPAMI 2022",
                "arxiv": "2006.09252",
                "title": "Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting",
            },
            {
                "name": "PEARL",
                "ref": "Kalogiannis et al., ICLR 2025",
                "title": "Learning Efficient Positional Encodings with Graph Neural Networks",
                "note": "PEARL results not available for MUTAG/PROTEINS/IMDB-BINARY/PTC_MR datasets. Evaluated on REDDIT-B/M and ZINC.",
            },
        ],
        "results": {
            "MUTAG": {
                "GIN-0": {"mean": 89.4, "std": 5.6},
                "GIN-eps": {"mean": 89.0, "std": 6.0},
                "GCN": {"mean": 85.6, "std": 5.8},
                "GraphSAGE": {"mean": 85.1, "std": 7.6},
                "WL_kernel": {"mean": 90.4, "std": 5.7},
                "GSN": {"mean": 92.2, "std": 7.5},
                "PPGN": {"mean": 90.6, "std": 8.7},
                "IGN": {"mean": 83.9, "std": 13.0},
            },
            "PROTEINS": {
                "GIN-0": {"mean": 76.2, "std": 2.8},
                "GIN-eps": {"mean": 75.9, "std": 3.8},
                "GCN": {"mean": 76.0, "std": 3.2},
                "GraphSAGE": {"mean": 75.9, "std": 3.2},
                "WL_kernel": {"mean": 75.0, "std": 3.1},
                "GSN": {"mean": 76.6, "std": 5.0},
                "PPGN": {"mean": 77.2, "std": 4.7},
                "IGN": {"mean": 76.6, "std": 5.5},
            },
            "IMDB-BINARY": {
                "GIN-0": {"mean": 75.1, "std": 5.1},
                "GIN-eps": {"mean": 74.3, "std": 5.1},
                "GCN": {"mean": 74.0, "std": 3.4},
                "GraphSAGE": {"mean": 72.3, "std": 5.3},
                "WL_kernel": {"mean": 73.8, "std": 3.9},
                "GSN": {"mean": 77.8, "std": 3.3},
                "PPGN": {"mean": 73.0, "std": 5.8},
                "IGN": {"mean": 72.0, "std": 5.5},
            },
            "PTC_MR": {
                "GIN-0": {"mean": 64.6, "std": 7.0},
                "GIN-eps": {"mean": 63.7, "std": 8.2},
                "GCN": {"mean": 64.2, "std": 4.3},
                "GraphSAGE": {"mean": 63.9, "std": 7.7},
                "WL_kernel": {"mean": 59.9, "std": 4.3},
                "GSN": {"mean": 68.2, "std": 7.2},
                "PPGN": {"mean": 66.2, "std": 6.6},
                "IGN": {"mean": 58.5, "std": 6.9},
            },
        },
    }


# ============================================================
# Step 5: Validation
# ============================================================

def validate_records(all_records: list) -> bool:
    """Validate the processed records match expected counts and properties."""
    expected_counts = {
        "MUTAG": 188,
        "PROTEINS": 1113,
        "IMDB-BINARY": 1000,
        "PTC_MR": 344,
    }
    total_expected = sum(expected_counts.values())

    # Check total count
    assert len(all_records) == total_expected, (
        f"Expected {total_expected} total graphs, got {len(all_records)}"
    )

    # Check per-dataset counts
    ds_counts = Counter(r["dataset"] for r in all_records)
    for ds, expected in expected_counts.items():
        actual = ds_counts.get(ds, 0)
        assert actual == expected, (
            f"{ds}: expected {expected} graphs, got {actual}"
        )

    # Check fold assignments
    for r in all_records:
        assert 0 <= r["fold"] <= 9, (
            f"Graph {r['graph_id']} in {r['dataset']} has invalid fold={r['fold']}"
        )

    # Check fold distribution per dataset
    for ds in expected_counts:
        ds_records = [r for r in all_records if r["dataset"] == ds]
        fold_counts = Counter(r["fold"] for r in ds_records)
        assert len(fold_counts) == 10, (
            f"{ds}: expected 10 folds, got {len(fold_counts)} ({fold_counts})"
        )

    # Check labels are binary (0 or 1)
    for r in all_records:
        assert r["label"] in (0, 1), (
            f"Graph {r['graph_id']} in {r['dataset']} has label={r['label']}, expected 0 or 1"
        )

    # Check label distribution per dataset (exactly 2 classes)
    for ds in expected_counts:
        ds_labels = set(r["label"] for r in all_records if r["dataset"] == ds)
        assert len(ds_labels) == 2, (
            f"{ds}: expected 2 classes, got {len(ds_labels)} ({ds_labels})"
        )

    # Check non-empty edge lists (every graph should have at least one edge)
    for r in all_records:
        assert len(r["edge_list"]) > 0, (
            f"Graph {r['graph_id']} in {r['dataset']} has empty edge_list"
        )

    # Check node features consistency within each dataset
    for ds in expected_counts:
        ds_records = [r for r in all_records if r["dataset"] == ds]
        feat_dims = set()
        for r in ds_records:
            for feat in r["node_features"]:
                feat_dims.add(len(feat))
        assert len(feat_dims) == 1, (
            f"{ds}: inconsistent feature dimensions: {feat_dims}"
        )

    # Check IMDB-BINARY has degree-based features (not empty)
    for r in all_records:
        if r["dataset"] == "IMDB-BINARY":
            assert all(len(f) > 0 for f in r["node_features"]), (
                f"IMDB-BINARY graph {r['graph_id']} has empty features"
            )

    print("All validation checks PASSED!")
    return True


# ============================================================
# Step 6: Build metadata
# ============================================================

def build_metadata(all_records: list) -> dict:
    """Build metadata section for data_out.json."""
    datasets_meta = {}
    for ds_name in ["MUTAG", "PROTEINS", "IMDB-BINARY", "PTC_MR"]:
        ds_records = [r for r in all_records if r["dataset"] == ds_name]
        num_graphs = len(ds_records)
        num_classes = len(set(r["label"] for r in ds_records))
        avg_nodes = sum(r["num_nodes"] for r in ds_records) / num_graphs
        avg_edges = sum(r["num_edges"] for r in ds_records) / num_graphs
        feat_dim = len(ds_records[0]["node_features"][0]) if ds_records else 0

        ds_meta = {
            "num_graphs": num_graphs,
            "num_classes": num_classes,
            "avg_nodes": round(avg_nodes, 2),
            "avg_edges": round(avg_edges, 2),
            "feature_dim": feat_dim,
        }

        if ds_name == "MUTAG":
            ds_meta.update({
                "has_node_labels": True,
                "has_edge_labels": True,
                "node_label_count": 7,
                "edge_label_count": 4,
                "description": "188 mutagenic aromatic/heteroaromatic nitro compounds, binary mutagenicity classification",
            })
        elif ds_name == "PROTEINS":
            ds_meta.update({
                "has_node_labels": True,
                "has_node_attrs": True,
                "node_label_count": 3,
                "node_attr_dim": 1,
                "description": "1113 protein graphs, binary enzyme/non-enzyme classification",
            })
        elif ds_name == "IMDB-BINARY":
            ds_meta.update({
                "has_node_labels": False,
                "feature_type": "degree",
                "description": "1000 movie collaboration ego-networks, binary genre classification (Action vs Romance)",
            })
        elif ds_name == "PTC_MR":
            ds_meta.update({
                "has_node_labels": True,
                "has_edge_labels": True,
                "node_label_count": 18,
                "description": "344 chemical compounds, binary carcinogenicity classification (male rats)",
            })

        # Label distribution
        label_dist = Counter(r["label"] for r in ds_records)
        ds_meta["label_distribution"] = {
            "class_0": label_dist[0],
            "class_1": label_dist[1],
        }

        datasets_meta[ds_name] = ds_meta

    return {
        "description": "TUDataset graph classification benchmarks for ISP-GNN evaluation",
        "source": "https://chrsmrrs.github.io/datasets/",
        "source_paper": "Morris et al., TUDataset: A collection of benchmark datasets for learning with graphs, arXiv:2007.08663",
        "datasets": datasets_meta,
        "cv_folds": 10,
        "cv_seed": 42,
        "evaluation_protocol": "Stratified 10-fold CV, report best epoch accuracy averaged over folds (GIN protocol)",
        "total_graphs": len(all_records),
    }


# ============================================================
# Step 7: Create mini and preview versions
# ============================================================

def create_mini_version(all_records: list, fraction: float = 0.1, seed: int = 42) -> list:
    """Create a ~10% stratified sample per dataset."""
    random.seed(seed)
    mini_records = []

    for ds_name in ["MUTAG", "PROTEINS", "IMDB-BINARY", "PTC_MR"]:
        ds_records = [r for r in all_records if r["dataset"] == ds_name]
        # Stratify by label
        by_label = defaultdict(list)
        for r in ds_records:
            by_label[r["label"]].append(r)

        for lbl in sorted(by_label.keys()):
            records_lbl = by_label[lbl]
            random.shuffle(records_lbl)
            n_sample = max(1, int(len(records_lbl) * fraction))
            mini_records.extend(records_lbl[:n_sample])

    return mini_records


def create_preview_version(all_records: list, per_dataset: int = 5) -> list:
    """Create a preview with N graphs per dataset."""
    preview_records = []
    for ds_name in ["MUTAG", "PROTEINS", "IMDB-BINARY", "PTC_MR"]:
        ds_records = [r for r in all_records if r["dataset"] == ds_name]
        preview_records.extend(ds_records[:per_dataset])
    return preview_records


# ============================================================
# Main
# ============================================================

def main():
    WORKSPACE = "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260213_112012/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
    DATASETS = ["MUTAG", "PROTEINS", "IMDB-BINARY", "PTC_MR"]

    print("=" * 60)
    print("Processing TUDataset benchmarks")
    print("=" * 60)

    all_records = []

    for ds_name in DATASETS:
        print(f"\nProcessing {ds_name}...")
        parsed = parse_tu_dataset(ds_name)
        records = build_graphs(parsed, ds_name)
        records = assign_folds(records)
        all_records.extend(records)
        print(f"  Processed {len(records)} graphs")

        # Print stats
        label_dist = Counter(r["label"] for r in records)
        avg_nodes = sum(r["num_nodes"] for r in records) / len(records)
        avg_edges = sum(r["num_edges"] for r in records) / len(records)
        feat_dim = len(records[0]["node_features"][0])
        print(f"  Labels: {dict(label_dist)}")
        print(f"  Avg nodes: {avg_nodes:.2f}, Avg edges: {avg_edges:.2f}")
        print(f"  Feature dim: {feat_dim}")

    print(f"\nTotal records: {len(all_records)}")

    # Validate
    print("\n" + "=" * 60)
    print("Validating...")
    validate_records(all_records)

    # Build output
    baselines = get_baselines()
    metadata = build_metadata(all_records)

    full_output = {
        "metadata": metadata,
        "baselines": baselines,
        "graphs": all_records,
    }

    # Write full version
    full_path = os.path.join(WORKSPACE, "data_out.json")
    with open(full_path, "w") as f:
        json.dump(full_output, f, indent=2)
    full_size = os.path.getsize(full_path)
    print(f"\nFull version: {full_path}")
    print(f"  Size: {full_size:,} bytes ({full_size / 1024 / 1024:.2f} MB)")
    print(f"  Graphs: {len(all_records)}")

    # Write mini version
    mini_records = create_mini_version(all_records)
    mini_output = {
        "metadata": {**metadata, "description": metadata["description"] + " (MINI ~10% sample)"},
        "baselines": baselines,
        "graphs": mini_records,
    }
    mini_path = os.path.join(WORKSPACE, "data_out_mini.json")
    with open(mini_path, "w") as f:
        json.dump(mini_output, f, indent=2)
    mini_size = os.path.getsize(mini_path)
    print(f"\nMini version: {mini_path}")
    print(f"  Size: {mini_size:,} bytes ({mini_size / 1024 / 1024:.2f} MB)")
    print(f"  Graphs: {len(mini_records)}")

    # Write preview version
    preview_records = create_preview_version(all_records)
    preview_output = {
        "metadata": {**metadata, "description": metadata["description"] + " (PREVIEW 5 per dataset)"},
        "baselines": baselines,
        "graphs": preview_records,
    }
    preview_path = os.path.join(WORKSPACE, "data_out_preview.json")
    with open(preview_path, "w") as f:
        json.dump(preview_output, f, indent=2)
    preview_size = os.path.getsize(preview_path)
    print(f"\nPreview version: {preview_path}")
    print(f"  Size: {preview_size:,} bytes ({preview_size / 1024 / 1024:.2f} MB)")
    print(f"  Graphs: {len(preview_records)}")

    print("\n" + "=" * 60)
    print("Done! All files written successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

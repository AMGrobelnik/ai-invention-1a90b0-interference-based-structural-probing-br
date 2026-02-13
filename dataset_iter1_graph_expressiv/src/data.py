# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.26.0",
# ]
# ///
"""
Standardize graph expressiveness benchmark data to exp_sel_data_out schema.

Reads collected graph pair data from temp/datasets/data_out.json and converts
each pair into an example conforming to the exp_sel_data_out.json schema.

Each graph pair becomes one example:
- input: JSON string with both graphs (edge lists + node features)
- output: ground truth label ("non_isomorphic" or "isomorphic")
- metadata_*: category, WL level, source, graph sizes, etc.

Output is grouped by dataset (category), with 4 selected datasets.
"""

import json
import resource
import sys
from pathlib import Path

# ── Resource limits (14GB RAM, 1 hour CPU) ─────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
INPUT_FILE = WORKSPACE / "temp" / "datasets" / "data_out.json"
OUTPUT_FILE = WORKSPACE / "full_data_out.json"


def pair_to_example(pair: dict) -> dict:
    """Convert a single graph pair to schema-conforming example."""
    # Build input: JSON string with both graphs
    input_data = {
        "graph_a": {
            "num_nodes": pair["graph_a"]["num_nodes"],
            "num_edges": pair["graph_a"]["num_edges"],
            "edge_list": pair["graph_a"]["edge_list"],
            "node_degrees": pair["graph_a"]["node_features"]["degree"],
        },
        "graph_b": {
            "num_nodes": pair["graph_b"]["num_nodes"],
            "num_edges": pair["graph_b"]["num_edges"],
            "edge_list": pair["graph_b"]["edge_list"],
            "node_degrees": pair["graph_b"]["node_features"]["degree"],
        },
    }

    # Build output: ground truth
    if pair["is_isomorphic"]:
        output_label = "isomorphic"
    else:
        output_label = "non_isomorphic"

    # Build example with metadata
    example = {
        "input": json.dumps(input_data, separators=(",", ":")),
        "output": output_label,
        "metadata_pair_id": pair["pair_id"],
        "metadata_wl_level": pair["wl_level"],
        "metadata_same_wl_color": pair["same_wl_color"],
        "metadata_source": pair["source"],
        "metadata_graph_a_nodes": pair["graph_a"]["num_nodes"],
        "metadata_graph_a_edges": pair["graph_a"]["num_edges"],
        "metadata_graph_b_nodes": pair["graph_b"]["num_nodes"],
        "metadata_graph_b_edges": pair["graph_b"]["num_edges"],
        "metadata_task_type": "classification",
        "metadata_n_classes": 2,
        "metadata_notes": pair.get("notes", ""),
    }

    return example


def main() -> None:
    """Load data_out.json and convert to exp_sel_data_out schema."""
    print(f"Loading data from: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    raw_data = json.loads(INPUT_FILE.read_text())
    total_pairs = raw_data["metadata"]["total_pairs"]
    print(f"Loaded {total_pairs} pairs across {len(raw_data['metadata']['categories'])} categories")

    # ── Selected top 4 datasets for ISP testing ──
    # Selection rationale:
    # 1. brec_regular (15 pairs) — Hardest 1-WL pairs, strongly regular graphs
    # 2. csl (20 pairs) — Largest set, established JMLR 2023 benchmark, cycle-based
    # 3. brec_cfi (10 pairs) — Very hard (3-WL level), combinatorial constructions
    # 4. brec_basic (10 pairs) — Standard 1-WL difficulty, general graphs, baseline
    #
    # Total: 55 pairs across 3 difficulty levels and diverse structures
    selected_datasets = {"brec_basic", "brec_regular", "brec_cfi", "csl"}

    category_to_dataset = {
        "brec_basic": "brec_basic",
        "brec_regular": "brec_regular",
        "brec_cfi": "brec_cfi",
        "csl": "csl",
    }

    # ── Group pairs by dataset (only selected) ──
    datasets_dict: dict[str, list[dict]] = {}
    for pair in raw_data["pairs"]:
        category = pair["category"]
        dataset_name = category_to_dataset.get(category)

        if dataset_name is None:
            continue  # Skip non-selected categories

        if dataset_name not in datasets_dict:
            datasets_dict[dataset_name] = []

        example = pair_to_example(pair)
        datasets_dict[dataset_name].append(example)

    # ── Build output in schema format ──
    # Order datasets by difficulty: basic → regular → cfi → csl
    dataset_order = [
        "brec_basic",
        "brec_regular",
        "brec_cfi",
        "csl",
    ]

    datasets_list = []
    for ds_name in dataset_order:
        if ds_name in datasets_dict:
            examples = datasets_dict[ds_name]
            datasets_list.append({
                "dataset": ds_name,
                "examples": examples,
            })
            print(f"  {ds_name}: {len(examples)} examples")

    output_data = {
        "metadata": {
            "description": "Graph expressiveness benchmark for ISP testing",
            "total_examples": sum(len(d["examples"]) for d in datasets_list),
            "total_datasets": len(datasets_list),
            "sources": {
                "BREC": "GitHub GraphPKU/BREC (NeurIPS 2023 benchmark)",
                "CSL": "HuggingFace graphs-datasets/CSL (JMLR 2023)",
                "programmatic": "NetworkX graph generation",
            },
            "schema": "exp_sel_data_out",
        },
        "datasets": datasets_list,
    }

    # ── Write output ──
    OUTPUT_FILE.write_text(json.dumps(output_data, indent=2))
    file_size = OUTPUT_FILE.stat().st_size
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"Total datasets: {len(datasets_list)}")
    print(f"Total examples: {output_data['metadata']['total_examples']}")


if __name__ == "__main__":
    main()

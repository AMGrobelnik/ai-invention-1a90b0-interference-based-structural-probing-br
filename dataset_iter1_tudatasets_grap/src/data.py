# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Convert TUDataset graph classification benchmarks from data_out.json
into the exp_sel_data_out.json schema format (full_data_out.json).

Each graph becomes one example with:
- input: JSON string of graph structure (edge_list, node_features, num_nodes, num_edges)
- output: class label as string ("0" or "1")
- metadata_*: fold, task_type, n_classes, graph_id, feature_dim, avg_nodes, avg_edges
"""

import json
import os
import resource
import sys
import time

# Resource limits: 14GB RAM, 1 hour CPU
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

WORKSPACE = "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260213_112012/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
INPUT_FILE = os.path.join(WORKSPACE, "data_out.json")
OUTPUT_FILE = os.path.join(WORKSPACE, "full_data_out.json")

# Optional: limit number of examples for gradual scaling
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = no limit


def main():
    t0 = time.time()

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    graphs = data["graphs"]
    metadata = data["metadata"]
    baselines = data["baselines"]

    print(f"Loaded {len(graphs)} graphs in {time.time() - t0:.2f}s")

    # Group graphs by dataset
    dataset_names = ["MUTAG", "PROTEINS", "IMDB-BINARY", "PTC_MR"]
    graphs_by_ds = {ds: [] for ds in dataset_names}
    for g in graphs:
        ds = g["dataset"]
        if ds in graphs_by_ds:
            graphs_by_ds[ds].append(g)

    # Build output in schema format
    output_datasets = []

    for ds_name in dataset_names:
        ds_graphs = graphs_by_ds[ds_name]
        ds_meta = metadata["datasets"][ds_name]

        # Apply limit if set
        if MAX_EXAMPLES > 0:
            ds_graphs = ds_graphs[:MAX_EXAMPLES]

        examples = []
        for g in ds_graphs:
            # Build input: JSON string of graph structure
            graph_input = {
                "edge_list": g["edge_list"],
                "node_features": g["node_features"],
                "num_nodes": g["num_nodes"],
                "num_edges": g["num_edges"],
            }
            input_str = json.dumps(graph_input, separators=(",", ":"))

            # Build output: class label as string
            output_str = str(g["label"])

            # Build metadata fields (flat, prefixed with metadata_)
            example = {
                "input": input_str,
                "output": output_str,
                "metadata_fold": g["fold"],
                "metadata_graph_id": g["graph_id"],
                "metadata_task_type": "classification",
                "metadata_n_classes": 2,
                "metadata_num_nodes": g["num_nodes"],
                "metadata_num_edges": g["num_edges"],
                "metadata_feature_dim": ds_meta["feature_dim"],
            }

            examples.append(example)

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

        print(f"  {ds_name}: {len(examples)} examples")

    # Build final output with top-level metadata
    final_output = {
        "metadata": {
            "description": "TUDataset graph classification benchmarks for ISP-GNN evaluation",
            "source": metadata["source"],
            "source_paper": metadata["source_paper"],
            "cv_folds": metadata["cv_folds"],
            "cv_seed": metadata["cv_seed"],
            "evaluation_protocol": metadata["evaluation_protocol"],
            "total_graphs": sum(len(d["examples"]) for d in output_datasets),
            "dataset_statistics": {
                ds_name: {
                    "num_graphs": ds_meta["num_graphs"],
                    "num_classes": ds_meta["num_classes"],
                    "avg_nodes": ds_meta["avg_nodes"],
                    "avg_edges": ds_meta["avg_edges"],
                    "feature_dim": ds_meta["feature_dim"],
                    "description": ds_meta["description"],
                    "label_distribution": ds_meta["label_distribution"],
                }
                for ds_name, ds_meta in metadata["datasets"].items()
            },
            "baselines": baselines,
        },
        "datasets": output_datasets,
    }

    print(f"\nWriting {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_output, f, indent=2)

    file_size = os.path.getsize(OUTPUT_FILE)
    total_examples = sum(len(d["examples"]) for d in output_datasets)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.2f}s")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"Total examples: {total_examples}")
    print(f"Datasets: {len(output_datasets)}")


if __name__ == "__main__":
    main()

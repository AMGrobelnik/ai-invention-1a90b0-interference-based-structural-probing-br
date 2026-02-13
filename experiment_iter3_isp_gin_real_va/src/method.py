#!/usr/bin/env python3
"""ISP-GIN Real-Valued Ablation & Interference Isolation Experiment.

Ablation experiment isolating complex-valued interference contribution in ISP-GIN.
Implements 5 methods (ISP-GIN, RealGIN-Aug, ISP-NoMP magnitude, ISP-NoMP phases,
WL baseline) and a random walk t-sweep across graph pairs from 4 datasets.
Pure numpy/networkx — no training needed.
"""

import json
import resource
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Resource limits (16 GB system → cap at 14 GB, 1 hr CPU)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPSILON_THRESHOLDS = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
PRIMARY_EPSILON = 1e-6
K_DEFAULT = 8
L_VALUES = [1, 2, 3, 4]
RW_T_VALUES = [2, 3, 4, 5, 6]


# ============================= GRAPH CONSTRUCTION =============================


def build_graph_from_example(example: dict) -> tuple[nx.Graph, nx.Graph]:
    """Parse 'input' JSON string into two networkx graphs."""
    input_data = json.loads(example["input"])

    ga = input_data["graph_a"]
    gb = input_data["graph_b"]

    G1 = nx.Graph()
    G1.add_nodes_from(range(ga["num_nodes"]))
    G1.add_edges_from(ga["edge_list"])

    G2 = nx.Graph()
    G2.add_nodes_from(range(gb["num_nodes"]))
    G2.add_edges_from(gb["edge_list"])

    return G1, G2


# ============================= NODE FEATURES ==================================


def compute_node_features(G: nx.Graph) -> dict[str, np.ndarray]:
    """Compute topological node features for a graph.

    Features: degree, clustering coefficient, neighbor degree variance,
    and random walk return probabilities for t=2..6.
    """
    n = G.number_of_nodes()
    features: dict[str, np.ndarray] = {}

    # 1. Degree
    degrees = np.array([G.degree(v) for v in range(n)], dtype=np.float64)
    features["degree"] = degrees

    # 2. Clustering coefficient
    cc = nx.clustering(G)
    features["clustering_coeff"] = np.array(
        [cc[v] for v in range(n)], dtype=np.float64
    )

    # 3. Neighbor degree variance
    ndv = np.zeros(n, dtype=np.float64)
    for v in range(n):
        if G.degree(v) > 0:
            neighbor_degs = [G.degree(u) for u in G.neighbors(v)]
            ndv[v] = np.var(neighbor_degs)
    features["neighbor_deg_var"] = ndv

    # 4. Random walk return probabilities for t=2..6
    A = nx.adjacency_matrix(G).toarray().astype(np.float64)
    deg_safe = np.maximum(degrees, 1.0)
    D_inv = np.diag(1.0 / deg_safe)
    P = D_inv @ A  # transition matrix

    for t in RW_T_VALUES:
        P_t = np.linalg.matrix_power(P, t)
        features[f"rw_return_t{t}"] = np.diag(P_t)

    return features


# ============================= HELPER: INIT FEATURE ===========================


def _get_init_features(
    features: dict[str, np.ndarray], init_type: str
) -> np.ndarray:
    """Get the scalar node feature vector f(v) for ISP initialization."""
    if init_type == "local_topology":
        cc = features["clustering_coeff"]
        ndv = features["neighbor_deg_var"]
        deg = features["degree"]
        ndv_max = np.max(ndv) + 1e-10
        deg_max = np.max(deg) + 1e-10
        return cc + ndv / ndv_max + deg / deg_max

    if init_type.startswith("random_walk_t"):
        t = int(init_type.split("t")[1])
        key = f"rw_return_t{t}"
        if key not in features:
            raise ValueError(f"Feature {key} not computed (t={t})")
        return features[key]

    raise ValueError(f"Unknown init_type: {init_type}")


# ============================= METHOD A: ISP-GIN ==============================


def isp_gin(
    G: nx.Graph,
    features: dict[str, np.ndarray],
    init_type: str,
    K: int = K_DEFAULT,
    L: int = 3,
    epsilon: float = 0.0,
) -> np.ndarray:
    """ISP-GIN: Complex-valued message passing with harmonic initialization.

    For each frequency channel k=1..K:
      1. Initialize: z_v^(0) = exp(i * omega_k * f(v))
      2. For l=1..L: z_v^(l) = (1+eps)*z_v^(l-1) + sum_{u in N(v)} z_u^(l-1)
      3. Extract: magnitude |z_v^(L)|

    Graph representation = sorted magnitudes concatenated across K channels.
    """
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G).toarray().astype(np.float64)
    f_v = _get_init_features(features, init_type)

    # K log-spaced frequencies: 0.1 to 100
    omegas = np.logspace(-1, 2, K)

    graph_repr = []
    for omega in omegas:
        z = np.exp(1j * omega * f_v)  # complex128

        for _ in range(L):
            z = (1 + epsilon) * z + A @ z

        magnitudes = np.abs(z)
        graph_repr.append(np.sort(magnitudes))

    return np.concatenate(graph_repr)


# ============================= METHOD B: REALGIN-AUG ==========================


def real_gin_aug(
    G: nx.Graph,
    features: dict[str, np.ndarray],
    feature_keys: list[str],
    L: int = 3,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Real-valued GIN with augmented topological features. NO complex arithmetic.

    For each node: x_v^(0) = [f1(v), f2(v), ..., fD(v)]
    For l=1..L: x_v^(l) = (1+eps)*x_v^(l-1) + sum_{u in N(v)} x_u^(l-1)
    Graph repr = sorted node features concatenated (lexicographic sort on rows).
    """
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G).toarray().astype(np.float64)

    X = np.column_stack([features[k] for k in feature_keys])  # (n, D)

    for _ in range(L):
        X = (1 + epsilon) * X + A @ X

    # Lexicographic sort for permutation invariance
    sort_idx = np.lexsort(X.T[::-1])
    return X[sort_idx].flatten()


# ============================= METHOD C: ISP-NoMP (magnitudes) ================


def isp_no_mp(
    G: nx.Graph,
    features: dict[str, np.ndarray],
    init_type: str,
    K: int = K_DEFAULT,
) -> np.ndarray:
    """ISP initialization only — NO message passing.

    z_v = exp(i * omega * f(v)), extract |z_v| immediately.
    NOTE: |exp(i*theta)| = 1 always, so this produces all-ones vectors.
    This is EXPECTED and proves MP is necessary for interference.
    """
    n = G.number_of_nodes()
    f_v = _get_init_features(features, init_type)
    omegas = np.logspace(-1, 2, K)

    graph_repr = []
    for omega in omegas:
        z = np.exp(1j * omega * f_v)
        magnitudes = np.abs(z)  # all 1.0 by construction
        graph_repr.append(np.sort(magnitudes))

    return np.concatenate(graph_repr)


# ============================= METHOD C': ISP-NoMP (phases) ===================


def isp_no_mp_phases(
    G: nx.Graph,
    features: dict[str, np.ndarray],
    init_type: str,
    K: int = K_DEFAULT,
) -> np.ndarray:
    """ISP-NoMP using sorted phases instead of magnitudes.

    phases = omega * f(v) mod 2pi, sorted.
    This reduces to: can sorted node features alone distinguish graphs?
    Gives ISP-NoMP a fairer chance than magnitude-based version.
    """
    n = G.number_of_nodes()
    f_v = _get_init_features(features, init_type)
    omegas = np.logspace(-1, 2, K)

    graph_repr = []
    for omega in omegas:
        phases = np.angle(np.exp(1j * omega * f_v))  # in [-pi, pi]
        graph_repr.append(np.sort(phases))

    return np.concatenate(graph_repr)


# ============================= METHOD D: 1-WL =================================


def wl_hash(G: nx.Graph, iterations: int = 5) -> str:
    """Standard 1-WL color refinement. Returns canonical hash string."""
    n = G.number_of_nodes()
    colors = {v: str(G.degree(v)) for v in range(n)}

    for _ in range(iterations):
        new_colors = {}
        for v in range(n):
            neighbor_colors = sorted([colors[u] for u in G.neighbors(v)])
            new_colors[v] = str(hash((colors[v], tuple(neighbor_colors))))
        colors = new_colors

    return str(sorted(colors.values()))


# ============================= DISTINGUISHABILITY TEST ========================


def are_distinguished(
    repr1: np.ndarray,
    repr2: np.ndarray,
    epsilon: float = PRIMARY_EPSILON,
) -> bool:
    """Two graphs are distinguished if their representations differ.

    Uses L-infinity norm (max absolute difference) > epsilon.
    """
    if repr1.shape != repr2.shape:
        return True  # different shapes → trivially different
    return float(np.max(np.abs(repr1 - repr2))) > epsilon


def _eps_to_safe_key(eps: float) -> str:
    """Convert epsilon to schema-safe key (no dots or minus signs)."""
    # 1e-10 → "1em10", 0.0001 → "1em4", 0.01 → "1em2"
    s = f"{eps:.0e}"  # e.g. "1e-10"
    return s.replace("-", "m").replace("+", "p").replace(".", "p")


def are_distinguished_all_eps(
    repr1: np.ndarray,
    repr2: np.ndarray,
) -> dict[str, bool]:
    """Test distinguishability at all epsilon thresholds."""
    results = {}
    if repr1.shape != repr2.shape:
        for eps in EPSILON_THRESHOLDS:
            results[f"eps_{_eps_to_safe_key(eps)}"] = True
        return results

    diff = float(np.max(np.abs(repr1 - repr2)))
    for eps in EPSILON_THRESHOLDS:
        results[f"eps_{_eps_to_safe_key(eps)}"] = diff > eps
    return results


# ============================= MAIN EXPERIMENT LOOP ===========================


@logger.catch
def run_experiment(
    data: dict,
    max_examples: int | None = None,
) -> dict:
    """Main experiment: process all pairs across all methods.

    Methods:
    1. WL baseline (1-WL hash)
    2. ISP-GIN with random_walk_t3 and local_topology, K=8, L=1..4
    3. RealGIN-Aug with all features, L=1..4
    4. ISP-NoMP (magnitude) with random_walk_t3 and local_topology
    5. ISP-NoMP (phases) with random_walk_t3 and local_topology
    6. ISP-GIN random walk sweep: t=2..6, K=8, L=3
    7. ISP-GIN extended frequency sweep (K=16) with random_walk_t3, L=3
    """
    results_by_dataset: list[dict] = []

    # Aggregate counters
    agg: dict[str, dict[str, int]] = {}  # method -> {distinguished, total}
    per_dataset_agg: dict[str, dict[str, dict[str, int]]] = {}  # ds -> method -> counts

    total_processed = 0

    for ds_entry in data["datasets"]:
        ds_name = ds_entry["dataset"]
        examples_out: list[dict] = []
        per_dataset_agg[ds_name] = {}

        examples = ds_entry["examples"]
        if max_examples is not None:
            # Calculate how many to take from this dataset proportionally,
            # but at least 1 if available, and cap at max_examples total
            remaining = max_examples - total_processed
            if remaining <= 0:
                break
            examples = examples[:remaining]

        logger.info(f"Processing dataset '{ds_name}': {len(examples)} pairs")

        for ex_idx, example in enumerate(examples):
            try:
                G1, G2 = build_graph_from_example(example)
                feats1 = compute_node_features(G1)
                feats2 = compute_node_features(G2)

                result: dict = {
                    "input": example["input"],
                    "output": example["output"],
                    "metadata_pair_id": example.get("metadata_pair_id", ""),
                    "metadata_dataset": ds_name,
                    "metadata_num_nodes_g1": G1.number_of_nodes(),
                    "metadata_num_nodes_g2": G2.number_of_nodes(),
                    "metadata_wl_level": example.get("metadata_wl_level", ""),
                }

                # --- WL baseline ---
                wl1, wl2 = wl_hash(G1), wl_hash(G2)
                wl_dist = "distinguished" if wl1 != wl2 else "not_distinguished"
                result["predict_wl_baseline"] = wl_dist
                _update_agg(agg, per_dataset_agg, ds_name, "wl_baseline", wl_dist)

                # --- ISP-GIN variants ---
                for init in ["random_walk_t3", "local_topology"]:
                    for L in L_VALUES:
                        r1 = isp_gin(G1, feats1, init, K=K_DEFAULT, L=L)
                        r2 = isp_gin(G2, feats2, init, K=K_DEFAULT, L=L)
                        dist = "distinguished" if are_distinguished(r1, r2) else "not_distinguished"
                        key = f"isp_gin_{init}_L{L}"
                        result[f"predict_{key}"] = dist
                        _update_agg(agg, per_dataset_agg, ds_name, key, dist)

                # --- RealGIN-Aug ---
                feature_keys_all = [
                    "clustering_coeff",
                    "neighbor_deg_var",
                    "rw_return_t3",
                    "degree",
                ]
                for L in L_VALUES:
                    r1 = real_gin_aug(G1, feats1, feature_keys_all, L=L)
                    r2 = real_gin_aug(G2, feats2, feature_keys_all, L=L)
                    dist = "distinguished" if are_distinguished(r1, r2) else "not_distinguished"
                    key = f"real_gin_aug_L{L}"
                    result[f"predict_{key}"] = dist
                    _update_agg(agg, per_dataset_agg, ds_name, key, dist)

                # --- ISP-NoMP (magnitude) ---
                for init in ["random_walk_t3", "local_topology"]:
                    r1 = isp_no_mp(G1, feats1, init, K=K_DEFAULT)
                    r2 = isp_no_mp(G2, feats2, init, K=K_DEFAULT)
                    dist = "distinguished" if are_distinguished(r1, r2) else "not_distinguished"
                    key = f"isp_nomp_mag_{init}"
                    result[f"predict_{key}"] = dist
                    _update_agg(agg, per_dataset_agg, ds_name, key, dist)

                # --- ISP-NoMP (phases) ---
                for init in ["random_walk_t3", "local_topology"]:
                    r1 = isp_no_mp_phases(G1, feats1, init, K=K_DEFAULT)
                    r2 = isp_no_mp_phases(G2, feats2, init, K=K_DEFAULT)
                    dist = "distinguished" if are_distinguished(r1, r2) else "not_distinguished"
                    key = f"isp_nomp_phase_{init}"
                    result[f"predict_{key}"] = dist
                    _update_agg(agg, per_dataset_agg, ds_name, key, dist)

                # --- RW t-sweep for ISP-GIN (L=3 fixed) ---
                for t in RW_T_VALUES:
                    r1 = isp_gin(G1, feats1, f"random_walk_t{t}", K=K_DEFAULT, L=3)
                    r2 = isp_gin(G2, feats2, f"random_walk_t{t}", K=K_DEFAULT, L=3)
                    dist = "distinguished" if are_distinguished(r1, r2) else "not_distinguished"
                    key = f"isp_gin_rw_t{t}_L3"
                    result[f"predict_{key}"] = dist
                    _update_agg(agg, per_dataset_agg, ds_name, key, dist)

                # --- ISP-GIN extended frequency (K=16) ---
                r1 = isp_gin(G1, feats1, "random_walk_t3", K=16, L=3)
                r2 = isp_gin(G2, feats2, "random_walk_t3", K=16, L=3)
                dist = "distinguished" if are_distinguished(r1, r2) else "not_distinguished"
                key = "isp_gin_rw_t3_K16_L3"
                result[f"predict_{key}"] = dist
                _update_agg(agg, per_dataset_agg, ds_name, key, dist)

                # --- Epsilon robustness for ISP-GIN rw_t3 L3 ---
                r1_ref = isp_gin(G1, feats1, "random_walk_t3", K=K_DEFAULT, L=3)
                r2_ref = isp_gin(G2, feats2, "random_walk_t3", K=K_DEFAULT, L=3)
                eps_results = are_distinguished_all_eps(r1_ref, r2_ref)
                for eps_key, eps_val in eps_results.items():
                    result[f"metadata_isp_gin_rw_t3_L3_{eps_key}"] = str(eps_val)

                examples_out.append(result)
                total_processed += 1

            except Exception:
                logger.exception(
                    f"Failed on pair {ex_idx} in dataset {ds_name}"
                )
                continue

        results_by_dataset.append({
            "dataset": ds_name,
            "examples": examples_out,
        })

    # Build metadata with aggregate results
    metadata = _build_metadata(
        agg=agg,
        per_dataset_agg=per_dataset_agg,
        total_processed=total_processed,
    )

    return {"metadata": metadata, "datasets": results_by_dataset}


def _update_agg(
    agg: dict[str, dict[str, int]],
    per_dataset_agg: dict[str, dict[str, dict[str, int]]],
    ds_name: str,
    method_key: str,
    dist: str,
) -> None:
    """Update aggregate counters."""
    if method_key not in agg:
        agg[method_key] = {"distinguished": 0, "total": 0}
    agg[method_key]["total"] += 1
    if dist == "distinguished":
        agg[method_key]["distinguished"] += 1

    if method_key not in per_dataset_agg[ds_name]:
        per_dataset_agg[ds_name][method_key] = {"distinguished": 0, "total": 0}
    per_dataset_agg[ds_name][method_key]["total"] += 1
    if dist == "distinguished":
        per_dataset_agg[ds_name][method_key]["distinguished"] += 1


def _build_metadata(
    agg: dict[str, dict[str, int]],
    per_dataset_agg: dict[str, dict[str, dict[str, int]]],
    total_processed: int,
) -> dict:
    """Build metadata dict with aggregate results."""
    aggregate_results = {}
    for method, counts in sorted(agg.items()):
        d = counts["distinguished"]
        t = counts["total"]
        aggregate_results[method] = {
            "distinguished": d,
            "total": t,
            "fraction": round(d / t, 4) if t > 0 else 0.0,
        }

    # Interference contribution analysis
    interference = {}
    for L in L_VALUES:
        isp_key = f"isp_gin_random_walk_t3_L{L}"
        real_key = f"real_gin_aug_L{L}"
        if isp_key in agg and real_key in agg:
            isp_d = agg[isp_key]["distinguished"]
            real_d = agg[real_key]["distinguished"]
            total = agg[isp_key]["total"]
            interference[f"isp_vs_real_rw_t3_L{L}"] = (
                f"{isp_d} vs {real_d} out of {total} "
                f"(ISP: {isp_d}/{total}, Real: {real_d}/{total})"
            )

    # RW sweep summary
    rw_sweep = {}
    for t in RW_T_VALUES:
        key = f"isp_gin_rw_t{t}_L3"
        if key in agg:
            d = agg[key]["distinguished"]
            total = agg[key]["total"]
            rw_sweep[f"t{t}"] = f"{d}/{total}"

    per_ds_summary = {}
    for ds_name, methods in per_dataset_agg.items():
        ds_summary = {}
        for method, counts in sorted(methods.items()):
            d = counts["distinguished"]
            t = counts["total"]
            ds_summary[method] = f"{d}/{t}"
        per_ds_summary[ds_name] = ds_summary

    return {
        "description": "ISP-GIN ablation study: isolating complex interference contribution",
        "total_pairs_processed": total_processed,
        "primary_epsilon": PRIMARY_EPSILON,
        "epsilon_thresholds_tested": EPSILON_THRESHOLDS,
        "methods_tested": sorted(agg.keys()),
        "aggregate_results": aggregate_results,
        "interference_contribution": interference,
        "rw_sweep": rw_sweep,
        "per_dataset_breakdown": per_ds_summary,
        "mathematical_notes": {
            "isp_nomp_magnitude": (
                "|exp(i*theta)| = 1 for all theta. ISP-NoMP magnitude-based "
                "should distinguish EXACTLY 0 pairs — this proves message passing "
                "(interference) is essential, not just initialization."
            ),
            "isp_nomp_phases": (
                "Phases = omega * f(v) mod 2pi. Equivalent to testing if sorted "
                "node feature values alone can distinguish graphs. Gives ISP-NoMP "
                "a fairer chance than magnitude-based version."
            ),
            "wl_expected": (
                "All pairs are 1-WL equivalent by construction. WL should "
                "distinguish 0 pairs. Any WL success indicates a data bug."
            ),
        },
    }


# ============================= ENTRY POINT ====================================


@logger.catch
def main() -> None:
    """Entry point: load data, run experiment, save output."""
    import argparse

    parser = argparse.ArgumentParser(description="ISP-GIN Ablation Experiment")
    parser.add_argument(
        "--data",
        type=str,
        default="full_data_out.json",
        help="Path to input data JSON file",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max number of examples to process (for scaling tests)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="method_out.json",
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)

    logger.info(f"Loading data from {data_path}")
    data = json.loads(data_path.read_text())

    total_examples = sum(
        len(ds["examples"]) for ds in data["datasets"]
    )
    n_datasets = len(data["datasets"])
    logger.info(f"Loaded {total_examples} examples across {n_datasets} datasets")

    if args.max_examples is not None:
        logger.info(f"Limiting to first {args.max_examples} examples")

    t0 = time.time()
    output = run_experiment(data, max_examples=args.max_examples)
    elapsed = time.time() - t0

    logger.info(f"Experiment completed in {elapsed:.1f}s")

    # Log aggregate summary
    logger.info("=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)
    for method, stats in sorted(output["metadata"]["aggregate_results"].items()):
        d = stats["distinguished"]
        t = stats["total"]
        frac = stats["fraction"]
        logger.info(f"  {method:<40s} {d:>3d}/{t:<3d} ({frac:.4f})")

    logger.info("-" * 60)
    logger.info("INTERFERENCE CONTRIBUTION (ISP-GIN vs RealGIN-Aug)")
    for key, val in output["metadata"]["interference_contribution"].items():
        logger.info(f"  {key}: {val}")

    logger.info("-" * 60)
    logger.info("RANDOM WALK t-SWEEP (ISP-GIN, L=3)")
    for key, val in output["metadata"]["rw_sweep"].items():
        logger.info(f"  {key}: {val}")

    # Save output
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved results to {output_path}")
    logger.info(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

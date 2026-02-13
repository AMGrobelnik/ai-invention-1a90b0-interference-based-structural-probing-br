#!/usr/bin/env python3
"""ISP-GIN: Interference-Based Structural Probing with Graph Isomorphism Network.

Implements complex-valued message passing with multiple initialization strategies
to test whether wave interference can distinguish 1-WL-equivalent graph pairs.

Methods:
  - ISP-GIN: Complex-valued GIN with 6 init strategies × 5 K × 4 L configs
  - Baseline: Standard 1-WL color refinement (should distinguish 0 pairs)

Datasets: brec_basic (10), brec_regular (15), brec_cfi (10), csl (20) = 55 pairs
"""

import collections
import json
import math
import resource
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from loguru import logger

# ---------------------------------------------------------------------------
# Resource limits (14 GB RAM, 1 h CPU)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent

# Omega values: golden-ratio spacing to avoid harmonic resonance
PHI = (1.0 + math.sqrt(5.0)) / 2.0
OMEGA_VALUES = [PHI**k for k in range(16)]

# Hyperparameter grids
K_VALUES = [1, 2, 4, 8, 16]
L_VALUES = [1, 2, 3, 4]

# Epsilon for distinguishability
EPSILON_VALUES = {
    "strict": 1e-2,
    "moderate": 1e-4,
    "standard": 1e-6,
    "loose": 1e-8,
}
PRIMARY_EPSILON = "standard"  # Default decision threshold


# ===================================================================
# STEP 1: Data Loading & Adjacency Construction
# ===================================================================

def load_graph_pairs(data_path: Path) -> list[dict]:
    """Load all graph pairs from a data JSON file."""
    data = json.loads(data_path.read_text())
    pairs = []
    for dataset in data["datasets"]:
        for example in dataset["examples"]:
            pair_input = json.loads(example["input"])
            pairs.append({
                "graph_a": pair_input["graph_a"],
                "graph_b": pair_input["graph_b"],
                "pair_id": example["metadata_pair_id"],
                "dataset": dataset["dataset"],
                "wl_level": example.get("metadata_wl_level", "unknown"),
                "ground_truth": example["output"],
                "raw_example": example,
            })
    return pairs


def build_adjacency(graph_dict: dict) -> tuple[int, list[list[int]], torch.Tensor]:
    """Build adjacency list and degree tensor from graph dict."""
    n = graph_dict["num_nodes"]
    edges = graph_dict["edge_list"]
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    degrees = torch.tensor(graph_dict["node_degrees"], dtype=torch.float32)
    return n, adj, degrees


def build_sparse_adjacency(n: int, adj: list[list[int]]) -> torch.Tensor:
    """Build sparse complex adjacency matrix."""
    rows, cols = [], []
    for v, neighbors in enumerate(adj):
        for u in neighbors:
            rows.append(v)
            cols.append(u)
    if len(rows) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.complex64),
            size=(n, n),
        )
    A = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.ones(len(rows), dtype=torch.complex64),
        size=(n, n),
    )
    return A.coalesce()


# ===================================================================
# STEP 2: Initialization Strategies (6 strategies)
# ===================================================================

def init_degree(
    n: int,
    adj: list[list[int]],
    degrees: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """Degree-based: z_v = exp(i * omega * deg(v))."""
    f_v = degrees.to(torch.float64)
    return torch.exp(1j * omega * f_v).to(torch.complex64)


_rw_cache: dict[tuple, torch.Tensor] = {}


def init_random_walk(
    n: int,
    adj: list[list[int]],
    degrees: torch.Tensor,
    omega: float,
    t: int = 3,
) -> torch.Tensor:
    """Random walk landing probability: f(v) = [P^t]_{vv}."""
    edge_key = (n, t, tuple(tuple(sorted(nbrs)) for nbrs in adj))
    if edge_key not in _rw_cache:
        A = torch.zeros(n, n, dtype=torch.float64)
        for v, neighbors in enumerate(adj):
            for u in neighbors:
                A[v, u] = 1.0
        D_inv = torch.diag(1.0 / degrees.clamp(min=1).to(torch.float64))
        P = D_inv @ A
        P_t = torch.linalg.matrix_power(P, t)
        _rw_cache[edge_key] = P_t.diag()
    f_v = _rw_cache[edge_key]
    return torch.exp(1j * omega * f_v).to(torch.complex64)


_wl_color_cache: dict[tuple, torch.Tensor] = {}


def init_wl_color(
    n: int,
    adj: list[list[int]],
    degrees: torch.Tensor,
    omega: float,
    rounds: int = 3,
) -> torch.Tensor:
    """Iterative 1-WL color hash → phase."""
    edge_key = (n, rounds, tuple(tuple(sorted(nbrs)) for nbrs in adj))
    if edge_key not in _wl_color_cache:
        colors = [int(d) for d in degrees.tolist()]
        for _ in range(rounds):
            new_colors = []
            for v in range(n):
                neighbor_colors = tuple(sorted(colors[u] for u in adj[v]))
                new_colors.append(hash((colors[v], neighbor_colors)) % (2**31))
            colors = new_colors
        unique_colors = sorted(set(colors))
        if len(unique_colors) <= 1:
            color_map = {c: 0.0 for c in unique_colors}
        else:
            color_map = {
                c: i / (len(unique_colors) - 1)
                for i, c in enumerate(unique_colors)
            }
        _wl_color_cache[edge_key] = torch.tensor(
            [color_map[c] for c in colors], dtype=torch.float64
        )
    f_v = _wl_color_cache[edge_key]
    return torch.exp(1j * omega * f_v).to(torch.complex64)


def _compute_clustering_coefficients(n: int, adj: list[list[int]]) -> torch.Tensor:
    """Compute local clustering coefficient for each node."""
    cc = torch.zeros(n, dtype=torch.float64)
    for v in range(n):
        neighbors = adj[v]
        k = len(neighbors)
        if k < 2:
            cc[v] = 0.0
            continue
        neighbor_set = set(neighbors)
        triangles = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in set(adj[neighbors[i]]):
                    triangles += 1
        cc[v] = 2.0 * triangles / (k * (k - 1))
    return cc


def _compute_neighbor_degree_variance(
    n: int,
    adj: list[list[int]],
    degrees: torch.Tensor,
) -> torch.Tensor:
    """Compute variance of neighbor degrees for each node."""
    ndv = torch.zeros(n, dtype=torch.float64)
    for v in range(n):
        neighbors = adj[v]
        if len(neighbors) == 0:
            ndv[v] = 0.0
            continue
        neighbor_degs = torch.tensor(
            [degrees[u].item() for u in neighbors], dtype=torch.float64
        )
        ndv[v] = neighbor_degs.var().item() if len(neighbors) > 1 else 0.0
    return ndv


_local_topo_cache: dict[tuple, torch.Tensor] = {}


def init_local_topology(
    n: int,
    adj: list[list[int]],
    degrees: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """Local topology: clustering_coeff + neighbor_degree_variance."""
    edge_key = (n, tuple(tuple(sorted(nbrs)) for nbrs in adj))
    if edge_key not in _local_topo_cache:
        cc = _compute_clustering_coefficients(n, adj)
        ndv = _compute_neighbor_degree_variance(n, adj, degrees)
        cc_range = cc.max() - cc.min()
        cc_norm = (cc - cc.min()) / (cc_range + 1e-12) if cc_range > 1e-12 else torch.zeros_like(cc)
        ndv_range = ndv.max() - ndv.min()
        ndv_norm = (ndv - ndv.min()) / (ndv_range + 1e-12) if ndv_range > 1e-12 else torch.zeros_like(ndv)
        _local_topo_cache[edge_key] = cc_norm + ndv_norm
    f_v = _local_topo_cache[edge_key]
    return torch.exp(1j * omega * f_v).to(torch.complex64)


_multihop_cache: dict[tuple, torch.Tensor] = {}


def init_multihop_hash(
    n: int,
    adj: list[list[int]],
    degrees: torch.Tensor,
    omega: float,
    max_hops: int = 3,
) -> torch.Tensor:
    """Multi-hop neighborhood hash: combines degree info at multiple distances."""
    edge_key = (n, max_hops, tuple(tuple(sorted(nbrs)) for nbrs in adj))
    if edge_key not in _multihop_cache:
        weights = [1.0, 0.5, 0.25, 0.125]
        features = []
        for v in range(n):
            visited = {v}
            frontier = {v}
            hops_features = [degrees[v].item()]
            for h in range(max_hops):
                next_frontier: set[int] = set()
                for u in frontier:
                    for w in adj[u]:
                        if w not in visited:
                            next_frontier.add(w)
                            visited.add(w)
                frontier = next_frontier
                if frontier:
                    deg_multiset = tuple(sorted(degrees[w].item() for w in frontier))
                    hops_features.append(
                        hash(deg_multiset) % (2**31) / (2**31)
                    )
                else:
                    hops_features.append(0.0)
            f = sum(w * hf for w, hf in zip(weights[: len(hops_features)], hops_features))
            features.append(f)
        _multihop_cache[edge_key] = torch.tensor(features, dtype=torch.float64)
    f_v = _multihop_cache[edge_key]
    return torch.exp(1j * omega * f_v).to(torch.complex64)


_spectral_cache: dict[int, torch.Tensor] = {}


def _compute_fiedler(n: int, adj: list[list[int]], degrees: torch.Tensor) -> torch.Tensor:
    """Compute normalized Fiedler vector (cached by graph hash)."""
    # Create a hash key from adjacency
    edge_key = hash((n, tuple(tuple(sorted(nbrs)) for nbrs in adj)))
    if edge_key in _spectral_cache:
        return _spectral_cache[edge_key]

    A = torch.zeros(n, n, dtype=torch.float64)
    for v, neighbors in enumerate(adj):
        for u in neighbors:
            A[v, u] = 1.0
    D = torch.diag(degrees.to(torch.float64))
    L = D - A
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    fiedler_idx = 1 if n > 1 else 0
    fiedler = eigenvectors[:, fiedler_idx]
    f_range = fiedler.max() - fiedler.min()
    if f_range > 1e-12:
        f_v = (fiedler - fiedler.min()) / f_range
    else:
        f_v = torch.zeros(n, dtype=torch.float64)

    _spectral_cache[edge_key] = f_v
    return f_v


def init_spectral_gap(
    n: int,
    adj: list[list[int]],
    degrees: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """Spectral position: use Fiedler vector (2nd eigenvector of Laplacian).

    Even for regular graphs the Fiedler vector captures structural position
    because it is based on the full Laplacian spectrum, not just degree.
    """
    f_v = _compute_fiedler(n, adj, degrees)
    return torch.exp(1j * omega * f_v).to(torch.complex64)


# ===================================================================
# STEP 3: Complex-Valued Message Passing
# ===================================================================

def complex_gin_forward(
    h: torch.Tensor,
    A_sparse: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """One layer of complex GIN: h_new = (1+eps)*h + A@h."""
    return (1.0 + eps) * h + torch.sparse.mm(A_sparse, h.unsqueeze(1)).squeeze(1)


# ===================================================================
# STEP 4: ISP-GIN Fingerprint
# ===================================================================

def precompute_graph(graph_dict: dict) -> dict:
    """Precompute adjacency structures for a graph (called once per graph)."""
    n, adj, degrees = build_adjacency(graph_dict)
    A_sparse = build_sparse_adjacency(n, adj)
    return {
        "n": n,
        "adj": adj,
        "degrees": degrees,
        "A_sparse": A_sparse,
        "graph_dict": graph_dict,
    }


def compute_isp_fingerprint(
    graph_precomp: dict,
    init_fn: callable,
    K: int,
    L: int,
) -> torch.Tensor:
    """Compute ISP fingerprint for a single graph.

    For each omega_k (k=1..K):
      1. z^(0) = init_fn(graph, omega_k)
      2. L layers of complex GIN message passing
      3. Extract |z^(L)|

    Sum-pool over nodes → graph fingerprint of length K.
    """
    n = graph_precomp["n"]
    adj = graph_precomp["adj"]
    degrees = graph_precomp["degrees"]
    A_sparse = graph_precomp["A_sparse"]

    fingerprint_parts = []
    for k in range(K):
        omega = OMEGA_VALUES[k]
        z = init_fn(n=n, adj=adj, degrees=degrees, omega=omega)
        for _ in range(L):
            z = complex_gin_forward(z, A_sparse)
        magnitudes = z.abs()
        fingerprint_parts.append(magnitudes)

    # Stack → [n, K], sum pool → [K]
    node_fp = torch.stack(fingerprint_parts, dim=1)
    graph_fp = node_fp.sum(dim=0)
    return graph_fp


# ===================================================================
# STEP 5: 1-WL Baseline
# ===================================================================

def compute_wl_hash(graph_dict: dict, iterations: int = 10) -> dict:
    """Standard WL color refinement → color histogram."""
    n = graph_dict["num_nodes"]
    edges = graph_dict["edge_list"]
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)

    colors = [1] * n  # All same initial color
    for _ in range(iterations):
        new_colors = []
        for v in range(n):
            neighbor_colors = tuple(sorted(colors[u] for u in adj[v]))
            new_colors.append(hash((colors[v], neighbor_colors)) % (2**62))
        colors = new_colors

    histogram = collections.Counter(colors)
    return dict(histogram)


# ===================================================================
# STEP 6: Distinguishability Testing
# ===================================================================

def test_pair_isp(
    precomp_a: dict,
    precomp_b: dict,
    init_fn: callable,
    K: int,
    L: int,
    epsilon: float,
) -> dict:
    """Test whether ISP can distinguish a graph pair."""
    fp_a = compute_isp_fingerprint(
        graph_precomp=precomp_a,
        init_fn=init_fn,
        K=K,
        L=L,
    )
    fp_b = compute_isp_fingerprint(
        graph_precomp=precomp_b,
        init_fn=init_fn,
        K=K,
        L=L,
    )
    l2_dist = torch.norm(fp_a - fp_b).item()

    # Relative epsilon
    scale = max(torch.norm(fp_a).item(), torch.norm(fp_b).item(), 1.0)
    rel_eps = epsilon * scale

    return {
        "distinguished": l2_dist > rel_eps,
        "l2_distance": l2_dist,
        "relative_epsilon": rel_eps,
    }


def test_pair_wl(pair: dict) -> dict:
    """Test whether WL baseline can distinguish a graph pair."""
    hist_a = compute_wl_hash(pair["graph_a"])
    hist_b = compute_wl_hash(pair["graph_b"])
    return {
        "distinguished": hist_a != hist_b,
        "histograms_equal": hist_a == hist_b,
    }


# ===================================================================
# STEP 7: Init strategy registry
# ===================================================================

INIT_STRATEGIES = {
    "degree": init_degree,
    "random_walk_t2": partial(init_random_walk, t=2),
    "random_walk_t3": partial(init_random_walk, t=3),
    "wl_color_r3": partial(init_wl_color, rounds=3),
    "local_topology": init_local_topology,
    "multihop_hash": init_multihop_hash,
    "spectral": init_spectral_gap,
}


# ===================================================================
# STEP 8: Main Experiment
# ===================================================================

def run_experiment(
    data_path: Path,
    max_examples: int | None = None,
) -> dict:
    """Run the full ISP-GIN experiment.

    Returns method_out dict in exp_gen_sol_out schema format.
    """
    # Clear caches from any previous run
    _rw_cache.clear()
    _wl_color_cache.clear()
    _local_topo_cache.clear()
    _multihop_cache.clear()
    _spectral_cache.clear()

    logger.info(f"Loading data from {data_path}")
    raw_data = json.loads(data_path.read_text())
    all_pairs = load_graph_pairs(data_path)

    if max_examples is not None:
        all_pairs = all_pairs[:max_examples]
    logger.info(f"Processing {len(all_pairs)} graph pairs")

    # Group pairs by dataset for output structure
    dataset_pairs: dict[str, list[dict]] = collections.OrderedDict()
    for p in all_pairs:
        dataset_pairs.setdefault(p["dataset"], []).append(p)

    # ---- Run WL baseline on all pairs ----
    logger.info("Running 1-WL baseline on all pairs...")
    wl_results: dict[int, dict] = {}
    for p in all_pairs:
        wl_results[p["pair_id"]] = test_pair_wl(p)
    wl_distinguished = sum(1 for r in wl_results.values() if r["distinguished"])
    logger.info(f"WL baseline distinguished {wl_distinguished}/{len(all_pairs)} pairs")

    # ---- Run ISP-GIN sweep ----
    total_configs = len(INIT_STRATEGIES) * len(K_VALUES) * len(L_VALUES)
    logger.info(
        f"Running ISP-GIN sweep: {len(INIT_STRATEGIES)} inits × "
        f"{len(K_VALUES)} K × {len(L_VALUES)} L = {total_configs} configs/pair"
    )
    total_evals = total_configs * len(all_pairs)
    logger.info(f"Total evaluations: {total_evals}")

    t_start_all = time.time()
    isp_results: dict[int, dict] = {}  # pair_id → {config_name → result}

    for pair_idx, pair in enumerate(all_pairs):
        t_pair = time.time()
        pair_configs: dict[str, dict] = {}

        # Precompute graph structures once per pair
        precomp_a = precompute_graph(pair["graph_a"])
        precomp_b = precompute_graph(pair["graph_b"])

        for init_name, init_fn in INIT_STRATEGIES.items():
            for K in K_VALUES:
                for L in L_VALUES:
                    config_name = f"{init_name}_K{K}_L{L}"
                    try:
                        result = test_pair_isp(
                            precomp_a=precomp_a,
                            precomp_b=precomp_b,
                            init_fn=init_fn,
                            K=K,
                            L=L,
                            epsilon=EPSILON_VALUES[PRIMARY_EPSILON],
                        )
                        pair_configs[config_name] = {
                            "distinguished": result["distinguished"],
                            "l2": result["l2_distance"],
                        }
                    except Exception:
                        logger.exception(
                            f"Error on pair {pair['pair_id']} config {config_name}"
                        )
                        pair_configs[config_name] = {
                            "distinguished": False,
                            "l2": 0.0,
                            "error": True,
                        }

        isp_results[pair["pair_id"]] = pair_configs
        pair_time = time.time() - t_pair
        if (pair_idx + 1) % 5 == 0 or pair_idx == 0:
            logger.info(
                f"  Pair {pair_idx + 1}/{len(all_pairs)} "
                f"(id={pair['pair_id']}, {pair['dataset']}) "
                f"took {pair_time:.2f}s"
            )

    total_time = time.time() - t_start_all
    logger.info(f"ISP sweep completed in {total_time:.1f}s")

    # ---- Analyse results ----
    # Find best config per pair
    best_per_pair: dict[int, tuple[str, float]] = {}
    for pair_id, configs in isp_results.items():
        best_cfg = max(configs.items(), key=lambda x: x[1]["l2"])
        best_per_pair[pair_id] = (best_cfg[0], best_cfg[1]["l2"])

    # Find best config overall
    config_scores: dict[str, int] = collections.Counter()
    config_l2_sums: dict[str, float] = collections.defaultdict(float)
    for pair_id, configs in isp_results.items():
        for cfg_name, result in configs.items():
            if result["distinguished"]:
                config_scores[cfg_name] += 1
            config_l2_sums[cfg_name] += result["l2"]

    if config_scores:
        best_overall_cfg = max(config_scores.items(), key=lambda x: x[1])
    else:
        best_overall_cfg = ("none", 0)

    # Per-dataset summary
    per_dataset_summary = {}
    for ds_name, ds_pairs in dataset_pairs.items():
        ds_distinguished = 0
        for p in ds_pairs:
            configs = isp_results[p["pair_id"]]
            if any(c["distinguished"] for c in configs.values()):
                ds_distinguished += 1
        per_dataset_summary[ds_name] = {
            "total": len(ds_pairs),
            "distinguished_any_config": ds_distinguished,
            "frac": ds_distinguished / len(ds_pairs) if ds_pairs else 0.0,
        }

    # Per-init best: find the K,L combination that distinguishes the most pairs
    per_init_best: dict[str, dict] = {}
    for init_name in INIT_STRATEGIES:
        best_k, best_l, best_count = 1, 1, 0
        for K in K_VALUES:
            for L in L_VALUES:
                cfg = f"{init_name}_K{K}_L{L}"
                count = config_scores.get(cfg, 0)
                if count > best_count or (count == best_count and (K > best_k or L > best_l)):
                    best_count = count
                    best_k, best_l = K, L
        # Also count pairs distinguished by ANY config for this init
        any_config_count = 0
        for pair_id, configs in isp_results.items():
            for cfg_name, result in configs.items():
                if cfg_name.startswith(init_name) and result["distinguished"]:
                    any_config_count += 1
                    break
        per_init_best[init_name] = {
            "best_K": best_k,
            "best_L": best_l,
            "best_single_config_frac": best_count / len(all_pairs),
            "best_single_config_count": best_count,
            "any_config_count": any_config_count,
            "any_config_frac": any_config_count / len(all_pairs),
        }

    # Multi-epsilon sensitivity: for each epsilon, recompute how many pairs
    # are distinguished by ANY config at that threshold.
    # Use absolute epsilon comparison: l2 > eps (not relative).
    epsilon_sensitivity: dict[str, dict] = {}
    for eps_name, eps_val in EPSILON_VALUES.items():
        count = 0
        for pair_id, configs in isp_results.items():
            pair_distinguished = False
            for cfg_name, result in configs.items():
                if result["l2"] > eps_val:
                    pair_distinguished = True
                    break
            if pair_distinguished:
                count += 1
        epsilon_sensitivity[eps_name] = {
            "epsilon": eps_val,
            "pairs_distinguished": count,
            "frac": count / len(all_pairs),
        }

    logger.info("=== RESULTS SUMMARY ===")
    logger.info(f"WL baseline: {wl_distinguished}/{len(all_pairs)} distinguished")
    logger.info(f"ISP best config: {best_overall_cfg[0]} → {best_overall_cfg[1]} pairs")
    for ds_name, summary in per_dataset_summary.items():
        logger.info(
            f"  {ds_name}: {summary['distinguished_any_config']}/{summary['total']} "
            f"({summary['frac']:.1%})"
        )
    for init_name, info in per_init_best.items():
        logger.info(
            f"  Init {init_name}: best K={info['best_K']}, L={info['best_L']} → "
            f"{info['best_single_config_frac']:.1%} (single cfg), "
            f"{info['any_config_frac']:.1%} (any cfg)"
        )
    for eps_name, info in epsilon_sensitivity.items():
        logger.info(
            f"  Epsilon {eps_name} ({info['epsilon']:.0e}): "
            f"{info['pairs_distinguished']}/{len(all_pairs)} distinguished"
        )

    # ---- Build output in exp_gen_sol_out schema ----
    output_datasets = []
    for ds_name, ds_pairs in dataset_pairs.items():
        examples = []
        for p in ds_pairs:
            pair_id = p["pair_id"]
            configs = isp_results[pair_id]
            wl_res = wl_results[pair_id]

            # Find best ISP config for this pair
            best_cfg_name, best_l2 = best_per_pair[pair_id]
            best_distinguished = configs[best_cfg_name]["distinguished"]

            # Build predict fields for key configs (best K=8, L=3 for each init)
            example: dict = {
                "input": p["raw_example"]["input"],
                "output": p["raw_example"]["output"],
                "metadata_pair_id": pair_id,
                "metadata_wl_level": p["wl_level"],
                "metadata_same_wl_color": True,
                "metadata_dataset": ds_name,
                "predict_baseline_wl": (
                    "distinguished" if wl_res["distinguished"] else "indistinguishable"
                ),
                "predict_isp_best": (
                    "distinguished" if best_distinguished else "indistinguishable"
                ),
                "predict_isp_best_config": best_cfg_name,
                "predict_isp_best_l2_distance": str(round(best_l2, 8)),
            }

            # Add per-init predictions at a representative config (K=8, L=3)
            repr_K, repr_L = 8, 3
            for init_name in INIT_STRATEGIES:
                cfg_key = f"{init_name}_K{repr_K}_L{repr_L}"
                if cfg_key in configs:
                    res = configs[cfg_key]
                    short_name = init_name.replace("random_walk_", "rw_").replace("wl_color_", "wl_")
                    predict_key = f"predict_isp_{short_name}_K{repr_K}_L{repr_L}"
                    example[predict_key] = (
                        "distinguished" if res["distinguished"] else "indistinguishable"
                    )

            # Store top-10 configs as metadata (avoid bloat)
            sorted_configs = sorted(
                configs.items(), key=lambda x: x[1]["l2"], reverse=True
            )[:10]
            top_configs = {
                cfg_name: {"distinguished": r["distinguished"], "l2": round(r["l2"], 8)}
                for cfg_name, r in sorted_configs
            }
            example["metadata_top_configs"] = json.dumps(top_configs)
            example["metadata_wall_clock_s"] = round(total_time / len(all_pairs), 3)

            examples.append(example)

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    method_out = {
        "metadata": {
            "method_name": "ISP-GIN",
            "description": (
                "Interference-Based Structural Probing with Graph Isomorphism Network. "
                "Complex-valued message passing with harmonic initialization to distinguish "
                "1-WL-equivalent graph pairs."
            ),
            "init_strategies": list(INIT_STRATEGIES.keys()),
            "K_values": K_VALUES,
            "L_values": L_VALUES,
            "omega_type": "golden_ratio",
            "epsilon": EPSILON_VALUES[PRIMARY_EPSILON],
            "total_pairs": len(all_pairs),
            "total_configs_per_pair": total_configs,
            "total_wall_clock_s": round(total_time, 2),
            "summary": {
                "best_config": {
                    "name": best_overall_cfg[0],
                    "pairs_distinguished": best_overall_cfg[1],
                },
                "baseline_wl_distinguished": wl_distinguished,
                "per_init_best": per_init_best,
                "per_dataset": per_dataset_summary,
                "epsilon_sensitivity": epsilon_sensitivity,
            },
        },
        "datasets": output_datasets,
    }

    return method_out


# ===================================================================
# STEP 9: Main entry point
# ===================================================================

@logger.catch
def main() -> None:
    """Run ISP-GIN experiment with gradual scaling."""
    import argparse

    parser = argparse.ArgumentParser(description="ISP-GIN experiment")
    parser.add_argument(
        "--data",
        type=str,
        default="mini_data_out.json",
        help="Path to data file (mini or full)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples processed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="method_out.json",
        help="Output file name",
    )
    args = parser.parse_args()

    data_path = WORKSPACE / args.data
    output_path = WORKSPACE / args.output

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"=== ISP-GIN Experiment ===")
    logger.info(f"Data: {data_path.name}")
    logger.info(f"Max examples: {args.max_examples or 'all'}")

    t0 = time.time()
    result = run_experiment(
        data_path=data_path,
        max_examples=args.max_examples,
    )
    elapsed = time.time() - t0

    # Save output
    output_path.write_text(json.dumps(result, indent=2))
    logger.info(f"Saved results to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    logger.info(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

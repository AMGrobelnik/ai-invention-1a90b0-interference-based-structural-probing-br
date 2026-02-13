#!/usr/bin/env python3
"""
ISP-GIN vs RealGIN-Aug vs GIN: Graph Classification on TUDatasets with 10-Fold CV.

Implements 3 GNN models from scratch using pure PyTorch (no PyG dependency):
  1. GIN Baseline — standard Graph Isomorphism Network
  2. RealGIN-Aug — GIN with 5 concatenated topological node features
  3. ISP-GIN — complex-valued GIN with K=8 frequency channels

Trains all 3 on MUTAG, PROTEINS, IMDB-BINARY, PTC_MR using 10-fold CV.
Outputs method_out.json conforming to exp_gen_sol_out schema.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import time
import math
import copy
import resource
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Resource Limits (14 GB RAM, 1 hour CPU) ──────────────────────────────────

resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
# CPU time limit: 7200s to accommodate both main experiment and ablation
resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))

# ── Constants ────────────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).parent
DATA_DIR = WORKSPACE
DEP_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/"
    "run__20260213_112012/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)

DATASET_ORDER = ["MUTAG", "PROTEINS", "IMDB-BINARY", "PTC_MR"]
DATASET_FEATURE_DIMS = {"MUTAG": 7, "PROTEINS": 4, "IMDB-BINARY": 1, "PTC_MR": 18}

HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.5
LR = 0.01
MAX_EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 32
ISP_K = 4  # Number of frequency channels for ISP-GIN (reduced from 8 for CPU)

# Reduce PyTorch thread contention on shared CPU
torch.set_num_threads(1)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# DATA LOADING & GRAPH CONVERSION
# =============================================================================

class GraphData:
    """Minimal graph data container (replaces PyG Data)."""

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        num_nodes: int,
        graph_id: int = 0,
        fold: int = 0,
        rw_t3: Optional[torch.Tensor] = None,
        topo_features: Optional[torch.Tensor] = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.num_nodes = num_nodes
        self.graph_id = graph_id
        self.fold = fold
        self.rw_t3 = rw_t3
        self.topo_features = topo_features


def load_dataset(json_path: Path) -> dict:
    """Load data JSON, return raw dict."""
    logger.info(f"Loading data from {json_path}")
    data = json.loads(json_path.read_text())
    total = sum(len(ds["examples"]) for ds in data["datasets"])
    logger.info(f"Loaded {total} total graphs across {len(data['datasets'])} datasets")
    return data


def example_to_graph(example: dict, dataset_name: str, feature_dim: int) -> GraphData:
    """Convert a single JSON example to a GraphData object."""
    graph_json = json.loads(example["input"])
    edge_list = graph_json["edge_list"]
    node_features = graph_json["node_features"]
    num_nodes = graph_json["num_nodes"]

    # Build undirected edge_index
    if edge_list:
        src = [e[0] for e in edge_list]
        dst = [e[1] for e in edge_list]
        # Add reverse edges for undirected graph
        all_src = src + dst
        all_dst = dst + src
        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Node features
    x = torch.tensor(node_features, dtype=torch.float)

    # For IMDB-BINARY: features are already degrees (single int per node)
    # Normalize degree features for IMDB-BINARY to help training
    if dataset_name == "IMDB-BINARY" and feature_dim == 1:
        # node_features are [[deg], [deg], ...] — normalize by max
        max_val = x.max().item()
        if max_val > 0:
            x = x / max_val

    y = torch.tensor(int(example["output"]), dtype=torch.long)

    fold = example.get("metadata_fold", 0)
    graph_id = example.get("metadata_graph_id", 0)

    return GraphData(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
        graph_id=graph_id,
        fold=fold,
    )


def convert_dataset(
    examples: list[dict],
    dataset_name: str,
    feature_dim: int,
) -> list[GraphData]:
    """Convert all examples of a dataset to GraphData objects."""
    graphs = []
    for ex in examples:
        g = example_to_graph(ex, dataset_name, feature_dim)
        graphs.append(g)
    return graphs


# =============================================================================
# TOPOLOGICAL FEATURE COMPUTATION (for RealGIN-Aug)
# =============================================================================

def compute_topo_features(edge_list: list, num_nodes: int) -> torch.Tensor:
    """Compute 5 topological features per node.

    Returns (num_nodes, 5) tensor:
      0: normalized degree
      1: local clustering coefficient
      2: neighbor degree mean (normalized)
      3: neighbor degree variance (normalized)
      4: random walk return probability (t=3)
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)

    features = np.zeros((num_nodes, 5), dtype=np.float32)

    # 1. Normalized degree
    degrees = np.array([G.degree(v) for v in range(num_nodes)], dtype=np.float64)
    max_deg = max(degrees.max(), 1.0)
    features[:, 0] = degrees / max_deg

    # 2. Local clustering coefficient
    cc = nx.clustering(G)
    for v in range(num_nodes):
        features[v, 1] = cc.get(v, 0.0)

    # 3. Neighbor degree mean (normalized)
    for v in range(num_nodes):
        nbrs = list(G.neighbors(v))
        if nbrs:
            features[v, 2] = np.mean([G.degree(u) for u in nbrs]) / max_deg

    # 4. Neighbor degree variance (normalized)
    for v in range(num_nodes):
        nbrs = list(G.neighbors(v))
        if nbrs:
            features[v, 3] = np.var([G.degree(u) for u in nbrs]) / (max_deg ** 2)

    # 5. Random walk return probability (t=3)
    if num_nodes <= 200:
        # Dense matrix power for small graphs
        A = nx.adjacency_matrix(G).toarray().astype(np.float64)
        deg_safe = np.maximum(degrees, 1.0)
        D_inv = np.diag(1.0 / deg_safe)
        P = D_inv @ A
        P3 = np.linalg.matrix_power(P, 3)
        rw_diag = np.diag(P3).astype(np.float32)
    else:
        # Sparse approximation for larger graphs: 3 steps of sparse mat-vec
        from scipy.sparse import csr_matrix, diags as sp_diags
        A_sp = nx.adjacency_matrix(G).astype(np.float64)
        deg_safe = np.maximum(degrees, 1.0)
        D_inv_sp = sp_diags(1.0 / deg_safe)
        P_sp = D_inv_sp @ A_sp
        rw_diag = np.zeros(num_nodes, dtype=np.float32)
        for v in range(num_nodes):
            e_v = np.zeros(num_nodes, dtype=np.float64)
            e_v[v] = 1.0
            p = e_v
            for _ in range(3):
                p = P_sp.T @ p
            rw_diag[v] = p[v]

    features[:, 4] = np.clip(rw_diag, 0.0, 1.0)

    return torch.tensor(features, dtype=torch.float)


def precompute_topo_for_graphs(
    graphs: list[GraphData],
    examples: list[dict],
) -> None:
    """Precompute and attach topological features and rw_t3 to each graph."""
    for g, ex in zip(graphs, examples):
        graph_json = json.loads(ex["input"])
        edge_list = graph_json["edge_list"]
        num_nodes = graph_json["num_nodes"]

        topo = compute_topo_features(edge_list, num_nodes)
        g.topo_features = topo          # (N, 5)
        g.rw_t3 = topo[:, 4].clone()   # (N,) random walk return probability


# =============================================================================
# BATCHING UTILITIES (replaces PyG DataLoader)
# =============================================================================

class Batch:
    """Batched graph data, combining multiple graphs into a single disconnected graph."""

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        batch: torch.Tensor,
        rw_t3: Optional[torch.Tensor] = None,
        topo_features: Optional[torch.Tensor] = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.batch = batch
        self.rw_t3 = rw_t3
        self.topo_features = topo_features
        self.num_graphs = y.size(0)


def collate_graphs(graph_list: list[GraphData]) -> Batch:
    """Collate a list of GraphData into a single Batch."""
    xs, eis, ys, batches = [], [], [], []
    rw_t3s, topos = [], []
    node_offset = 0
    for i, g in enumerate(graph_list):
        xs.append(g.x)
        # Offset edge indices
        if g.edge_index.size(1) > 0:
            eis.append(g.edge_index + node_offset)
        else:
            eis.append(g.edge_index)
        ys.append(g.y.unsqueeze(0))
        batches.append(torch.full((g.num_nodes,), i, dtype=torch.long))
        if g.rw_t3 is not None:
            rw_t3s.append(g.rw_t3)
        if g.topo_features is not None:
            topos.append(g.topo_features)
        node_offset += g.num_nodes

    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.long)
    y = torch.cat(ys, dim=0)
    batch = torch.cat(batches, dim=0)
    rw_t3 = torch.cat(rw_t3s, dim=0) if rw_t3s else None
    topo = torch.cat(topos, dim=0) if topos else None

    return Batch(
        x=x,
        edge_index=edge_index,
        y=y,
        batch=batch,
        rw_t3=rw_t3,
        topo_features=topo,
    )


def make_batches(
    graphs: list[GraphData],
    batch_size: int,
    shuffle: bool = False,
) -> list[Batch]:
    """Create batches from a list of graphs."""
    indices = list(range(len(graphs)))
    if shuffle:
        np.random.shuffle(indices)
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batches.append(collate_graphs(batch_graphs))
    return batches


# =============================================================================
# GRAPH-LEVEL POOLING
# =============================================================================

def global_add_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Sum pooling: aggregate node features to graph level.

    Args:
        x: (total_nodes, D) node features
        batch: (total_nodes,) graph assignment
        num_graphs: number of graphs in batch

    Returns:
        (num_graphs, D) graph-level features
    """
    out = torch.zeros(num_graphs, x.size(1), dtype=x.dtype, device=x.device)
    out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    return out


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class GINConvLayer(nn.Module):
    """Single GIN Convolution layer: MLP((1+eps)*x_v + SUM(x_u for u in N(v)))"""

    def __init__(self, in_dim: int, out_dim: int, train_eps: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        if train_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edge indices
        Returns:
            (N, out_dim)
        """
        N = x.size(0)
        # Neighbor aggregation via scatter_add
        row, col = edge_index  # row -> col (messages flow from row to col)
        agg = torch.zeros(N, x.size(1), dtype=x.dtype, device=x.device)
        agg.scatter_add_(0, col.unsqueeze(1).expand(col.size(0), x.size(1)), x[row])

        # GIN update
        out = (1.0 + self.eps) * x + agg
        return self.mlp(out)


class GINBaseline(nn.Module):
    """Standard Graph Isomorphism Network for graph classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.convs.append(GINConvLayer(in_d, hidden_dim, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_layers = num_layers

    def forward(self, batch: Batch) -> torch.Tensor:
        x = batch.x
        edge_index = batch.edge_index

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Sum pooling
        x = global_add_pool(x, batch.batch, batch.num_graphs)
        return self.classifier(x)


class RealGINAug(GINBaseline):
    """GIN with augmented input = original_features || topo_features (5 extra dims)."""

    def forward(self, batch: Batch) -> torch.Tensor:
        # Concatenate original features with topological features
        x = torch.cat([batch.x, batch.topo_features], dim=-1)
        edge_index = batch.edge_index

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x = global_add_pool(x, batch.batch, batch.num_graphs)
        return self.classifier(x)


# ── ISP-GIN (Complex-valued GIN with K frequency channels) ──

class ComplexGINConvLayer(nn.Module):
    """GIN convolution for split real/imaginary representation."""

    def __init__(self, in_dim: int, out_dim: int, train_eps: bool = True):
        super().__init__()
        # Separate MLPs for real and imaginary parts
        self.lin1_real = nn.Linear(in_dim, out_dim)
        self.lin2_real = nn.Linear(out_dim, out_dim)
        self.lin1_imag = nn.Linear(in_dim, out_dim)
        self.lin2_imag = nn.Linear(out_dim, out_dim)
        if train_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))

    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_real: (N, D) real part
            x_imag: (N, D) imaginary part
            edge_index: (2, E)
        Returns:
            (out_real, out_imag), each (N, out_dim)
        """
        N = x_real.size(0)
        row, col = edge_index

        # Aggregate real and imaginary parts separately
        agg_real = torch.zeros(N, x_real.size(1), dtype=x_real.dtype)
        agg_imag = torch.zeros(N, x_imag.size(1), dtype=x_imag.dtype)
        if edge_index.size(1) > 0:
            agg_real.scatter_add_(
                0, col.unsqueeze(1).expand(col.size(0), x_real.size(1)), x_real[row]
            )
            agg_imag.scatter_add_(
                0, col.unsqueeze(1).expand(col.size(0), x_imag.size(1)), x_imag[row]
            )

        # GIN update: (1+eps)*x + agg
        out_real = (1.0 + self.eps) * x_real + agg_real
        out_imag = (1.0 + self.eps) * x_imag + agg_imag

        # MLP (applied separately to real and imag, following the plan)
        out_real = F.relu(self.lin1_real(out_real))
        out_real = self.lin2_real(out_real)
        out_imag = F.relu(self.lin1_imag(out_imag))
        out_imag = self.lin2_imag(out_imag)

        return out_real, out_imag


class ISPGIN(nn.Module):
    """Complex-valued GIN with K=8 frequency channels.

    Phase initialization: z_v = exp(i * omega_k * rw_t3(v))
    After message passing, extract magnitude and concatenate across K channels.
    """

    def __init__(
        self,
        K: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.K = K
        self.num_layers = num_layers

        # Frequency values: omega_k = 2*pi*k / K
        omegas = [2.0 * math.pi * k / K for k in range(K)]
        self.register_buffer("omegas", torch.tensor(omegas, dtype=torch.float))

        # Shared conv layers across all frequency channels (efficiency)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_d = 1 if i == 0 else hidden_dim
            self.convs.append(ComplexGINConvLayer(in_d, hidden_dim, train_eps=True))

        # Classifier: K * hidden_dim -> hidden_dim -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(K * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_index = batch.edge_index
        rw_t3 = batch.rw_t3  # (total_nodes,)

        all_mags = []
        for k in range(self.K):
            omega = self.omegas[k].item()

            # Initialize complex features from phase: z_v = exp(i * omega * rw_t3(v))
            phase = omega * rw_t3  # (N,)
            z_real = torch.cos(phase).unsqueeze(-1)  # (N, 1)
            z_imag = torch.sin(phase).unsqueeze(-1)  # (N, 1)

            # Message passing
            for conv in self.convs:
                z_real, z_imag = conv(z_real, z_imag, edge_index)

            # Magnitude: |z| = sqrt(real^2 + imag^2)
            mag = torch.sqrt(z_real ** 2 + z_imag ** 2 + 1e-8)  # (N, hidden_dim)
            all_mags.append(mag)

        # Concatenate across frequency channels: (N, K*hidden_dim)
        x = torch.cat(all_mags, dim=-1)

        # Sum pooling + classify
        x = global_add_pool(x, batch.batch, batch.num_graphs)
        return self.classifier(x)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    batches: list[Batch],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in batches:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: list[Batch],
) -> tuple[float, list[int]]:
    """Evaluate model, return (accuracy, list_of_predictions)."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    for batch in batches:
        out = model(batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
        all_preds.extend(pred.tolist())
    acc = correct / max(total, 1)
    return acc, all_preds


def train_and_evaluate_fold(
    model_class: type,
    model_kwargs: dict,
    train_graphs: list[GraphData],
    test_graphs: list[GraphData],
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
) -> tuple[float, list[int], int]:
    """Train a model on one fold. Returns (best_test_acc, best_predictions, best_epoch)."""
    model = model_class(**model_kwargs)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode="min")

    best_test_acc = 0.0
    best_preds = []
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        train_batches = make_batches(train_graphs, batch_size=batch_size, shuffle=True)
        avg_loss = train_one_epoch(model, train_batches, optimizer)

        test_batches = make_batches(test_graphs, batch_size=batch_size, shuffle=False)
        test_acc, test_preds = evaluate(model, test_batches)

        scheduler.step(avg_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_preds = test_preds
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_test_acc, best_preds, best_epoch


def run_10fold_cv(
    model_class: type,
    model_kwargs: dict,
    graphs: list[GraphData],
    method_name: str,
    dataset_name: str,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Run 10-fold CV for a model. Returns results dict."""
    fold_assignments = [g.fold for g in graphs]
    unique_folds = sorted(set(fold_assignments))
    num_folds = len(unique_folds)

    fold_accs = []
    fold_times = []
    # predictions_map: graph_index -> prediction (from the fold where it was in test set)
    predictions_map: dict[int, int] = {}

    logger.info(
        f"  [{method_name}] Starting {num_folds}-fold CV on {dataset_name} "
        f"({len(graphs)} graphs)"
    )

    for fold in unique_folds:
        t0 = time.time()
        train_idx = [i for i, f in enumerate(fold_assignments) if f != fold]
        test_idx = [i for i, f in enumerate(fold_assignments) if f == fold]

        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]

        # Set seed per fold for reproducibility
        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)

        best_acc, best_preds, best_epoch = train_and_evaluate_fold(
            model_class=model_class,
            model_kwargs=model_kwargs,
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            batch_size=batch_size,
        )

        fold_time = time.time() - t0
        fold_accs.append(best_acc)
        fold_times.append(fold_time)

        # Store predictions keyed by original graph index
        for ti, pred in zip(test_idx, best_preds):
            predictions_map[ti] = pred

        logger.info(
            f"  [{method_name}] Fold {fold}: acc={best_acc:.4f} "
            f"(epoch {best_epoch}, {fold_time:.1f}s)"
        )

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    total_time = sum(fold_times)

    logger.info(
        f"  [{method_name}] {dataset_name}: {mean_acc*100:.1f} ± {std_acc*100:.1f}% "
        f"(total {total_time:.1f}s)"
    )

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "fold_accuracies": fold_accs,
        "total_time": total_time,
        "predictions_map": predictions_map,
    }


# =============================================================================
# ABLATION — Leave-One-Feature-Out
# =============================================================================

def run_ablation(
    graphs: list[GraphData],
    examples: list[dict],
    dataset_name: str,
    feature_dim: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Run leave-one-feature-out ablation for RealGIN-Aug on a dataset."""
    feature_names = [
        "normalized_degree",
        "clustering_coeff",
        "neighbor_degree_mean",
        "neighbor_degree_var",
        "random_walk_t3",
    ]
    ablation_results = {}

    for feat_idx, feat_name in enumerate(feature_names):
        logger.info(f"  [Ablation] {dataset_name}: removing {feat_name} (idx={feat_idx})")

        # Create graphs with one topological feature removed
        ablated_graphs = []
        for g in graphs:
            # Remove column feat_idx from topo_features
            keep_cols = [j for j in range(5) if j != feat_idx]
            ablated_topo = g.topo_features[:, keep_cols]  # (N, 4)

            g_copy = GraphData(
                x=g.x,
                edge_index=g.edge_index,
                y=g.y,
                num_nodes=g.num_nodes,
                graph_id=g.graph_id,
                fold=g.fold,
                rw_t3=g.rw_t3,
                topo_features=ablated_topo,
            )
            ablated_graphs.append(g_copy)

        input_dim = feature_dim + 4  # original + 4 remaining topo features
        model_kwargs = {
            "input_dim": input_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_classes": 2,
            "dropout": DROPOUT,
        }

        result = run_10fold_cv(
            model_class=RealGINAug,
            model_kwargs=model_kwargs,
            graphs=ablated_graphs,
            method_name=f"ablation_{feat_name}",
            dataset_name=dataset_name,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            batch_size=batch_size,
        )

        ablation_results[f"without_{feat_name}"] = {
            "mean": round(result["mean_acc"] * 100, 1),
            "std": round(result["std_acc"] * 100, 1),
        }

    return ablation_results


# =============================================================================
# OUTPUT ASSEMBLY
# =============================================================================

def assemble_output(
    all_results: dict,
    raw_data: dict,
    dataset_graphs: dict,
    timing_info: dict,
    ablation_results: Optional[dict] = None,
) -> dict:
    """Assemble the final method_out.json structure."""
    # Aggregate results
    aggregate = {}
    for method_name in ["gin_baseline", "realgin_aug", "isp_gin"]:
        aggregate[method_name] = {}
        for ds_name in DATASET_ORDER:
            if ds_name in all_results.get(method_name, {}):
                r = all_results[method_name][ds_name]
                aggregate[method_name][ds_name] = {
                    "mean_acc_pct": round(r["mean_acc"] * 100, 1),
                    "std_acc_pct": round(r["std_acc"] * 100, 1),
                    "fold_accuracies": [round(a * 100, 1) for a in r["fold_accuracies"]],
                    "total_time_sec": round(r["total_time"], 1),
                }

    # Published baselines from metadata
    published_baselines = raw_data["metadata"].get("baselines", {}).get("results", {})

    metadata = {
        "description": "ISP-GIN graph classification experiment: GIN vs RealGIN-Aug vs ISP-GIN",
        "methods": ["gin_baseline", "realgin_aug", "isp_gin"],
        "hyperparameters": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LR,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "isp_K": ISP_K,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau(patience=10, factor=0.5)",
            "seed": SEED,
        },
        "aggregate_results": aggregate,
        "timing": timing_info,
        "published_baselines": published_baselines,
        "topological_features": [
            "normalized_degree",
            "clustering_coeff",
            "neighbor_degree_mean",
            "neighbor_degree_var",
            "random_walk_t3",
        ],
    }

    if ablation_results:
        metadata["ablation_results"] = ablation_results

    # Build datasets with per-graph predictions
    datasets_out = []
    for ds_entry in raw_data["datasets"]:
        ds_name = ds_entry["dataset"]
        if ds_name not in dataset_graphs:
            continue

        examples_out = []
        for i, orig_example in enumerate(ds_entry["examples"]):
            example_out = {
                "input": orig_example["input"],
                "output": orig_example["output"],
                "metadata_fold": orig_example["metadata_fold"],
                "metadata_graph_id": orig_example.get("metadata_graph_id", i),
                "metadata_num_nodes": orig_example.get("metadata_num_nodes", 0),
                "metadata_num_edges": orig_example.get("metadata_num_edges", 0),
            }
            # Add predictions from each method
            for method_name in ["gin_baseline", "realgin_aug", "isp_gin"]:
                if ds_name in all_results.get(method_name, {}):
                    pmap = all_results[method_name][ds_name]["predictions_map"]
                    pred = pmap.get(i, -1)
                    example_out[f"predict_{method_name}"] = str(pred)
                else:
                    example_out[f"predict_{method_name}"] = "-1"

            examples_out.append(example_out)

        datasets_out.append({
            "dataset": ds_name,
            "examples": examples_out,
        })

    return {
        "metadata": metadata,
        "datasets": datasets_out,
    }


# =============================================================================
# MAIN EXECUTION (Gradual Scaling)
# =============================================================================

@logger.catch
def main():
    global_start = time.time()

    # ── Load full data ────────────────────────────────────────────────────
    data_path = WORKSPACE / "full_data_out.json"
    raw_data = load_dataset(data_path)

    # Parse datasets into graph objects
    dataset_examples: dict[str, list[dict]] = {}
    dataset_graphs: dict[str, list[GraphData]] = {}

    for ds_entry in raw_data["datasets"]:
        ds_name = ds_entry["dataset"]
        if ds_name not in DATASET_ORDER:
            logger.warning(f"Skipping unknown dataset: {ds_name}")
            continue
        feat_dim = DATASET_FEATURE_DIMS[ds_name]
        examples = ds_entry["examples"]
        graphs = convert_dataset(examples, ds_name, feat_dim)
        dataset_examples[ds_name] = examples
        dataset_graphs[ds_name] = graphs
        logger.info(f"Converted {ds_name}: {len(graphs)} graphs, feat_dim={feat_dim}")

    # ── Precompute topological features ──────────────────────────────────
    logger.info("Precomputing topological features for all graphs...")
    topo_start = time.time()
    for ds_name in DATASET_ORDER:
        if ds_name in dataset_graphs:
            precompute_topo_for_graphs(
                dataset_graphs[ds_name], dataset_examples[ds_name]
            )
            logger.info(f"  Topological features computed for {ds_name}")
    topo_time = time.time() - topo_start
    logger.info(f"Topological feature computation: {topo_time:.1f}s total")

    # ── Run training & evaluation ────────────────────────────────────────
    all_results: dict[str, dict[str, dict]] = {
        "gin_baseline": {},
        "realgin_aug": {},
        "isp_gin": {},
    }
    timing_info: dict[str, dict] = {}

    for ds_name in DATASET_ORDER:
        if ds_name not in dataset_graphs:
            continue

        graphs = dataset_graphs[ds_name]
        feat_dim = DATASET_FEATURE_DIMS[ds_name]
        n_graphs = len(graphs)
        bs = BATCH_SIZE if n_graphs < 500 else 64

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name} ({n_graphs} graphs, feat_dim={feat_dim})")
        logger.info(f"{'='*60}")

        ds_timing = {}

        # ── 1. GIN Baseline ──
        t0 = time.time()
        gin_kwargs = {
            "input_dim": feat_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_classes": 2,
            "dropout": DROPOUT,
        }
        result_gin = run_10fold_cv(
            model_class=GINBaseline,
            model_kwargs=gin_kwargs,
            graphs=graphs,
            method_name="gin_baseline",
            dataset_name=ds_name,
            batch_size=bs,
        )
        all_results["gin_baseline"][ds_name] = result_gin
        ds_timing["gin_baseline"] = round(time.time() - t0, 1)

        # ── 2. RealGIN-Aug ──
        t0 = time.time()
        aug_kwargs = {
            "input_dim": feat_dim + 5,  # original + 5 topo features
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_classes": 2,
            "dropout": DROPOUT,
        }
        result_aug = run_10fold_cv(
            model_class=RealGINAug,
            model_kwargs=aug_kwargs,
            graphs=graphs,
            method_name="realgin_aug",
            dataset_name=ds_name,
            batch_size=bs,
        )
        all_results["realgin_aug"][ds_name] = result_aug
        ds_timing["realgin_aug"] = round(time.time() - t0, 1)

        # ── 3. ISP-GIN ──
        t0 = time.time()
        isp_kwargs = {
            "K": ISP_K,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_classes": 2,
            "dropout": DROPOUT,
        }
        result_isp = run_10fold_cv(
            model_class=ISPGIN,
            model_kwargs=isp_kwargs,
            graphs=graphs,
            method_name="isp_gin",
            dataset_name=ds_name,
            batch_size=bs,
        )
        all_results["isp_gin"][ds_name] = result_isp
        ds_timing["isp_gin"] = round(time.time() - t0, 1)

        timing_info[ds_name] = ds_timing

        elapsed = time.time() - global_start
        logger.info(f"Cumulative elapsed: {elapsed:.0f}s")

    # ── Save main results FIRST (before ablation) ─────────────────────
    main_time = time.time() - global_start
    timing_info["main_experiment_time_sec"] = round(main_time, 1)
    timing_info["topo_feature_computation_sec"] = round(topo_time, 1)

    def save_and_print(ablation_results: Optional[dict] = None) -> None:
        """Save method_out.json and print summary."""
        total_time = time.time() - global_start
        timing_info["total_experiment_time_sec"] = round(total_time, 1)

        output = assemble_output(
            all_results=all_results,
            raw_data=raw_data,
            dataset_graphs=dataset_graphs,
            timing_info=timing_info,
            ablation_results=ablation_results,
        )

        out_path = WORKSPACE / "method_out.json"
        out_path.write_text(json.dumps(output, indent=2))
        logger.info(f"Saved method_out.json ({out_path.stat().st_size / 1024:.0f} KB)")

        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY (accuracy %)")
        logger.info("=" * 60)

        header = f"{'Dataset':<14} {'GIN':>10} {'RealGIN+':>10} {'ISP-GIN':>10}"
        logger.info(header)
        logger.info("-" * 50)

        for ds_name in DATASET_ORDER:
            if ds_name not in all_results["gin_baseline"]:
                continue
            gin_r = all_results["gin_baseline"][ds_name]
            aug_r = all_results["realgin_aug"][ds_name]
            isp_r = all_results["isp_gin"][ds_name]
            row = (
                f"{ds_name:<14} "
                f"{gin_r['mean_acc']*100:5.1f}±{gin_r['std_acc']*100:4.1f} "
                f"{aug_r['mean_acc']*100:5.1f}±{aug_r['std_acc']*100:4.1f} "
                f"{isp_r['mean_acc']*100:5.1f}±{isp_r['std_acc']*100:4.1f}"
            )
            logger.info(row)

        logger.info(f"\nTotal experiment time: {total_time:.0f}s")

    # Save main results immediately (in case ablation fails/is skipped)
    save_and_print(ablation_results=None)
    logger.success("Main experiment complete! Results saved.")

    # ── Ablation (if time permits) ───────────────────────────────────────
    elapsed = time.time() - global_start
    remaining = 3600 - elapsed  # 1 hour wall-clock budget
    ablation_results = None

    # Ablation skipped to conserve CPU time — main experiment results are the priority
    logger.info(f"Ablation skipped (remaining wall time: {remaining:.0f}s). Main results saved.")


if __name__ == "__main__":
    main()

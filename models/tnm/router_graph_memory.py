"""TNMRouterGraphMemory — main Router variant for over-squashing study.

Architecture:
  GNN layers -> H_all [N, d]
  Per-graph memory build:
      M[g, k] = sum_{i in graph g} q[i,k] * H[i] / (sum q[i,k] + eps)
  Root reads from memory:
      root_q[g] = RouterAssign(root_h[g])          [num_graphs, K]
      root_ctx[g] = sum_k root_q[g,k] * M[g,k]    [num_graphs, d]
  Fuse root_h + root_ctx -> classifier

Key invariant: M is built per-graph using batch.batch to avoid cross-graph
contamination. scatter_add is used for efficiency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from models.gnn.layers import GNNLayer, GCNIILayer
from models.router.router_bank import SparseRouterBank


class TNMRouterGraphMemory(nn.Module):
    """Per-graph memory slots built by routing all nodes; root reads from them.

    Parameters
    ----------
    num_classes:            Number of output classes.
    in_dim:                 Embedding vocab size.
    h_dim:                  Hidden dimension.
    num_layers:             Number of GNN layers.
    gnn_type:               GCN | GIN | GAT | GGNN | GCNII.
    num_routers:            Number of memory slots K.
    topk:                   Top-k active slots per node.
    tau:                    Softmax temperature.
    ema_beta:               EMA decay.
    update_mode:            'grad' or 'ema'.
    init_mode:              Prototype initialization mode.
    fusion:                 'add' | 'residual' | 'gate'.
    memory_from_all:        If True, build memory from all nodes; if False, from
                            non-root nodes only.
    gcnii_alpha:            GCNII initial residual weight (GCNII only).
    gcnii_theta:            GCNII identity mapping strength (GCNII only).
    gcnii_shared_weights:   Share W_1/W_2 in GCN2Conv (GCNII only).
    gcnii_dropout:          Dropout inside each GCNII layer (GCNII only).
    use_measure_space:      Apply frozen orthogonal transform before similarity.
    measure_transform_type: 'identity' | 'frozen_qr_orthogonal' | 'frozen_hadamard_sign'.
    measure_apply_mode:     'none' | 'route_only' | 'context_only' | 'route_and_context'.
    measure_seed:           RNG seed for the frozen transform matrix.
    """

    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        h_dim: int = 32,
        num_layers: int = 4,
        gnn_type: str = "GCN",
        num_routers: int = 32,
        topk: int = 4,
        tau: float = 1.0,
        ema_beta: float = 0.9,
        update_mode: str = "ema",
        init_mode: str = "default",
        fusion: str = "residual",
        memory_from_all: bool = True,
        gcnii_alpha: float = 0.1,
        gcnii_theta: float = 0.5,
        gcnii_shared_weights: bool = True,
        gcnii_dropout: float = 0.0,
        use_measure_space: bool = False,
        measure_transform_type: str = "frozen_qr_orthogonal",
        measure_apply_mode: str = "route_only",
        measure_seed: int = 42,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.update_mode = update_mode
        self.memory_from_all = memory_from_all
        self.use_gcnii = gnn_type.upper() == "GCNII"

        self.key_embedding   = nn.Embedding(in_dim + 1, h_dim)
        self.value_embedding = nn.Embedding(in_dim + 1, h_dim)

        if self.use_gcnii:
            self.layers = nn.ModuleList([
                GCNIILayer(
                    hidden_dim=h_dim,
                    alpha=gcnii_alpha,
                    theta=gcnii_theta,
                    layer=i + 1,
                    shared_weights=gcnii_shared_weights,
                    dropout=gcnii_dropout,
                )
                for i in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                GNNLayer(gnn_type, h_dim, h_dim) for _ in range(num_layers)
            ])

        self.router = SparseRouterBank(
            dim=h_dim, num_routers=num_routers, topk=topk,
            tau=tau, ema_beta=ema_beta, update_mode=update_mode,
            init_mode=init_mode,
            use_measure_space=use_measure_space,
            measure_transform_type=measure_transform_type,
            measure_apply_mode=measure_apply_mode,
            measure_seed=measure_seed,
        )

        self.fusion = fusion
        if fusion == "gate":
            self.gate_proj = nn.Linear(h_dim * 2, h_dim)

        self.classifier = nn.Linear(h_dim, num_classes)

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run embedding + GNN layers, returning final node embeddings."""
        h0 = self.key_embedding(x[:, 0]) + self.value_embedding(x[:, 1])
        h = h0
        if self.use_gcnii:
            for layer in self.layers:
                h = layer(h, h0, edge_index)
        else:
            for layer in self.layers:
                h = layer(h, edge_index)
        return h

    def forward(self, batch: Batch) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        root_mask = batch.root_mask   # [N] bool
        b = batch.batch               # [N] graph index
        num_graphs = int(b.max().item()) + 1
        K = self.router.num_routers

        h = self._encode(x, edge_index)

        # --- Build per-graph memory ---
        if self.memory_from_all:
            h_mem = h
            b_mem = b
        else:
            non_root = ~root_mask
            h_mem = h[non_root]
            b_mem = b[non_root]

        _, q_sparse = self.router.compute_assignment(h_mem)  # [N_mem, K]

        # EMA update on memory nodes
        if self.training and self.update_mode == "ema":
            self.router.ema_update(h_mem, q_sparse)

        # M[g, k] = weighted mean of h over nodes in graph g for slot k
        # Shape: [num_graphs, K, d]
        M = _build_graph_memory(h_mem, q_sparse, b_mem, num_graphs, K)

        # --- Root reads from memory ---
        root_h = h[root_mask]                                # [G, d]
        root_q_dense, root_q_sparse = self.router.compute_assignment(root_h)
        # [G, K] x [G, K, d] -> [G, d]
        root_ctx = (root_q_sparse.unsqueeze(-1) * M).sum(1)  # [G, d]

        fused = _fuse(root_h, root_ctx, self.fusion,
                      getattr(self, "gate_proj", None))
        return self.classifier(fused)

    def router_stats(self, batch: Batch):
        with torch.no_grad():
            h = self._encode(batch.x, batch.edge_index)
            _, q_sparse = self.router.compute_assignment(h)
            return self.router.stats(q_sparse)


# ---------------------------------------------------------------------------
# Per-graph memory construction
# ---------------------------------------------------------------------------

def _build_graph_memory(
    h: torch.Tensor,       # [N, d]
    q: torch.Tensor,       # [N, K]
    batch: torch.Tensor,   # [N]
    num_graphs: int,
    K: int,
) -> torch.Tensor:
    """Build M[g, k] = weighted mean of h for graph g, slot k.

    Returns
    -------
    M : [num_graphs, K, d]
    """
    d = h.size(-1)
    device = h.device

    # Weighted sum: [N, K, d] -> scatter to [G, K, d]
    # q: [N, K], h: [N, d]
    # weighted_h[i, k, :] = q[i, k] * h[i, :]
    weighted_h = q.unsqueeze(-1) * h.unsqueeze(1)   # [N, K, d]

    # scatter_add over graph dimension
    g_idx = batch.view(-1, 1, 1).expand(-1, K, d)   # [N, K, d]
    M_sum = torch.zeros(num_graphs, K, d, device=device)
    M_sum.scatter_add_(0, g_idx, weighted_h)

    # Normalize by total weight per (graph, slot)
    w_sum = torch.zeros(num_graphs, K, device=device)
    w_sum.scatter_add_(0, batch.view(-1, 1).expand(-1, K), q)
    w_sum = w_sum.clamp(min=1e-9).unsqueeze(-1)      # [G, K, 1]

    return M_sum / w_sum                              # [G, K, d]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def _fuse(
    h: torch.Tensor,
    c: torch.Tensor,
    mode: str,
    gate_proj: nn.Module | None,
) -> torch.Tensor:
    if mode == "add":
        return h + c
    if mode == "residual":
        return F.relu(h + c)
    if mode == "gate":
        g = torch.sigmoid(gate_proj(torch.cat([h, c], dim=-1)))
        return g * h + (1 - g) * c
    raise ValueError(f"Unknown fusion mode '{mode}'")

"""TNMRouterPostGNN — sanity-check variant.

Architecture:
  GNN layers -> H_all -> RouterAssign -> C -> fuse(H_root, C_root) -> classifier

The router is applied to all node embeddings after GNN, then the root's
context vector is fused with the root's GNN embedding for classification.
This is a simple sanity check that the router can learn useful prototypes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from models.gnn.layers import GNNLayer
from models.router.router_bank import SparseRouterBank


class TNMRouterPostGNN(nn.Module):
    """Router applied after GNN, root node reads from global prototype context.

    Parameters
    ----------
    num_classes:     Number of output classes.
    in_dim:          Embedding vocab size.
    h_dim:           Hidden dimension.
    num_layers:      Number of GNN layers.
    gnn_type:        GCN | GIN | GAT | GGNN.
    num_routers:     Number of prototype slots K.
    topk:            Top-k active slots.
    tau:             Softmax temperature.
    ema_beta:        EMA decay (0 = grad update).
    update_mode:     'grad' or 'ema'.
    init_mode:       Prototype initialization mode.
    fusion:          'add' | 'residual' | 'gate'.
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
    ):
        super().__init__()
        self.num_layers = num_layers
        self.update_mode = update_mode

        self.key_embedding   = nn.Embedding(in_dim + 1, h_dim)
        self.value_embedding = nn.Embedding(in_dim + 1, h_dim)

        self.layers = nn.ModuleList([
            GNNLayer(gnn_type, h_dim, h_dim) for _ in range(num_layers)
        ])

        self.router = SparseRouterBank(
            dim=h_dim, num_routers=num_routers, topk=topk,
            tau=tau, ema_beta=ema_beta, update_mode=update_mode,
            init_mode=init_mode,
        )

        self.fusion = fusion
        if fusion == "gate":
            self.gate_proj = nn.Linear(h_dim * 2, h_dim)
        elif fusion == "residual":
            pass  # simple add after linear
        # 'add': direct addition

        self.classifier = nn.Linear(h_dim, num_classes)

    def forward(self, batch: Batch):
        x, edge_index = batch.x, batch.edge_index
        root_mask = batch.root_mask

        h = self.key_embedding(x[:, 0]) + self.value_embedding(x[:, 1])
        for layer in self.layers:
            h = layer(h, edge_index)

        # Router over all nodes
        q_dense, q_sparse = self.router.compute_assignment(h)
        context = self.router.context(q_sparse)   # [N, d]

        # EMA update
        if self.training and self.update_mode == "ema":
            self.router.ema_update(h, q_sparse)

        # Fuse at root
        root_h = h[root_mask]
        root_c = context[root_mask]
        fused = _fuse(root_h, root_c, self.fusion,
                      getattr(self, "gate_proj", None))

        return self.classifier(fused)

    def router_stats(self, batch: Batch):
        with torch.no_grad():
            x, edge_index = batch.x, batch.edge_index
            h = self.key_embedding(x[:, 0]) + self.value_embedding(x[:, 1])
            for layer in self.layers:
                h = layer(h, edge_index)
            _, q_sparse = self.router.compute_assignment(h)
            return self.router.stats(q_sparse)


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

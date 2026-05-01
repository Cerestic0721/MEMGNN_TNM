"""TNM Baseline model — faithful port of bottleneck/models/graph_model.py.

Architecture:
  x[:, 0] -> key_embedding   (Embedding)
  x[:, 1] -> value_embedding (Embedding)
  h = key_emb + val_emb
  for each layer:
      h = GNNLayer(h, edge_index)   [last layer optionally FA]
  root_h = h[root_mask]
  logits = Linear(h_dim, num_classes)(root_h)

FA (Fully-Adjacent) last layer:
  Replaces the last GNN layer's edge_index with a star graph where every
  node in the graph connects to the root, enabling direct information flow
  and bypassing over-squashing.  Controlled by use_fa_layer=True.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from models.gnn.layers import GNNLayer


class TNMBaselineModel(nn.Module):
    """Baseline GNN for Tree-NeighborsMatch, mirroring bottleneck GraphModel.

    Parameters
    ----------
    num_classes:  Number of output classes (= num_leaves).
    in_dim:       Embedding vocabulary size (= num_leaves; indices go up to in_dim).
    h_dim:        Hidden dimension.
    num_layers:   Number of GNN layers (bottleneck default: depth + 1).
    gnn_type:     One of GCN, GIN, GAT, GGNN.
    use_fa_layer: Replace last layer with a Fully-Adjacent layer.
    """

    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        h_dim: int = 32,
        num_layers: int = 4,
        gnn_type: str = "GCN",
        use_fa_layer: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.use_fa_layer = use_fa_layer

        # Embedding vocab: indices 0..in_dim (0 = padding/intermediate)
        self.key_embedding   = nn.Embedding(in_dim + 1, h_dim)
        self.value_embedding = nn.Embedding(in_dim + 1, h_dim)

        self.layers: nn.ModuleList = nn.ModuleList()
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if is_last and use_fa_layer:
                # FA layer: plain linear (no graph conv), applied after star-edge rewiring
                self.layers.append(_FALayer(h_dim, h_dim, gnn_type))
            else:
                self.layers.append(GNNLayer(gnn_type, h_dim, h_dim))

        self.classifier = nn.Linear(h_dim, num_classes)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return logits [num_graphs, num_classes]."""
        x          = batch.x           # [N, 2] LongTensor
        edge_index = batch.edge_index  # [2, E]
        root_mask  = batch.root_mask   # [N] bool
        b          = batch.batch       # [N] graph index

        h = self.key_embedding(x[:, 0]) + self.value_embedding(x[:, 1])

        for i, layer in enumerate(self.layers):
            is_last = (i == self.num_layers - 1)
            if is_last and self.use_fa_layer:
                fa_edge_index = _build_fa_edge_index(root_mask, b)
                h = layer(h, fa_edge_index)
            else:
                h = layer(h, edge_index)

        root_h = h[root_mask]          # [num_graphs, h_dim]
        return self.classifier(root_h)


# ---------------------------------------------------------------------------
# FA layer helpers
# ---------------------------------------------------------------------------

class _FALayer(nn.Module):
    """Fully-Adjacent layer: every node attends to the graph root.

    Implemented as a standard GNNLayer but called with a star edge_index
    (all nodes -> root) constructed externally.
    """

    def __init__(self, in_dim: int, out_dim: int, gnn_type: str):
        super().__init__()
        self.gnn = GNNLayer(gnn_type, in_dim, out_dim)

    def forward(self, h: torch.Tensor, fa_edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn(h, fa_edge_index)


def _build_fa_edge_index(root_mask: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Build star edge_index: every node -> its graph's root node.

    Parameters
    ----------
    root_mask : [N] bool — True at root node of each graph.
    batch     : [N] int  — graph index for each node.

    Returns
    -------
    edge_index : [2, N] — (src=all nodes, dst=corresponding root).
    """
    device = root_mask.device
    num_nodes = root_mask.size(0)

    # root node index per graph
    root_indices = torch.where(root_mask)[0]          # [num_graphs]
    # map each node to its graph's root
    dst = root_indices[batch]                          # [N]
    src = torch.arange(num_nodes, device=device)      # [N]

    return torch.stack([src, dst], dim=0)

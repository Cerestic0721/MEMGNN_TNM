"""GNN layer implementations matching bottleneck/common.py GNN_TYPE logic.

Supported types: GCN, GIN, GAT, GGNN
Each type has fixed defaults for activation, residual, and layer_norm
that match the bottleneck paper's Figure 3 configurations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, GatedGraphConv

# Per-type defaults matching bottleneck/common.py
_GNN_DEFAULTS = {
    "GCN":  dict(activation=True,  residual=True,  layer_norm=True),
    "GIN":  dict(activation=True,  residual=False, layer_norm=False),
    "GAT":  dict(activation=False, residual=True,  layer_norm=True),
    "GGNN": dict(activation=False, residual=True,  layer_norm=True),
}


class GNNLayer(nn.Module):
    """Single GNN message-passing layer with optional activation/residual/norm.

    Parameters
    ----------
    gnn_type:   One of GCN, GIN, GAT, GGNN.
    in_dim:     Input feature dimension.
    out_dim:    Output feature dimension.
    activation: Apply ReLU after message passing (overrides type default if given).
    residual:   Add residual connection (overrides type default if given).
    layer_norm: Apply LayerNorm after residual (overrides type default if given).
    """

    def __init__(
        self,
        gnn_type: str,
        in_dim: int,
        out_dim: int,
        activation: bool | None = None,
        residual: bool | None = None,
        layer_norm: bool | None = None,
    ):
        super().__init__()
        gnn_type = gnn_type.upper()
        if gnn_type not in _GNN_DEFAULTS:
            raise ValueError(f"Unknown gnn_type '{gnn_type}'. Choose from {list(_GNN_DEFAULTS)}")

        defaults = _GNN_DEFAULTS[gnn_type]
        self.use_activation = activation if activation is not None else defaults["activation"]
        self.use_residual   = residual   if residual   is not None else defaults["residual"]
        self.use_layer_norm = layer_norm if layer_norm is not None else defaults["layer_norm"]

        self.conv = _build_conv(gnn_type, in_dim, out_dim)

        self.residual_proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if self.use_residual and in_dim != out_dim
            else None
        )
        self.norm = nn.LayerNorm(out_dim) if self.use_layer_norm else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)

        if self.use_residual:
            res = self.residual_proj(x) if self.residual_proj is not None else x
            h = h + res

        if self.use_layer_norm:
            h = self.norm(h)

        if self.use_activation:
            h = F.relu(h)

        return h


def _build_conv(gnn_type: str, in_dim: int, out_dim: int) -> nn.Module:
    if gnn_type == "GCN":
        return GCNConv(in_dim, out_dim)

    if gnn_type == "GGNN":
        # GatedGraphConv ignores in_dim at runtime; out_dim is the hidden size.
        return GatedGraphConv(out_channels=out_dim, num_layers=1)

    if gnn_type == "GIN":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        return GINConv(mlp)

    if gnn_type == "GAT":
        heads = 4
        assert out_dim % heads == 0, f"out_dim={out_dim} must be divisible by heads={heads} for GAT"
        return GATConv(in_dim, out_dim // heads, heads=heads, concat=True)

    raise ValueError(gnn_type)

"""TNMRouterGraphMemory — GCNII + graph-wise prototype memory.

Main pipeline:
  1. key/value embedding -> h0
  2. GCNII (or GCN/GIN/GAT/GGNN) backbone -> H [N, d]
  3. All nodes (or non-root) compute sparse assignment q[i,k]
  4. Per-graph memory: M[g,k] = weighted mean of H over nodes in graph g
  5. Root reads from memory: root_ctx[g] = sum_k root_q[g,k] * M[g,k]
  6. Fusion: H_final = fuse(root_h, root_ctx)
  7. Classifier: logits = Linear(H_final)
  8. EMA M-step every m_step_interval outer epochs (called from train loop)

Key invariant: M is built per-graph via batch.batch — no cross-graph contamination.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from models.gnn.layers import GNNLayer, GCNIILayer
from models.router.prototype_bank import PrototypeBank
from models.router.assigner import SoftAssigner
from models.router.topk_router import TopKRouter


class TNMRouterGraphMemory(nn.Module):
    """Per-graph memory slots built by routing all nodes; root reads from them.

    Parameters
    ----------
    num_classes:            Number of output classes.
    in_dim:                 Embedding vocab size.
    h_dim:                  Hidden dimension.
    num_layers:             Number of GNN/GCNII layers.
    gnn_type:               GCN | GIN | GAT | GGNN | GCNII.
    dropout:                Dropout on embeddings and GCNII layers.
    num_routers:            Number of prototype slots K.
    topk_assign:            Top-k active slots per node.
    assignment_tau:         Softmax temperature for assignment.
    assign_type:            dot | bilinear | linear_proj.
    router_fusion:          none | add | residual | concat.
    residual_gamma:         Scalar for residual fusion (None = learnable).
    proto_update:           grad | ema.
    ema_beta:               EMA decay (CMMP default: 0.03).
    ema_normalize_proto:    L2-normalize prototypes after EMA update.
    ema_reinit_dead:        Reinitialize dead prototypes.
    ema_dead_threshold:     Usage threshold below which a prototype is dead.
    ema_reinit_patience:    Steps before reinitializing a dead prototype.
    proto_init_mode:        default | gaussian_normalized | gaussian_scaled | qr_orthogonal.
    m_step_interval:        EMA update every N outer epochs (called externally).
    memory_from_all:        Build memory from all nodes (True) or non-root only (False).
    gcnii_alpha:            GCNII initial residual weight.
    gcnii_theta:            GCNII identity mapping strength.
    gcnii_shared_weights:   Share W_1/W_2 in GCN2Conv.
    use_measure_space:      Apply frozen orthogonal transform before similarity.
    measure_transform_type: identity | frozen_qr_orthogonal | frozen_hadamard_sign.
    measure_apply_mode:     none | route_only | context_only | route_and_context.
    measure_seed:           RNG seed for the frozen transform matrix.
    """

    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        h_dim: int = 512,
        num_layers: int = 16,
        gnn_type: str = "GCNII",
        dropout: float = 0.7,
        num_routers: int = 128,
        topk_assign: int = 16,
        assignment_tau: float = 1.0,
        assign_type: str = "dot",
        router_fusion: str = "residual",
        residual_gamma: float | None = 0.02,
        proto_update: str = "ema",
        ema_beta: float = 0.03,
        ema_normalize_proto: bool = True,
        ema_reinit_dead: bool = False,
        ema_dead_threshold: float = 1e-4,
        ema_reinit_patience: int = 20,
        proto_init_mode: str = "default",
        m_step_interval: int = 20,
        memory_from_all: bool = False,
        gcnii_alpha: float = 0.2,
        gcnii_theta: float = 1.0,
        gcnii_shared_weights: bool = True,
        use_measure_space: bool = False,
        measure_transform_type: str = "frozen_qr_orthogonal",
        measure_apply_mode: str = "route_only",
        measure_seed: int = 42,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.dropout = dropout
        self.memory_from_all = memory_from_all
        self.m_step_interval = m_step_interval
        self.assignment_tau = assignment_tau
        self.router_fusion = router_fusion
        self.use_gcnii = gnn_type.upper() == "GCNII"
        self._outer_epoch = 0

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
                    dropout=dropout,
                )
                for i in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                GNNLayer(gnn_type, h_dim, h_dim) for _ in range(num_layers)
            ])

        self.proto_bank = PrototypeBank(
            num_prototypes=num_routers,
            hidden_dim=h_dim,
            proto_update=proto_update,
            ema_beta=ema_beta,
            ema_normalize_proto=ema_normalize_proto,
            ema_reinit_dead=ema_reinit_dead,
            ema_dead_threshold=ema_dead_threshold,
            ema_reinit_patience=ema_reinit_patience,
            proto_init_mode=proto_init_mode,
        )

        self.assigner = SoftAssigner(h_dim, num_routers, assign_type)
        self.topk_router = TopKRouter(topk_assign)

        self.use_measure_space = use_measure_space
        self.measure_apply_mode = measure_apply_mode
        measure_matrix = self._build_measure_matrix(
            h_dim, use_measure_space, measure_transform_type, measure_seed
        )
        self.register_buffer("measure_matrix", measure_matrix)

        if router_fusion == "concat":
            self.concat_proj = nn.Linear(h_dim * 2, h_dim)
        if router_fusion == "residual":
            if residual_gamma is None:
                self.residual_gamma = nn.Parameter(torch.tensor(0.02))
            else:
                self.register_buffer("residual_gamma", torch.tensor(float(residual_gamma)))

        self.classifier = nn.Linear(h_dim, num_classes)

    @staticmethod
    def _build_measure_matrix(
        d: int, use: bool, transform_type: str, seed: int
    ) -> torch.Tensor:
        if not use or transform_type == "identity":
            return torch.eye(d)
        if transform_type == "frozen_qr_orthogonal":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            gaussian = torch.randn(d, d, generator=gen)
            q, r = torch.linalg.qr(gaussian)
            signs = torch.sign(torch.diag(r))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            return q * signs.unsqueeze(0)
        if d & (d - 1) != 0:
            raise ValueError("frozen_hadamard_sign requires hidden_dim to be a power of two")
        hadamard = torch.tensor([[1.0]])
        while hadamard.size(0) < d:
            hadamard = torch.cat([
                torch.cat([hadamard, hadamard], dim=1),
                torch.cat([hadamard, -hadamard], dim=1),
            ], dim=0)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        signs = (torch.randint(0, 2, (d,), generator=gen, dtype=torch.float32)
                 .mul_(2.0).sub_(1.0))
        return (hadamard / d ** 0.5) * signs.unsqueeze(0)

    def _route_prototypes(self, P: torch.Tensor) -> torch.Tensor:
        if self.use_measure_space and self.measure_apply_mode in ("route_only", "route_and_context"):
            return P @ self.measure_matrix.to(P.device, P.dtype)
        return P

    def _context_prototypes(self, P: torch.Tensor) -> torch.Tensor:
        if self.use_measure_space and self.measure_apply_mode in ("context_only", "route_and_context"):
            return P @ self.measure_matrix.to(P.device, P.dtype)
        return P

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h0 = self.key_embedding(x[:, 0]) + self.value_embedding(x[:, 1])
        h0 = F.dropout(h0, self.dropout, training=self.training)
        h = h0
        if self.use_gcnii:
            for layer in self.layers:
                h = layer(h, h0, edge_index)
        else:
            for layer in self.layers:
                h = layer(h, edge_index)
        return h

    def _assign(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        P = self.proto_bank()
        P_route = self._route_prototypes(P)
        logits = self.assigner.logits(h, P_route) / self.assignment_tau
        q_dense = F.softmax(logits, dim=-1)
        q_sparse = self.topk_router(q_dense)
        return q_dense, q_sparse

    def forward(self, batch: Batch) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        root_mask = batch.root_mask
        b = batch.batch
        num_graphs = int(b.max().item()) + 1
        K = self.proto_bank.K

        h = self._encode(x, edge_index)

        if self.memory_from_all:
            h_mem, b_mem = h, b
        else:
            non_root = ~root_mask
            h_mem, b_mem = h[non_root], b[non_root]

        self.proto_bank.ensure_initialized(h_mem)
        _, q_sparse = self._assign(h_mem)

        # Store for post-backward EMA update (called by train loop via step_ema)
        if self.training and self.proto_bank.proto_update == "ema":
            self._last_h_mem = h_mem.detach()
            self._last_q_sparse = q_sparse.detach()

        P_ctx = self._context_prototypes(self.proto_bank())
        M = _build_graph_memory(h_mem, q_sparse, b_mem, num_graphs, K, P_ctx)

        root_h = h[root_mask]
        _, root_q = self._assign(root_h)
        root_ctx = (root_q.unsqueeze(-1) * M).sum(1)   # [G, d]

        fused = self._fuse(root_h, root_ctx)
        return self.classifier(fused)

    def step_ema(self) -> dict:
        """Call after loss.backward() to update prototypes via EMA.

        Only updates when outer_epoch % m_step_interval == 0.
        """
        if not self._should_update_ema():
            return {}
        h = getattr(self, "_last_h_mem", None)
        q = getattr(self, "_last_q_sparse", None)
        if h is None or q is None:
            return {}
        return self.proto_bank.update_ema(h, q)

    def _should_update_ema(self) -> bool:
        if self.proto_bank.proto_update != "ema":
            return False
        return self._outer_epoch % self.m_step_interval == 0

    def set_epoch(self, epoch: int) -> None:
        self._outer_epoch = int(epoch)

    def _fuse(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mode = self.router_fusion
        if mode == "none":
            return h
        if mode == "add":
            return h + c
        if mode == "residual":
            return h + self.residual_gamma * c
        if mode == "concat":
            return self.concat_proj(torch.cat([h, c], dim=-1))
        raise ValueError(f"Unknown router_fusion '{mode}'")

    @torch.no_grad()
    def router_stats(self, batch: Batch) -> dict:
        h = self._encode(batch.x, batch.edge_index)
        _, q_sparse = self._assign(h)
        usage = (q_sparse > 1e-6).float().mean(0)
        active_count = float((usage > 0).sum().item())
        entropy = -(q_sparse * (q_sparse + 1e-9).log()).sum(-1).mean().item()
        dead_ratio = float((usage < 1e-4).float().mean().item())
        return {
            "active_router_num": active_count,
            "dead_router_ratio": dead_ratio,
            "assignment_entropy": entropy,
        }


# ---------------------------------------------------------------------------
# Per-graph memory construction
# ---------------------------------------------------------------------------

def _build_graph_memory(
    h: torch.Tensor,
    q: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    K: int,
    P_ctx: torch.Tensor,
) -> torch.Tensor:
    """Build M[g, k] = weighted mean of h for graph g, slot k.

    Falls back to prototype P_ctx for slots with zero weight.

    Returns M : [num_graphs, K, d]
    """
    d = h.size(-1)
    device = h.device

    # Avoid materializing [N, K, d] (can be ~8 GiB for deep trees).
    # Loop over K slots instead: peak memory is [N, d] per iteration.
    M_sum = torch.zeros(num_graphs, K, d, device=device)
    b_idx = batch.unsqueeze(-1).expand(-1, d)               # [N, d]
    for k in range(K):
        M_sum[:, k, :].scatter_add_(0, b_idx, q[:, k:k+1] * h)

    w_sum = torch.zeros(num_graphs, K, device=device)
    w_sum.scatter_add_(0, batch.view(-1, 1).expand(-1, K), q)

    zero_mask = (w_sum < 1e-9).unsqueeze(-1)                 # [G, K, 1]
    M_norm = M_sum / w_sum.clamp(min=1e-9).unsqueeze(-1)
    M_fallback = P_ctx.unsqueeze(0).expand(num_graphs, -1, -1)
    return torch.where(zero_mask, M_fallback, M_norm)

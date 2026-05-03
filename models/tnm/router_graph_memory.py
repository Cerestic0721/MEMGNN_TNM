"""TNMRouterGraphMemory — GCNII + batch-level prototype memory.

Main pipeline:
  1. key/value embedding -> h0
  2. GCNII (or GCN/GIN/GAT/GGNN) backbone -> H [N, d]
  3. Non-root (or all) nodes compute sparse assignment q [N_mem, K]
  4. Batch-level memory: proto_context[k] = weighted mean of H over all nodes in batch
     shape: [K, d]  (stable across small graphs; aggregates entire batch)
  5. Root reads from memory: root_ctx = root_q @ proto_context  [G, d]
  6. Fusion: H_final = fuse(root_h, root_ctx)
  7. Classifier: logits = Linear(H_final)

EMA update is epoch-level (not per mini-batch):
  Call epoch_ema_update(loader, device) once per outer_epoch from the train loop.
  It iterates over the full loader, accumulates q/h statistics, then commits
  a single prototype update — avoiding the instability of per-step updates.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.data import Batch

from models.gnn.layers import GNNLayer, GCNIILayer
from models.router.prototype_bank import PrototypeBank
from models.router.assigner import SoftAssigner
from models.router.topk_router import TopKRouter


class TNMRouterGraphMemory(nn.Module):
    """Batch-level prototype memory; root reads from it.

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
    residual_gamma:         Scalar for residual fusion; None = learnable sigmoid ~0.1.
    proto_update:           grad | ema.
    ema_beta:               EMA decay (CMMP default: 0.03).
    ema_init:               random | sample_h | farthest_h | kmeans_h.
    ema_normalize_proto:    L2-normalize prototypes after EMA update.
    ema_reinit_dead:        Reinitialize dead prototypes.
    ema_dead_threshold:     Usage threshold below which a prototype is dead.
    ema_reinit_patience:    Steps before reinitializing a dead prototype.
    proto_init_mode:        default | gaussian_normalized | gaussian_scaled | qr_orthogonal.
    m_step_interval:        EMA update every N outer epochs (1 = every epoch).
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
        residual_gamma: float | None = 0.1,
        proto_update: str = "ema",
        ema_beta: float = 0.03,
        ema_init: str = "sample_h",
        ema_normalize_proto: bool = True,
        ema_reinit_dead: bool = False,
        ema_dead_threshold: float = 1e-4,
        ema_reinit_patience: int = 20,
        proto_init_mode: str = "default",
        m_step_interval: int = 1,
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
            ema_init=ema_init,
            ema_normalize_proto=ema_normalize_proto,
            ema_reinit_dead=ema_reinit_dead,
            ema_dead_threshold=ema_dead_threshold,
            ema_reinit_patience=ema_reinit_patience,
            proto_init_mode=proto_init_mode,
        )

        self.assigner    = SoftAssigner(h_dim, num_routers, assign_type)
        self.topk_router = TopKRouter(topk_assign)

        self.use_measure_space  = use_measure_space
        self.measure_apply_mode = measure_apply_mode
        measure_matrix = self._build_measure_matrix(
            h_dim, use_measure_space, measure_transform_type, measure_seed
        )
        self.register_buffer("measure_matrix", measure_matrix)

        # Fusion layer
        self._gamma_learnable = False
        if router_fusion == "concat":
            self.concat_proj = nn.Linear(h_dim * 2, h_dim)
        if router_fusion == "residual":
            if residual_gamma is None:
                # Sigmoid-parameterized learnable gamma, initialized so sigmoid ≈ 0.1
                self._gamma_learnable = True
                self._gamma_raw = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
            else:
                self.register_buffer("residual_gamma", torch.tensor(float(residual_gamma)))

        self.classifier = nn.Linear(h_dim, num_classes)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

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
                torch.cat([hadamard,  hadamard], dim=1),
                torch.cat([hadamard, -hadamard], dim=1),
            ], dim=0)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        signs = (torch.randint(0, 2, (d,), generator=gen, dtype=torch.float32)
                 .mul_(2.0).sub_(1.0))
        return (hadamard / d ** 0.5) * signs.unsqueeze(0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
                if self.training:
                    # Recompute activations during backward to save memory (16 layers).
                    h = grad_checkpoint(layer, h, h0, edge_index, use_reentrant=False)
                else:
                    h = layer(h, h0, edge_index)
        else:
            for layer in self.layers:
                h = layer(h, edge_index)
        return h

    def _assign(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        P       = self.proto_bank()
        P_route = self._route_prototypes(P)
        logits  = self.assigner.logits(h, P_route) / self.assignment_tau
        q_dense  = F.softmax(logits, dim=-1)
        q_sparse = self.topk_router(q_dense)
        return q_dense, q_sparse

    def _fuse(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mode = self.router_fusion
        if mode == "none":
            return h
        if mode == "add":
            return h + c
        if mode == "residual":
            gamma = torch.sigmoid(self._gamma_raw) if self._gamma_learnable else self.residual_gamma
            return h + gamma * c
        if mode == "concat":
            return self.concat_proj(torch.cat([h, c], dim=-1))
        raise ValueError(f"Unknown router_fusion '{mode}'")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Batch) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        root_mask = batch.root_mask
        b         = batch.batch

        h = self._encode(x, edge_index)                          # [N, d]

        if self.memory_from_all:
            h_mem = h
        else:
            h_mem = h[~root_mask]                                # [N_mem, d]

        # Initialize prototypes from first batch if using sample_h
        self.proto_bank.ensure_initialized(h_mem)

        _, q_sparse = self._assign(h_mem)                        # [N_mem, K]

        # Batch-level memory: aggregate all nodes in batch → [K, d]
        P_ctx        = self._context_prototypes(self.proto_bank())
        proto_context = _build_batch_memory(h_mem, q_sparse, P_ctx)  # [K, d]

        # Root reads from batch-level memory
        root_h       = h[root_mask]                              # [G, d]
        _, root_q    = self._assign(root_h)                      # [G, K]
        root_ctx     = root_q @ proto_context                    # [G, d]

        assert root_ctx.shape == root_h.shape, (
            f"root_ctx {root_ctx.shape} != root_h {root_h.shape}"
        )

        fused = self._fuse(root_h, root_ctx)
        return self.classifier(fused)

    # ------------------------------------------------------------------
    # Epoch-level EMA update (called once per outer_epoch from train loop)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def epoch_ema_update(self, loader, device, quiet: bool = False) -> dict:
        """Full-pass EMA update over the entire train loader.

        Accumulates q/h statistics across all batches, then commits a single
        prototype update.  Guarantees exactly one update per outer_epoch call.

        Returns a dict with diagnostics; prints a summary line unless quiet=True.
        """
        if self.proto_bank.proto_update != "ema":
            return {"ema_executed": False}
        if not self._should_update_ema():
            return {"ema_executed": False}

        K = self.proto_bank.K
        d = self.h_dim

        weighted_sum = torch.zeros(K, d, device=device)
        w_sum        = torch.zeros(K,    device=device)

        self.eval()
        for batch in loader:
            batch = batch.to(device)
            h     = self._encode(batch.x, batch.edge_index)

            if self.memory_from_all:
                h_mem = h
            else:
                h_mem = h[~batch.root_mask]

            self.proto_bank.ensure_initialized(h_mem)
            _, q_sparse = self._assign(h_mem)                    # [N_mem, K]

            weighted_sum += q_sparse.T @ h_mem                   # [K, d]
            w_sum        += q_sparse.sum(dim=0)                  # [K]

        # Compute per-slot mean; empty slots keep their current prototype value
        active = w_sum > self.proto_bank.ema_min_count
        mu     = self.proto_bank().clone()                       # [K, d] — fallback
        mu[active] = weighted_sum[active] / w_sum[active].unsqueeze(-1)

        stats = self.proto_bank.update_ema_from_mu(mu, w_sum)

        non_empty = int(active.sum().item())
        if not quiet:
            print(
                f"  [EMA ep={self._outer_epoch}] executed=True  "
                f"non_empty_slots={non_empty}/{K}  "
                f"w_sum min={w_sum.min():.1f} max={w_sum.max():.1f} mean={w_sum.mean():.1f}  "
                f"delta={stats.get('ema_update_delta', 0):.4f}  "
                f"dead={stats.get('dead_proto_count', 0)}",
                flush=True,
            )

        return {"ema_executed": True, "non_empty_slots": non_empty, **stats}

    def _should_update_ema(self) -> bool:
        return self._outer_epoch % self.m_step_interval == 0

    def set_epoch(self, epoch: int) -> None:
        self._outer_epoch = int(epoch)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def router_stats(self, batch: Batch) -> dict:
        h = self._encode(batch.x, batch.edge_index)
        _, q_sparse  = self._assign(h)
        usage        = (q_sparse > 1e-6).float().mean(0)
        active_count = float((usage > 0).sum().item())
        entropy      = -(q_sparse * (q_sparse + 1e-9).log()).sum(-1).mean().item()
        dead_ratio   = float((usage < 1e-4).float().mean().item())
        return {
            "active_router_num":   active_count,
            "dead_router_ratio":   dead_ratio,
            "assignment_entropy":  entropy,
        }


# ---------------------------------------------------------------------------
# Batch-level memory construction
# ---------------------------------------------------------------------------

def _build_batch_memory(
    h: torch.Tensor,
    q: torch.Tensor,
    P_ctx: torch.Tensor,
) -> torch.Tensor:
    """Aggregate all nodes in the batch into a [K, d] proto_context.

    proto_context[k] = weighted mean of h over all nodes assigned to slot k.
    Empty slots fall back to the current prototype P_ctx[k].

    Args:
        h     : [N_mem, d]  node embeddings
        q     : [N_mem, K]  sparse assignment weights
        P_ctx : [K, d]      current prototype vectors (fallback)

    Returns:
        proto_context : [K, d]
    """
    weighted_sum  = q.T @ h                                      # [K, d]
    w_sum         = q.sum(dim=0)                                 # [K]
    empty         = w_sum < 1e-9                                 # [K] bool

    proto_context = weighted_sum / w_sum.clamp(min=1e-9).unsqueeze(-1)  # [K, d]
    # Fallback: empty slots use current prototype
    proto_context = torch.where(empty.unsqueeze(-1), P_ctx, proto_context)

    return proto_context                                         # [K, d]

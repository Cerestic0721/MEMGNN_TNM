"""SparseRouterBank — simplified port of CMMP PrototypeBank + TopKRouter.

Stripped of: KL loss, mixture_pi, Dirichlet pi, usage_balance_loss,
             orthogonal_regularizer, measure_space, encode/decode, router_gate.

Core flow:
  h [N, d]
  -> sim(h, P) / tau                    # [N, K] logits
  -> softmax -> q_dense [N, K]
  -> top-k truncate + renorm -> q_sparse [N, K]
  -> context = q_sparse @ P             # [N, d]
  -> EMA update: P_k <- beta*P_k + (1-beta)*mean(h[q_k>0])
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseRouterBank(nn.Module):
    """K prototype vectors with sparse top-k routing.

    Parameters
    ----------
    dim:               Feature dimension.
    num_routers:       Number of prototype slots K.
    topk:              Number of active slots per node.
    tau:               Softmax temperature.
    ema_beta:          EMA decay for prototype update (0 = no EMA, use grad).
    update_mode:       'grad' (gradient) or 'ema'.
    init_mode:         'default' | 'gaussian_normalized' | 'gaussian_scaled' | 'qr_orthogonal'.
    init_var:          Variance for gaussian init modes.
    normalize_router:  L2-normalize prototypes before similarity.
    m_step_interval:   EMA update every N forward calls (1 = every step).
    """

    def __init__(
        self,
        dim: int,
        num_routers: int = 32,
        topk: int = 4,
        tau: float = 1.0,
        ema_beta: float = 0.9,
        update_mode: str = "ema",
        init_mode: str = "default",
        init_var: float = 1.0,
        normalize_router: bool = False,
        m_step_interval: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_routers = num_routers
        self.topk = topk
        self.tau = tau
        self.ema_beta = ema_beta
        self.update_mode = update_mode
        self.normalize_router = normalize_router
        self.m_step_interval = m_step_interval

        self._step = 0

        # Prototype matrix P: [K, d]
        P = self._init_prototypes(dim, num_routers, init_mode, init_var)
        if update_mode == "ema":
            self.register_buffer("P", P)
        else:
            self.P = nn.Parameter(P)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def compute_assignment(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dense and sparse routing weights.

        Parameters
        ----------
        h : [N, d]

        Returns
        -------
        q_dense  : [N, K] — softmax weights
        q_sparse : [N, K] — top-k truncated and renormalized
        """
        P = F.normalize(self.P, dim=-1) if self.normalize_router else self.P
        logits = h @ P.t() / self.tau          # [N, K]
        q_dense = F.softmax(logits, dim=-1)    # [N, K]
        q_sparse = _topk_normalize(q_dense, self.topk)
        return q_dense, q_sparse

    def context(self, q: torch.Tensor) -> torch.Tensor:
        """Compute context vectors from routing weights.

        Parameters
        ----------
        q : [N, K] — routing weights (typically q_sparse)

        Returns
        -------
        [N, d]
        """
        P = F.normalize(self.P, dim=-1) if self.normalize_router else self.P
        return q @ P                           # [N, d]

    def ema_update(
        self,
        h: torch.Tensor,
        q: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Update prototypes via EMA (no-op if update_mode='grad').

        Parameters
        ----------
        h : [N, d] — node features (detached internally)
        q : [N, K] — sparse routing weights

        Returns
        -------
        stats dict with 'active_count', 'mean_entropy'
        """
        if self.update_mode != "ema":
            return {}

        self._step += 1
        if self._step % self.m_step_interval != 0:
            return {}

        with torch.no_grad():
            h_det = h.detach()
            # weighted mean per prototype: [K, d]
            weight_sum = q.sum(0)              # [K]
            new_P = (q.t() @ h_det)            # [K, d]
            active = weight_sum > 1e-6
            new_P[active] = new_P[active] / weight_sum[active].unsqueeze(-1)
            # EMA — use .data to avoid autograd version-counter conflicts
            self.P.data[active] = (
                self.ema_beta * self.P.data[active]
                + (1 - self.ema_beta) * new_P[active]
            )

        return self.stats(q)

    def stats(self, q: torch.Tensor) -> Dict[str, float]:
        """Compute routing statistics for logging."""
        with torch.no_grad():
            usage = (q > 1e-6).float().mean(0)          # [K]
            active_count = float((usage > 0).sum().item())
            entropy = -(q * (q + 1e-9).log()).sum(-1).mean().item()
        return {"active_count": active_count, "mean_entropy": entropy}

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_prototypes(
        dim: int, K: int, mode: str, var: float
    ) -> torch.Tensor:
        if mode == "default":
            P = torch.empty(K, dim)
            nn.init.xavier_uniform_(P)
            return P

        if mode == "gaussian_normalized":
            P = torch.randn(K, dim) * math.sqrt(var)
            return F.normalize(P, dim=-1)

        if mode == "gaussian_scaled":
            return torch.randn(K, dim) * math.sqrt(var)

        if mode == "qr_orthogonal":
            if K <= dim:
                P = torch.randn(dim, K)
                Q, _ = torch.linalg.qr(P)
                return Q[:K].contiguous()
            else:
                # More prototypes than dim: fill with orthogonal blocks
                blocks = []
                remaining = K
                while remaining > 0:
                    n = min(remaining, dim)
                    P = torch.randn(dim, n)
                    Q, _ = torch.linalg.qr(P)
                    blocks.append(Q[:n].t())
                    remaining -= n
                return torch.cat(blocks, dim=0)[:K]

        raise ValueError(f"Unknown init_mode '{mode}'")


# ---------------------------------------------------------------------------
# Top-k truncation + renormalization
# ---------------------------------------------------------------------------

def _topk_normalize(q: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k values per row, zero the rest, renormalize to sum=1."""
    if k >= q.size(-1):
        return q
    topk_vals, topk_idx = q.topk(k, dim=-1)
    mask = torch.zeros_like(q)
    mask.scatter_(-1, topk_idx, 1.0)
    q_sparse = q * mask
    row_sum = q_sparse.sum(-1, keepdim=True).clamp(min=1e-9)
    return q_sparse / row_sum

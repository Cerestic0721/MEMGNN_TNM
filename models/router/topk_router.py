"""TopKRouter — migrated from CMMP_clean_github/models/routing/topk_router.py.

Truncates dense soft assignment to top-k and renormalizes.
"""

import torch
import torch.nn as nn


class TopKRouter(nn.Module):
    def __init__(self, topk: int):
        super().__init__()
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        self.topk = topk

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        K = r.size(-1)
        k = min(self.topk, K)
        _, topk_idx = torch.topk(r, k, dim=-1)
        mask = torch.zeros_like(r).scatter_(-1, topk_idx, 1.0)
        r_sparse = r * mask
        row_sum = r_sparse.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return r_sparse / row_sum

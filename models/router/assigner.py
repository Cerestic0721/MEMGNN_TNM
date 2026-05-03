"""SoftAssigner — migrated from CMMP_clean_github/models/routing/assigner.py.

Computes soft assignment r[N,K] = softmax(logits(h, P) / tau).
assign_type: "dot" | "bilinear" | "linear_proj"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAssigner(nn.Module):
    def __init__(self, hidden_dim: int, num_prototypes: int, assign_type: str = "dot"):
        super().__init__()
        self.assign_type = assign_type
        if assign_type == "bilinear":
            self.W = nn.Parameter(torch.eye(hidden_dim))
        elif assign_type == "linear_proj":
            self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif assign_type != "dot":
            raise ValueError(f"assign_type must be dot | bilinear | linear_proj, got '{assign_type}'")

    def logits(self, h: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        if self.assign_type == "dot":
            return h @ P.T
        if self.assign_type == "bilinear":
            return (h @ self.W) @ P.T
        return self.proj(h) @ P.T

    def forward(self, h: torch.Tensor, P: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return F.softmax(self.logits(h, P) / tau, dim=-1)

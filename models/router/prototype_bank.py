"""PrototypeBank — migrated from CMMP_clean_github/models/routing/prototype_bank.py.

Stripped of: pi/Dirichlet, KL loss, usage balance, orthogonal regularizer.
Kept: grad/ema update, dead reinit, ema_normalize_proto, init modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.init import xavier_uniform_


class PrototypeBank(nn.Module):
    """K prototype vectors with grad or EMA update."""

    def __init__(
        self,
        num_prototypes: int,
        hidden_dim: int,
        proto_update: str = "ema",
        ema_beta: float = 0.03,
        ema_normalize_proto: bool = True,
        ema_detach: bool = True,
        ema_init: str = "random",
        ema_min_count: float = 1e-8,
        ema_reinit_dead: bool = False,
        ema_dead_threshold: float = 1e-4,
        ema_reinit_patience: int = 20,
        proto_init_mode: str = "default",
        proto_init_gain: float = 1.0,
        seed: int = 123,
    ):
        super().__init__()
        if proto_update not in ("grad", "ema"):
            raise ValueError("proto_update must be 'grad' or 'ema'")
        if ema_init not in ("random", "sample_h", "farthest_h", "kmeans_h"):
            raise ValueError("ema_init must be random | sample_h | farthest_h | kmeans_h")
        if proto_init_mode not in ("default", "gaussian_normalized", "gaussian_scaled", "qr_orthogonal"):
            raise ValueError("proto_init_mode must be default | gaussian_normalized | gaussian_scaled | qr_orthogonal")

        self.K = num_prototypes
        self.d = hidden_dim
        self.proto_update = proto_update
        self.ema_beta = float(ema_beta)
        self.ema_normalize_proto = bool(ema_normalize_proto)
        self.ema_detach = bool(ema_detach)
        self.ema_init = ema_init
        self.ema_min_count = float(ema_min_count)
        self.ema_reinit_dead = bool(ema_reinit_dead)
        self.ema_dead_threshold = float(ema_dead_threshold)
        self.ema_reinit_patience = int(ema_reinit_patience)
        self.proto_init_mode = proto_init_mode
        self.proto_init_gain = float(proto_init_gain)
        self.seed = int(seed)
        self._sample_offset = 0
        self._ema_update_count = 0

        initial = self._initial_prototypes(num_prototypes, hidden_dim, proto_init_mode)

        if proto_update == "grad":
            self.prototypes = nn.Parameter(initial)
        else:
            self.register_buffer("prototypes", initial)

        initialized = ema_init == "random" or proto_update == "grad"
        self.register_buffer("_ema_initialized", torch.tensor(initialized, dtype=torch.bool))
        self.register_buffer("_dead_steps", torch.zeros(num_prototypes, dtype=torch.long))

    def forward(self) -> torch.Tensor:
        return self.prototypes

    def _initial_prototypes(self, K: int, d: int, mode: str) -> torch.Tensor:
        if mode == "default":
            initial = torch.empty(K, d)
            xavier_uniform_(initial)
            return F.normalize(initial, p=2, dim=-1)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        if mode in ("gaussian_normalized", "gaussian_scaled"):
            initial = torch.randn(K, d, generator=generator)
            gain = self.proto_init_gain if mode == "gaussian_scaled" else 1.0
            return F.normalize(initial, p=2, dim=-1) * gain

        if K > d:
            warnings.warn(
                "K > hidden_dim, strict row-wise orthogonality impossible. Falling back to gaussian_normalized.",
                UserWarning, stacklevel=2,
            )
            initial = torch.randn(K, d, generator=generator)
            return F.normalize(initial, p=2, dim=-1)

        gaussian = torch.randn(d, K, generator=generator)
        q, r = torch.linalg.qr(gaussian)
        signs = torch.sign(torch.diag(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(0)
        return q[:, :K].T.contiguous()

    @property
    def ema_initialized(self) -> bool:
        return bool(self._ema_initialized.item())

    def ensure_initialized(self, h: torch.Tensor, force: bool = False) -> None:
        if self.proto_update != "ema":
            return
        if self.ema_initialized and not force:
            return
        with torch.no_grad():
            h_source = h.detach() if self.ema_detach else h
            if self.ema_init == "random":
                proto = self.prototypes.detach().clone()
            elif self.ema_init == "sample_h":
                proto = self._init_sample_h(h_source)
            elif self.ema_init == "farthest_h":
                proto = self._init_farthest_h(h_source)
            else:
                proto = self._init_kmeans_h(h_source)
            if self.ema_normalize_proto:
                proto = F.normalize(proto, p=2, dim=-1)
            self.prototypes.copy_(proto)
            self._ema_initialized.fill_(True)
            self._dead_steps.zero_()

    def update_ema(self, h: torch.Tensor, q: torch.Tensor) -> dict:
        """EMA M-step update. h: [N,d], q: [N,K] sparse assignment."""
        if self.proto_update != "ema":
            return {"ema_update_delta": 0.0, "dead_proto_count": 0}

        with torch.no_grad():
            if self.ema_detach:
                h = h.detach()
                q = q.detach()
            self.ensure_initialized(h)

            old = self.prototypes.detach().clone()
            count = q.sum(dim=0)                                    # [K]
            weighted_sum = q.T @ h                                  # [K, d]
            mu = weighted_sum / count.clamp(min=self.ema_min_count).unsqueeze(-1)
            active = count > self.ema_min_count

            new = old.clone()
            if active.any():
                updated = (1.0 - self.ema_beta) * old[active] + self.ema_beta * mu[active]
                new[active] = updated
            if self.ema_normalize_proto:
                new = F.normalize(new, p=2, dim=-1)

            usage = count / count.sum().clamp(min=self.ema_min_count)
            dead_now = usage < self.ema_dead_threshold
            self._dead_steps = torch.where(
                dead_now, self._dead_steps + 1, torch.zeros_like(self._dead_steps)
            )
            if self.ema_reinit_dead:
                reinit_mask = self._dead_steps >= self.ema_reinit_patience
                if reinit_mask.any():
                    replacements = self._replacement_from_h(h, int(reinit_mask.sum().item()))
                    new[reinit_mask] = replacements.to(new.device, new.dtype)
                    self._dead_steps[reinit_mask] = 0

            self.prototypes.copy_(new)
            self._ema_update_count += 1
            denom = old.norm().clamp(min=self.ema_min_count)
            delta = (self.prototypes.detach() - old).norm() / denom
            return {
                "ema_update_delta": float(delta.item()),
                "dead_proto_count": int(dead_now.sum().item()),
            }

    def update_ema_from_mu(self, mu: torch.Tensor, count: torch.Tensor) -> dict:
        """EMA update from pre-computed per-slot mean mu [K,d] and weight count [K].

        Used by epoch-level EMA where statistics are accumulated over the full dataset
        before a single prototype update is committed.
        Assumes ensure_initialized() has already been called by the caller.
        """
        if self.proto_update != "ema":
            return {"ema_update_delta": 0.0, "dead_proto_count": 0}

        with torch.no_grad():
            old = self.prototypes.detach().clone()
            active = count > self.ema_min_count

            new = old.clone()
            if active.any():
                new[active] = (1.0 - self.ema_beta) * old[active] + self.ema_beta * mu[active]
            if self.ema_normalize_proto:
                new = F.normalize(new, p=2, dim=-1)

            usage = count / count.sum().clamp(min=self.ema_min_count)
            dead_now = usage < self.ema_dead_threshold
            self._dead_steps = torch.where(
                dead_now, self._dead_steps + 1, torch.zeros_like(self._dead_steps)
            )
            if self.ema_reinit_dead:
                reinit_mask = self._dead_steps >= self.ema_reinit_patience
                if reinit_mask.any():
                    n = int(reinit_mask.sum().item())
                    gen = self._generator(new.device)
                    replacements = F.normalize(
                        torch.randn(n, self.d, generator=gen, device=new.device), p=2, dim=-1
                    )
                    new[reinit_mask] = replacements.to(new.dtype)
                    self._dead_steps[reinit_mask] = 0

            self.prototypes.copy_(new)
            self._ema_update_count += 1
            denom = old.norm().clamp(min=self.ema_min_count)
            delta = (self.prototypes.detach() - old).norm() / denom
            return {
                "ema_update_delta": float(delta.item()),
                "dead_proto_count": int(dead_now.sum().item()),
            }

    def _generator(self, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed + self._sample_offset)
        self._sample_offset += 1
        return generator

    def _init_sample_h(self, h: torch.Tensor) -> torch.Tensor:
        n = h.size(0)
        generator = self._generator(h.device)
        if n >= self.K:
            idx = torch.randperm(n, device=h.device, generator=generator)[:self.K]
        else:
            idx = torch.randint(0, n, (self.K,), device=h.device, generator=generator)
        return h[idx].clone()

    def _init_farthest_h(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = F.normalize(h, p=2, dim=-1)
        n = h_norm.size(0)
        generator = self._generator(h.device)
        first = torch.randint(0, n, (1,), device=h.device, generator=generator)
        selected = [int(first.item())]
        min_dist = 1.0 - (h_norm @ h_norm[selected[0]].unsqueeze(-1)).squeeze(-1)
        for _ in range(1, self.K):
            next_idx = int(torch.argmax(min_dist).item())
            selected.append(next_idx)
            dist = 1.0 - (h_norm @ h_norm[next_idx].unsqueeze(-1)).squeeze(-1)
            min_dist = torch.minimum(min_dist, dist)
        return h[selected].clone()

    def _init_kmeans_h(self, h: torch.Tensor, iters: int = 10) -> torch.Tensor:
        centers = self._init_sample_h(h)
        for _ in range(iters):
            scores = h @ centers.T
            assign = scores.argmax(dim=-1)
            new_centers = centers.clone()
            for k in range(self.K):
                mask = assign == k
                if mask.any():
                    new_centers[k] = h[mask].mean(dim=0)
                else:
                    new_centers[k] = self._init_sample_h(h)[0]
            centers = new_centers
        return centers

    def _replacement_from_h(self, h: torch.Tensor, count: int) -> torch.Tensor:
        if self.ema_init in ("farthest_h", "kmeans_h"):
            proto = self._init_farthest_h(h)
            return F.normalize(proto[:count], p=2, dim=-1)
        proto = self._init_sample_h(h)[:count]
        return F.normalize(proto, p=2, dim=-1)

    def extra_repr(self) -> str:
        return f"K={self.K}, d={self.d}, proto_update={self.proto_update}, ema_beta={self.ema_beta}"

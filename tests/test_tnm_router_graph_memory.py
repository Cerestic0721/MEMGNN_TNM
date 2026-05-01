"""Tests for TNMRouterGraphMemory forward pass and memory correctness."""

import pytest
import torch
from torch_geometric.data import Batch

from datasets.tnm_dataset import build_tnm_datasets
from models.tnm.router_graph_memory import TNMRouterGraphMemory, _build_graph_memory
from models.tnm.router_post_gnn import TNMRouterPostGNN


def _make_batch(depth=3, n_samples=4, max_examples=64, seed=0):
    train, _, meta = build_tnm_datasets(depth=depth, max_examples=max_examples, seed=seed)
    samples = train[:n_samples]
    batch = Batch.from_data_list(samples)
    return batch, meta


class TestBuildGraphMemory:
    def test_shape(self):
        N, K, d, G = 20, 8, 16, 4
        h = torch.randn(N, d)
        q = torch.softmax(torch.randn(N, K), dim=-1)
        batch = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5)
        M = _build_graph_memory(h, q, batch, G, K)
        assert M.shape == (G, K, d)

    def test_no_cross_graph_contamination(self):
        # Two graphs with very different features; memory should not mix them
        N, K, d = 6, 4, 8
        h = torch.zeros(N, d)
        h[:3] = 1.0   # graph 0
        h[3:] = -1.0  # graph 1
        q = torch.ones(N, K) / K
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        M = _build_graph_memory(h, q, batch, 2, K)
        # Graph 0 memory should be ~1, graph 1 memory should be ~-1
        assert M[0].mean().item() > 0.5
        assert M[1].mean().item() < -0.5

    def test_weighted_mean_correctness(self):
        # Single graph, 2 nodes, 1 slot
        h = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        q = torch.tensor([[0.75], [0.25]])
        batch = torch.tensor([0, 0])
        M = _build_graph_memory(h, q, batch, 1, 1)
        # Expected: (0.75*[2,0] + 0.25*[0,2]) / 1.0 = [1.5, 0.5]
        expected = torch.tensor([[[1.5, 0.5]]])
        assert torch.allclose(M, expected, atol=1e-5)


class TestTNMRouterGraphMemoryForward:
    @pytest.mark.parametrize("gnn_type", ["GCN", "GIN"])
    def test_output_shape(self, gnn_type):
        batch, meta = _make_batch(depth=3, n_samples=4)
        model = TNMRouterGraphMemory(
            num_classes=meta["num_classes"],
            in_dim=meta["in_dim"],
            h_dim=32,
            num_layers=4,
            gnn_type=gnn_type,
            num_routers=16,
            topk=4,
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (4, meta["num_classes"])

    def test_no_nan(self):
        batch, meta = _make_batch(depth=3, n_samples=8)
        model = TNMRouterGraphMemory(
            num_classes=meta["num_classes"],
            in_dim=meta["in_dim"],
            h_dim=32,
            num_layers=4,
            gnn_type="GCN",
            num_routers=16,
            topk=4,
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        assert not torch.isnan(logits).any()

    @pytest.mark.parametrize("fusion", ["add", "residual", "gate"])
    def test_fusion_modes(self, fusion):
        batch, meta = _make_batch(depth=3, n_samples=4)
        model = TNMRouterGraphMemory(
            num_classes=meta["num_classes"],
            in_dim=meta["in_dim"],
            h_dim=32,
            num_layers=4,
            gnn_type="GCN",
            num_routers=16,
            topk=4,
            fusion=fusion,
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (4, meta["num_classes"])

    def test_ema_update_runs(self):
        batch, meta = _make_batch(depth=3, n_samples=8)
        model = TNMRouterGraphMemory(
            num_classes=meta["num_classes"],
            in_dim=meta["in_dim"],
            h_dim=32,
            num_layers=4,
            gnn_type="GCN",
            num_routers=16,
            topk=4,
            update_mode="ema",
        )
        model.train()
        P_before = model.router.P.clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        # EMA should have changed P
        assert not torch.allclose(model.router.P, P_before)

    def test_grad_update_mode(self):
        batch, meta = _make_batch(depth=3, n_samples=8)
        model = TNMRouterGraphMemory(
            num_classes=meta["num_classes"],
            in_dim=meta["in_dim"],
            h_dim=32,
            num_layers=4,
            gnn_type="GCN",
            num_routers=16,
            topk=4,
            update_mode="grad",
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0


class TestTNMRouterPostGNNForward:
    def test_output_shape(self):
        batch, meta = _make_batch(depth=3, n_samples=4)
        model = TNMRouterPostGNN(
            num_classes=meta["num_classes"],
            in_dim=meta["in_dim"],
            h_dim=32,
            num_layers=4,
            gnn_type="GCN",
            num_routers=16,
            topk=4,
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (4, meta["num_classes"])

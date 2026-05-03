"""tests/test_gcnii_tnm_forward.py — GCNII backbone forward/backward on TNM batch."""

import torch
import pytest
from torch_geometric.data import Batch
from datasets.tnm_dataset import build_tnm_datasets
from torch_geometric.loader import DataLoader
from models.gnn.layers import GCNIILayer
from models.tnm.baseline import TNMBaselineModel


@pytest.fixture
def small_batch():
    train_list, _, _ = build_tnm_datasets(depth=2, max_examples=64, seed=0)
    loader = DataLoader(train_list[:8], batch_size=8, shuffle=False)
    return next(iter(loader))


def test_gcnii_layer_forward(small_batch):
    b = small_batch
    h_dim = 16
    layer = GCNIILayer(hidden_dim=h_dim, alpha=0.2, theta=1.0, layer=1)
    h0 = torch.randn(b.x.size(0), h_dim)
    h = layer(h0, h0, b.edge_index)
    assert h.shape == (b.x.size(0), h_dim)


def test_gcnii_layer_backward(small_batch):
    b = small_batch
    h_dim = 16
    layer = GCNIILayer(hidden_dim=h_dim, alpha=0.2, theta=1.0, layer=1)
    h0 = torch.randn(b.x.size(0), h_dim, requires_grad=True)
    h = layer(h0, h0, b.edge_index)
    h.sum().backward()
    assert h0.grad is not None


def test_baseline_gcnii_forward(small_batch):
    b = small_batch
    model = TNMBaselineModel(
        num_classes=4, in_dim=4, h_dim=32, num_layers=3, gnn_type="GCN"
    )
    logits = model(b)
    assert logits.shape == (8, 4)


def test_baseline_gcnii_output_shape(small_batch):
    """Verify output shape matches num_graphs x num_classes."""
    b = small_batch
    num_graphs = int(b.batch.max().item()) + 1
    model = TNMBaselineModel(num_classes=4, in_dim=4, h_dim=32, num_layers=3, gnn_type="GCN")
    logits = model(b)
    assert logits.shape[0] == num_graphs
    assert logits.shape[1] == 4

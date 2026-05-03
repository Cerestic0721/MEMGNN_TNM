"""tests/test_router_graph_memory.py — TNMRouterGraphMemory correctness tests."""

import torch
import pytest
from torch_geometric.loader import DataLoader
from datasets.tnm_dataset import build_tnm_datasets
from models.tnm.router_graph_memory import TNMRouterGraphMemory, _build_graph_memory


@pytest.fixture
def small_batch():
    train_list, _, _ = build_tnm_datasets(depth=2, max_examples=64, seed=0)
    loader = DataLoader(train_list[:8], batch_size=8, shuffle=False)
    return next(iter(loader))


def _make_model(**kwargs):
    defaults = dict(
        num_classes=4, in_dim=4, h_dim=32, num_layers=3,
        gnn_type="GCN", dropout=0.0, num_routers=8, topk_assign=2,
        assignment_tau=1.0, router_fusion="residual", residual_gamma=0.02,
        proto_update="grad", m_step_interval=1,
    )
    defaults.update(kwargs)
    return TNMRouterGraphMemory(**defaults)


def test_forward_shape(small_batch):
    model = _make_model()
    model.eval()
    with torch.no_grad():
        logits = model(small_batch)
    num_graphs = int(small_batch.batch.max().item()) + 1
    assert logits.shape == (num_graphs, 4)


def test_backward(small_batch):
    model = _make_model()
    logits = model(small_batch)
    loss = torch.nn.functional.cross_entropy(logits, small_batch.y)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"


def test_no_cross_graph_contamination():
    """Memory for graph 0 must not depend on graph 1's nodes."""
    train_list, _, _ = build_tnm_datasets(depth=2, max_examples=16, seed=0)
    loader = DataLoader(train_list[:2], batch_size=2, shuffle=False)
    batch = next(iter(loader))

    h = torch.randn(batch.x.size(0), 8)
    K = 4
    q = torch.softmax(torch.randn(batch.x.size(0), K), dim=-1)
    P_ctx = torch.randn(K, 8)
    M = _build_graph_memory(h, q, batch.batch, 2, K, P_ctx)
    assert M.shape == (2, K, 8)

    # Perturb graph 1's nodes and verify graph 0's memory is unchanged
    h2 = h.clone()
    mask_g1 = batch.batch == 1
    h2[mask_g1] = h2[mask_g1] * 100.0
    M2 = _build_graph_memory(h2, q, batch.batch, 2, K, P_ctx)
    assert torch.allclose(M[0], M2[0], atol=1e-5), "Graph 0 memory changed when graph 1 nodes perturbed"


def test_fusion_modes(small_batch):
    for fusion in ["none", "add", "residual", "concat"]:
        model = _make_model(router_fusion=fusion)
        model.eval()
        with torch.no_grad():
            logits = model(small_batch)
        assert logits.shape[1] == 4, f"fusion={fusion} gave wrong output shape"


def test_gcnii_forward(small_batch):
    model = _make_model(gnn_type="GCNII", num_layers=3, h_dim=32,
                        gcnii_alpha=0.2, gcnii_theta=1.0)
    model.eval()
    with torch.no_grad():
        logits = model(small_batch)
    assert logits.shape[1] == 4


def test_ema_update_only_on_interval():
    """EMA should only update when outer_epoch % m_step_interval == 0."""
    train_list, _, _ = build_tnm_datasets(depth=2, max_examples=64, seed=0)
    loader = DataLoader(train_list[:4], batch_size=4, shuffle=False)
    batch = next(iter(loader))

    model = _make_model(num_classes=4, in_dim=4, proto_update="ema", ema_beta=0.5, m_step_interval=5)
    model.train()

    P_before = model.proto_bank.prototypes.clone()

    # epoch 1: should NOT update (1 % 5 != 0)
    model.set_epoch(1)
    with torch.no_grad():
        model(batch)
    model.step_ema()
    P_after_1 = model.proto_bank.prototypes.clone()
    assert torch.allclose(P_before, P_after_1), "EMA updated when it should not have"

    # epoch 5: should update (5 % 5 == 0)
    model.set_epoch(5)
    with torch.no_grad():
        model(batch)
    model.step_ema()
    P_after_5 = model.proto_bank.prototypes.clone()
    assert not torch.allclose(P_before, P_after_5), "EMA did not update at m_step_interval"

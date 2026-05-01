"""Tests for TNMBaselineModel forward pass."""

import pytest
import torch
from torch_geometric.data import Batch

from datasets.tnm_dataset import build_tnm_datasets
from models.tnm.baseline import TNMBaselineModel, _build_fa_edge_index


def _make_batch(depth=3, n_samples=4, max_examples=None, seed=0):
    # depth=4 has 16 classes; need enough samples for stratified split
    if max_examples is None:
        max_examples = max(256, 32 * (2 ** depth))
    train, _, meta = build_tnm_datasets(depth=depth, max_examples=max_examples, seed=seed)
    samples = train[:n_samples]
    batch = Batch.from_data_list(samples)
    return batch, meta


class TestTNMBaselineForward:
    @pytest.mark.parametrize("gnn_type", ["GCN", "GIN", "GAT", "GGNN"])
    def test_output_shape(self, gnn_type):
        batch, meta = _make_batch(depth=3, n_samples=4)
        model = TNMBaselineModel(
            num_classes=meta["num_classes"],
            in_dim=meta["in_dim"],
            h_dim=32,
            num_layers=4,
            gnn_type=gnn_type,
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (4, meta["num_classes"])

    @pytest.mark.parametrize("depth", [2, 3, 4])
    def test_depth_variants(self, depth):
        batch, meta = _make_batch(depth=depth, n_samples=4)
        model = TNMBaselineModel(
            num_classes=meta["num_classes"],
            in_dim=meta["in_dim"],
            h_dim=32,
            num_layers=depth + 1,
            gnn_type="GCN",
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (4, meta["num_classes"])

    def test_fa_layer(self):
        batch, meta = _make_batch(depth=3, n_samples=4)
        model = TNMBaselineModel(
            num_classes=meta["num_classes"],
            in_dim=meta["in_dim"],
            h_dim=32,
            num_layers=4,
            gnn_type="GCN",
            use_fa_layer=True,
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (4, meta["num_classes"])

    def test_no_nan_in_output(self):
        batch, meta = _make_batch(depth=3, n_samples=8)
        for gnn_type in ["GCN", "GIN", "GAT", "GGNN"]:
            model = TNMBaselineModel(
                num_classes=meta["num_classes"],
                in_dim=meta["in_dim"],
                h_dim=32,
                num_layers=4,
                gnn_type=gnn_type,
            )
            model.eval()
            with torch.no_grad():
                logits = model(batch)
            assert not torch.isnan(logits).any(), f"NaN in {gnn_type} output"

    def test_training_step(self):
        batch, meta = _make_batch(depth=3, n_samples=8)
        model = TNMBaselineModel(
            num_classes=meta["num_classes"],
            in_dim=meta["in_dim"],
            h_dim=32,
            num_layers=4,
            gnn_type="GCN",
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0


class TestFAEdgeIndex:
    def test_shape(self):
        batch, meta = _make_batch(depth=3, n_samples=4)
        ei = _build_fa_edge_index(batch.root_mask, batch.batch)
        N = batch.x.size(0)
        assert ei.shape == (2, N)

    def test_all_dst_are_roots(self):
        batch, meta = _make_batch(depth=3, n_samples=4)
        ei = _build_fa_edge_index(batch.root_mask, batch.batch)
        root_indices = set(torch.where(batch.root_mask)[0].tolist())
        for dst in ei[1].tolist():
            assert dst in root_indices

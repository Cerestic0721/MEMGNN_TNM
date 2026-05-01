"""Tests for TNM dataset generation."""

import pytest
import torch

from datasets.tnm_dataset import TNMDataset, build_tnm_datasets, _build_tree_edges


class TestBuildTreeEdges:
    def test_depth1_structure(self):
        edges, leaves = _build_tree_edges(1)
        # depth=1: root=0, left=1, right=2
        assert sorted(leaves) == [1, 2]
        assert [1, 0] in edges
        assert [2, 0] in edges

    def test_depth2_node_count(self):
        edges, leaves = _build_tree_edges(2)
        # depth=2: 7 nodes, 4 leaves
        assert len(leaves) == 4
        # 6 edges (child->parent, no self-loops)
        assert len(edges) == 6

    def test_leaf_count(self):
        for depth in range(1, 6):
            _, leaves = _build_tree_edges(depth)
            assert len(leaves) == 2 ** depth


class TestTNMDataset:
    @pytest.fixture(params=[2, 3, 4])
    def ds(self, request):
        return TNMDataset(depth=request.param, max_examples=256, seed=42)

    def test_num_nodes(self, ds):
        assert ds.num_nodes == 2 ** (ds.depth + 1) - 1

    def test_num_leaves(self, ds):
        assert ds.num_leaves == 2 ** ds.depth

    def test_generate_returns_lists(self, ds):
        train, test = ds.generate()
        assert len(train) > 0
        assert len(test) > 0

    def test_split_ratio(self, ds):
        train, test = ds.generate()
        total = len(train) + len(test)
        ratio = len(train) / total
        assert 0.75 <= ratio <= 0.85

    def test_sample_shapes(self, ds):
        train, _ = ds.generate()
        sample = train[0]
        assert sample.x.shape == (ds.num_nodes, 2)
        assert sample.x.dtype == torch.long
        assert sample.root_mask.shape == (ds.num_nodes,)
        assert sample.root_mask.dtype == torch.bool
        assert sample.root_mask[0].item() is True
        assert sample.root_mask[1:].sum().item() == 0

    def test_label_range(self, ds):
        train, test = ds.generate()
        for sample in train + test:
            y = int(sample.y.item())
            assert 0 <= y < ds.num_leaves

    def test_root_feature(self, ds):
        train, _ = ds.generate()
        for sample in train[:20]:
            key = int(sample.x[0, 0].item())
            assert 1 <= key <= ds.num_leaves
            assert int(sample.x[0, 1].item()) == 0

    def test_leaf_features(self, ds):
        train, _ = ds.generate()
        sample = train[0]
        for leaf_idx in ds.leaf_indices:
            leaf_num = ds.leaf_indices.index(leaf_idx)
            assert int(sample.x[leaf_idx, 0].item()) == leaf_num + 1
            val = int(sample.x[leaf_idx, 1].item())
            assert 1 <= val <= ds.num_leaves

    def test_intermediate_features(self, ds):
        train, _ = ds.generate()
        sample = train[0]
        leaf_set = set(ds.leaf_indices)
        for i in range(1, ds.num_nodes):
            if i not in leaf_set:
                assert int(sample.x[i, 0].item()) == 0
                assert int(sample.x[i, 1].item()) == 0

    def test_label_correctness(self, ds):
        train, _ = ds.generate()
        for sample in train[:50]:
            key = int(sample.x[0, 0].item())
            # Find the leaf with leaf_num == key-1
            leaf_idx = ds.leaf_indices[key - 1]
            expected_val = int(sample.x[leaf_idx, 1].item())
            assert int(sample.y.item()) == expected_val - 1


class TestBuildTNMDatasets:
    def test_metadata_keys(self):
        _, _, meta = build_tnm_datasets(depth=2, max_examples=64, seed=0)
        for key in ["depth", "num_nodes", "num_leaves", "num_classes",
                    "in_dim", "train_size", "test_size", "total_size"]:
            assert key in meta

    def test_metadata_values(self):
        _, _, meta = build_tnm_datasets(depth=3, max_examples=128, seed=0)
        assert meta["depth"] == 3
        assert meta["num_nodes"] == 15
        assert meta["num_leaves"] == 8
        assert meta["num_classes"] == 8
        assert meta["train_size"] + meta["test_size"] == meta["total_size"]

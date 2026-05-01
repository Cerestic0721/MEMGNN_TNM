"""TNM (Tree-NeighborsMatch / DictionaryLookup) dataset generation.

Faithfully migrated from bottleneck/tasks/tree_dataset.py and
bottleneck/tasks/dictionary_lookup.py, adapted to clean PyG style.

Tree structure:
  - Full binary tree of depth `depth`.
  - num_nodes = 2**(depth+1) - 1
  - Edges: child -> parent (same as bottleneck).
  - Self-loops added via add_remaining_self_loops.
  - root_mask: [True, False, False, ...]

Node features (x: LongTensor [num_nodes, 2]):
  - root:         (selected_key, 0)
  - leaf j:       (j+1, values[j])
  - intermediate: (0, 0)

Label:
  - zero_based=True (default): y = values[selected_key-1] - 1  in [0, num_leaves-1]
  - zero_based=False:          y = values[selected_key-1]       in [1, num_leaves]
"""

from __future__ import annotations

import itertools
import math
import random
from typing import List, Tuple

import numpy as np
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Tree topology helpers
# ---------------------------------------------------------------------------

def _build_tree_edges(depth: int) -> Tuple[List[List[int]], List[int]]:
    """Return (edges, leaf_indices) for a full binary tree of given depth.

    Edges are directed child -> parent, matching bottleneck exactly.
    Node 0 is the root; nodes are numbered in level-order.
    """
    max_node_id = 2 ** (depth + 1) - 2
    edges: List[List[int]] = []
    leaf_indices: List[int] = []

    stack = [(0, max_node_id)]
    while stack:
        cur_node, max_node = stack.pop()
        if cur_node == max_node:
            leaf_indices.append(cur_node)
            continue
        left_child = cur_node + 1
        right_child = cur_node + 1 + ((max_node - cur_node) // 2)
        edges.append([left_child, cur_node])
        edges.append([right_child, cur_node])
        stack.append((right_child, max_node))
        stack.append((left_child, right_child - 1))

    return edges, leaf_indices


def _base_edge_index(edges: List[List[int]], num_nodes: int,
                     add_self_loops: bool = True) -> torch.Tensor:
    """Convert edge list to edge_index tensor, optionally adding self-loops."""
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if add_self_loops:
        edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index)
    return edge_index


# ---------------------------------------------------------------------------
# Combination generation (mirrors DictionaryLookupDataset.get_combinations)
# ---------------------------------------------------------------------------

def _get_combinations(depth: int, num_leaves: int,
                      max_examples: int, rng: random.Random):
    """Yield (selected_key, permutation) pairs.

    Mirrors bottleneck logic exactly:
      depth <= 3: sample from all permutations (up to 1000)
      depth >  3: random permutations, capped by max_examples // num_leaves
    """
    num_permutations = 1000

    if depth > 3:
        per_depth = min(
            num_permutations,
            math.factorial(num_leaves),
            max_examples // num_leaves,
        )
        permutations = [
            list(np.random.permutation(range(1, num_leaves + 1)))
            for _ in range(per_depth)
        ]
    else:
        all_perms = list(itertools.permutations(range(1, num_leaves + 1)))
        k = min(num_permutations, len(all_perms))
        permutations = rng.sample(all_perms, k)

    for perm in permutations:
        for key in range(1, num_leaves + 1):
            yield (key, list(perm))


# ---------------------------------------------------------------------------
# Single-sample builder
# ---------------------------------------------------------------------------

def _make_sample(selected_key: int, values: List[int],
                 leaf_indices: List[int], num_nodes: int,
                 edge_index: torch.Tensor,
                 zero_based_label: bool) -> Data:
    """Build one PyG Data object for a (key, permutation) combination."""
    nodes = []
    for i in range(num_nodes):
        if i == 0:
            nodes.append((selected_key, 0))
        elif i in leaf_indices:
            leaf_num = leaf_indices.index(i)
            nodes.append((leaf_num + 1, values[leaf_num]))
        else:
            nodes.append((0, 0))

    x = torch.tensor(nodes, dtype=torch.long)
    root_mask = torch.zeros(num_nodes, dtype=torch.bool)
    root_mask[0] = True

    raw_label = int(values[selected_key - 1])
    y = torch.tensor(raw_label - 1 if zero_based_label else raw_label,
                     dtype=torch.long)

    return Data(x=x, edge_index=edge_index, root_mask=root_mask, y=y)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TNMDataset:
    """Tree-NeighborsMatch dataset for a fixed depth.

    Parameters
    ----------
    depth:            Tree depth (= problem radius r).
    train_fraction:   Fraction of data used for training.
    max_examples:     Upper bound on total generated samples.
    seed:             Random seed for reproducibility.
    zero_based_label: If True (default), labels are in [0, num_leaves-1].
    add_self_loops:   Add self-loops to each graph (default True).
    """

    def __init__(
        self,
        depth: int,
        train_fraction: float = 0.8,
        max_examples: int = 32000,
        seed: int = 11,
        zero_based_label: bool = True,
        add_self_loops: bool = True,
    ):
        self.depth = depth
        self.train_fraction = train_fraction
        self.max_examples = max_examples
        self.seed = seed
        self.zero_based_label = zero_based_label

        self.num_nodes = 2 ** (depth + 1) - 1
        self.num_leaves = 2 ** depth

        edges, self.leaf_indices = _build_tree_edges(depth)
        self.base_edge_index = _base_edge_index(
            edges, self.num_nodes, add_self_loops=add_self_loops
        )

        self.num_classes = self.num_leaves
        self.in_dim = self.num_leaves  # embedding vocab size hint

    def generate(self) -> Tuple[List[Data], List[Data]]:
        """Generate and split the dataset.

        Returns
        -------
        train_list, test_list
        """
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        data_list: List[Data] = []
        for key, perm in _get_combinations(
            self.depth, self.num_leaves, self.max_examples, rng
        ):
            data_list.append(
                _make_sample(
                    key, perm,
                    self.leaf_indices, self.num_nodes,
                    self.base_edge_index,
                    self.zero_based_label,
                )
            )

        labels = [int(d.y.item()) for d in data_list]
        train_list, test_list = train_test_split(
            data_list,
            train_size=self.train_fraction,
            shuffle=True,
            stratify=labels,
            random_state=self.seed,
        )
        return train_list, test_list


def build_tnm_datasets(
    depth: int,
    train_fraction: float = 0.8,
    max_examples: int = 32000,
    seed: int = 11,
    zero_based_label: bool = True,
    add_self_loops: bool = True,
):
    """Build train/test splits for the TNM task at a given depth.

    Returns
    -------
    train_dataset : list[Data]
    test_dataset  : list[Data]
    metadata      : dict
    """
    ds = TNMDataset(
        depth=depth,
        train_fraction=train_fraction,
        max_examples=max_examples,
        seed=seed,
        zero_based_label=zero_based_label,
        add_self_loops=add_self_loops,
    )
    train_list, test_list = ds.generate()

    metadata = {
        "depth": depth,
        "num_nodes": ds.num_nodes,
        "num_leaves": ds.num_leaves,
        "num_classes": ds.num_classes,
        "in_dim": ds.in_dim,
        "train_size": len(train_list),
        "test_size": len(test_list),
        "total_size": len(train_list) + len(test_list),
        "zero_based_label": zero_based_label,
    }
    return train_list, test_list, metadata

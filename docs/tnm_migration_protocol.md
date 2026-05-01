# TNM Migration Protocol

## Source References

| Component | Source file |
|-----------|-------------|
| Tree topology | `bottleneck/tasks/tree_dataset.py` |
| Dataset generation | `bottleneck/tasks/dictionary_lookup.py` |
| GNN layers | `bottleneck/common.py` GNN_TYPE.get_layer() |
| GraphModel | `bottleneck/models/graph_model.py` |
| Training protocol | `bottleneck/experiment.py` |
| Batch sizes | `bottleneck/run-gcn-2-8.py` |
| PrototypeBank | `CMMP_clean_github/models/routing/prototype_bank.py` |
| TopKRouter | `CMMP_clean_github/models/routing/topk_router.py` |

## Key Design Decisions

### Label encoding
`zero_based_label=True` (default): `y = values[selected_key-1] - 1`, range `[0, num_leaves-1]`.  
Matches standard CrossEntropyLoss convention.

### Output dimension
`Linear(h_dim, num_classes)` — standard `out_dim`, not `out_dim+1` as in original bottleneck.  
Confirmed by user.

### Edge direction
Child → parent, matching bottleneck exactly.  
Self-loops added via `add_remaining_self_loops`.

### GNN type defaults (Figure 3 alignment)

| Type | activation | residual | layer_norm |
|------|-----------|---------|-----------|
| GCN  | True | True | True |
| GIN  | True | False | False |
| GAT  | False | True | True |
| GGNN | False | True | True |

### num_layers default
`depth + 1` — matches bottleneck `run-gcn-2-8.py`.

### Combination generation
- `depth <= 3`: sample from all permutations (up to 1000)
- `depth > 3`: random permutations, capped by `max_examples // num_leaves`

### Router: what was NOT migrated
- KL divergence loss
- Mixture pi / Dirichlet pi
- Usage balance loss
- Orthogonal regularizer
- Measure space
- Encode/decode
- Router gate

### Router graph memory invariant
`M[g, k]` is built using `scatter_add` over `batch.batch` — no cross-graph contamination.  
Root reads from its own graph's memory only.

## Verification Checklist

- [ ] `python -m compileall -q .` — no syntax errors
- [ ] `python -m pytest tests/ -q` — all tests pass
- [ ] Baseline depth=3 GCN trains without error
- [ ] Router graph memory depth=3 GCN trains without error
- [ ] Figure 3 dry-run generates correct config_plan.csv
- [ ] Router sweep dry-run generates correct config_plan.csv

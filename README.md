# MEMGNN_TNM

MEM-GNN Sparse Router applied to the Tree-NeighborsMatch (TNM) over-squashing benchmark.

## Overview

Two-phase project:

**Phase 1 — Baseline reproduction**  
Faithfully reproduce bottleneck Figure 3 baselines (GCN / GIN / GAT / GGNN across depths 2–8).

**Phase 2 — Sparse Router Memory**  
Integrate per-graph prototype memory to test whether it alleviates root bottleneck (over-squashing) without requiring a Fully-Adjacent layer.

## Task

Full binary tree of depth `d`. Root holds a query key; leaves hold key-value pairs. The model must route the correct leaf value to the root for classification. Requires propagating information across `d` hops — a direct test of over-squashing.

## Architecture

```
X [N, 2]
-> key_emb + val_emb -> H [N, h_dim]
-> GNN x num_layers (default: depth+1)
-> root readout
-> Linear(h_dim, num_classes)
```

Router variant (router_graph_memory):

```
GNN -> H_all
M[g, k] = weighted_mean_{i in g}(H[i], q[i,k])   # per-graph memory
root_ctx[g] = sum_k root_q[g,k] * M[g,k]
fuse(root_h, root_ctx) -> classifier
```

## Models

| Model | Description |
|-------|-------------|
| `baseline` | Faithful port of bottleneck GraphModel |
| `router_post_gnn` | Router applied after GNN, root reads global context (sanity check) |
| `router_graph_memory` | Per-graph memory slots; root reads from them (main approach) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify
python -m compileall -q .
python -m pytest tests/ -q

# Baseline training (depth 3, GCN)
python train_tnm.py --model baseline --depth 3 --gnn_type GCN \
    --epochs 500 --batch_size 64 --save_dir results/baseline_d3_GCN

# Router training
python train_tnm.py --model router_graph_memory --depth 3 --gnn_type GCN \
    --num_routers 32 --topk_router 4 --save_dir results/router_d3_GCN

# Figure 3 dry-run (check config plan)
python experiments/run_tnm_figure3.py --depths 2,3 --gnn_types GCN --dry-run

# Router sweep dry-run
python experiments/run_tnm_router_sweep.py --pilot --dry-run
```

## Training Protocol

Mirrors bottleneck/experiment.py:
- Adam lr=0.001
- ReduceLROnPlateau(mode=max, factor=0.5, patience=10) on train accuracy
- EarlyStopping on train accuracy, patience=20

## Outputs

Each run writes to `--save_dir`:
- `per_epoch.csv` — loss/acc per epoch
- `final_metrics.json` — summary metrics
- `per_run.csv` — single-row for sweep aggregation

## Validate

```bash
python -m compileall -q .
python -m pytest tests/ -q
python train_tnm.py --model baseline --depth 3 --gnn_type GCN \
    --epochs 2 --batch_size 16 --max_examples 128 \
    --save_dir results/debug_baseline
python train_tnm.py --model router_graph_memory --depth 3 --gnn_type GCN \
    --epochs 2 --batch_size 16 --max_examples 128 \
    --num_routers 16 --topk_router 4 \
    --save_dir results/debug_router
python experiments/run_tnm_figure3.py --depths 2,3 --gnn_types GCN --dry-run
python experiments/run_tnm_router_sweep.py --pilot --dry-run
```

## Reference

- [Alon & Yahav (2021) — On the Bottleneck of Graph Neural Networks](https://arxiv.org/abs/2006.05205)
- Original bottleneck code: `bottleneck/tasks/` and `bottleneck/models/`

"""train_tnm.py — Training entry point for MEMGNN_TNM.

Supports three model variants:
  baseline            — TNMBaselineModel (bottleneck faithful port)
  router_post_gnn     — TNMRouterPostGNN (sanity check)
  router_graph_memory — TNMRouterGraphMemory (main approach)

Outputs (written to --save_dir):
  per_epoch.csv   — loss/acc per epoch
  final_metrics.json
  per_run.csv     — single-row summary (for sweep aggregation)

Training protocol mirrors bottleneck/experiment.py:
  Adam lr=0.001
  ReduceLROnPlateau(mode=max, factor=0.5, patience=10)
  EarlyStopping on TRAIN accuracy, patience=20
  eval_every=1 (simplified from bottleneck's 100-step eval)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from datasets.tnm_dataset import build_tnm_datasets
from models.tnm.baseline import TNMBaselineModel
from models.tnm.router_graph_memory import TNMRouterGraphMemory
from models.tnm.router_post_gnn import TNMRouterPostGNN


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train MEMGNN_TNM model")

    # Task
    p.add_argument("--depth",        type=int,   default=3)
    p.add_argument("--max_examples", type=int,   default=32000)
    p.add_argument("--seed",         type=int,   default=11)

    # Model
    p.add_argument("--model",     choices=["baseline", "router_post_gnn", "router_graph_memory"],
                   default="baseline")
    p.add_argument("--gnn_type",  choices=["GCN", "GIN", "GAT", "GGNN", "GCNII"], default="GCN")
    p.add_argument("--h_dim",     type=int,   default=32)
    p.add_argument("--num_layers",type=int,   default=None,
                   help="Default: depth+1 (bottleneck convention)")
    p.add_argument("--use_fa_layer", action="store_true",
                   help="Replace last GNN layer with FA layer (baseline only)")

    # GCNII-specific params (only used when --gnn_type GCNII)
    p.add_argument("--gcnii_alpha",          type=float, default=0.1)
    p.add_argument("--gcnii_theta",          type=float, default=0.5)
    p.add_argument("--gcnii_shared_weights", action="store_true", default=True)
    p.add_argument("--gcnii_dropout",        type=float, default=0.0)

    # Router params (router_* models only)
    p.add_argument("--num_routers",  type=int,   default=32)
    p.add_argument("--topk_router",  type=int,   default=4)
    p.add_argument("--tau",          type=float, default=1.0)
    p.add_argument("--ema_beta",     type=float, default=0.9)
    p.add_argument("--update_mode",  choices=["grad", "ema"], default="ema")
    p.add_argument("--init_mode",    choices=["default", "gaussian_normalized",
                                              "gaussian_scaled", "qr_orthogonal"],
                   default="default")
    p.add_argument("--fusion",       choices=["add", "residual", "gate"], default="residual")
    p.add_argument("--memory_from_all", action="store_true",
                   help="Build memory from all nodes (router_graph_memory only)")

    # Measure space params (router_graph_memory only)
    p.add_argument("--use_measure_space",      action="store_true",
                   help="Apply frozen orthogonal transform before prototype similarity")
    p.add_argument("--measure_transform_type",
                   choices=["identity", "frozen_qr_orthogonal", "frozen_hadamard_sign"],
                   default="frozen_qr_orthogonal")
    p.add_argument("--measure_apply_mode",
                   choices=["none", "route_only", "context_only", "route_and_context"],
                   default="route_only")
    p.add_argument("--measure_seed", type=int, default=42)

    # Training
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience",     type=int,   default=20,
                   help="EarlyStopping patience on train accuracy")
    p.add_argument("--lr_patience",  type=int,   default=10,
                   help="ReduceLROnPlateau patience")
    p.add_argument("--lr_factor",    type=float, default=0.5)

    # I/O
    p.add_argument("--save_dir",     type=str,   default="results/debug")
    p.add_argument("--device",       type=str,   default="auto")
    p.add_argument("--quiet",        action="store_true")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args, num_classes: int, in_dim: int) -> nn.Module:
    num_layers = args.num_layers if args.num_layers is not None else args.depth + 1

    if args.model == "baseline":
        return TNMBaselineModel(
            num_classes=num_classes,
            in_dim=in_dim,
            h_dim=args.h_dim,
            num_layers=num_layers,
            gnn_type=args.gnn_type,
            use_fa_layer=args.use_fa_layer,
        )

    router_kwargs = dict(
        num_classes=num_classes,
        in_dim=in_dim,
        h_dim=args.h_dim,
        num_layers=num_layers,
        gnn_type=args.gnn_type,
        num_routers=args.num_routers,
        topk=args.topk_router,
        tau=args.tau,
        ema_beta=args.ema_beta,
        update_mode=args.update_mode,
        init_mode=args.init_mode,
        fusion=args.fusion,
    )

    if args.model == "router_post_gnn":
        return TNMRouterPostGNN(**router_kwargs)

    if args.model == "router_graph_memory":
        return TNMRouterGraphMemory(
            **router_kwargs,
            memory_from_all=args.memory_from_all,
            gcnii_alpha=args.gcnii_alpha,
            gcnii_theta=args.gcnii_theta,
            gcnii_shared_weights=args.gcnii_shared_weights,
            gcnii_dropout=args.gcnii_dropout,
            use_measure_space=args.use_measure_space,
            measure_transform_type=args.measure_transform_type,
            measure_apply_mode=args.measure_apply_mode,
            measure_seed=args.measure_seed,
        )

    raise ValueError(args.model)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = nn.functional.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * logits.size(0)
        total_correct += int((logits.argmax(-1) == batch.y).sum())
        total_n += logits.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_correct = total_n = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        total_correct += int((logits.argmax(-1) == batch.y).sum())
        total_n += logits.size(0)
    return total_correct / total_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    # Data
    train_list, test_list, meta = build_tnm_datasets(
        depth=args.depth,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_list,  batch_size=args.batch_size, shuffle=False)

    # Model
    model = build_model(args, meta["num_classes"], meta["in_dim"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor,
        patience=args.lr_patience,
    )

    # Output dir
    os.makedirs(args.save_dir, exist_ok=True)
    epoch_csv_path = os.path.join(args.save_dir, "per_epoch.csv")

    epoch_rows = []
    best_train_acc = 0.0
    best_test_acc  = 0.0
    no_improve     = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_acc = eval_epoch(model, test_loader, device)
        scheduler.step(train_acc)

        row = dict(
            epoch=epoch,
            train_loss=round(train_loss, 6),
            train_acc=round(train_acc, 6),
            test_acc=round(test_acc, 6),
            lr=optimizer.param_groups[0]["lr"],
        )
        epoch_rows.append(row)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_test_acc  = test_acc
            no_improve = 0
        else:
            no_improve += 1

        if not args.quiet:
            print(
                f"[{epoch:4d}] loss={train_loss:.4f}  "
                f"train={train_acc:.4f}  test={test_acc:.4f}  "
                f"best_test={best_test_acc:.4f}  no_imp={no_improve}"
            )

        if no_improve >= args.patience:
            if not args.quiet:
                print(f"Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # Write per_epoch.csv
    with open(epoch_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    # final_metrics.json
    args_dict = {k: v for k, v in vars(args).items()
                 if k not in ("save_dir", "device", "quiet")}
    final = dict(
        best_train_acc=round(best_train_acc, 6),
        best_test_acc=round(best_test_acc, 6),
        final_test_acc=round(epoch_rows[-1]["test_acc"], 6),
        epochs_run=len(epoch_rows),
        elapsed_s=round(elapsed, 2),
        **args_dict,
    )
    # override num_layers with resolved value
    final["num_layers"] = args.num_layers if args.num_layers else args.depth + 1
    with open(os.path.join(args.save_dir, "final_metrics.json"), "w") as f:
        json.dump(final, f, indent=2)

    # per_run.csv (single row, for sweep aggregation)
    per_run_path = os.path.join(args.save_dir, "per_run.csv")
    write_header = not os.path.exists(per_run_path)
    with open(per_run_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(final)

    print(
        f"\nDone. best_train={best_train_acc:.4f}  "
        f"best_test={best_test_acc:.4f}  "
        f"elapsed={elapsed:.1f}s"
    )
    return final


if __name__ == "__main__":
    main()

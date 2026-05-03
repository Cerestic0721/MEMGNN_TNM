"""train_tnm.py — Training entry point for MEMGNN_TNM.

Supports three model variants:
  baseline            — TNMBaselineModel (bottleneck faithful port)
  router_post_gnn     — TNMRouterPostGNN (sanity check)
  router_graph_memory — TNMRouterGraphMemory (main approach)

Training protocol:
  Adam lr=0.001
  ReduceLROnPlateau(mode=max, factor=lr_factor, patience=lr_patience)
  EarlyStopping on TRAIN accuracy, patience=patience
  eval_every: number of mini-batch steps between evaluations
              (outer loop = epochs, inner loop = eval_every steps)

Outputs (written to --save_dir):
  per_epoch.csv   — loss/acc per eval point
  final_metrics.json
  per_run.csv     — single-row summary (for sweep aggregation)
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

# Ensure stdout is line-buffered even when redirected to a file
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from datasets.tnm_dataset import build_tnm_datasets
from models.tnm.baseline import TNMBaselineModel
from models.tnm.router_graph_memory import TNMRouterGraphMemory
from models.tnm.router_post_gnn import TNMRouterPostGNN


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def _gamma_or_learnable(v):
    """Accept a float, -1, or the string 'learnable' (all map to learnable gamma)."""
    if isinstance(v, float):
        return v
    if str(v).lower() == "learnable":
        return -1.0
    return float(v)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train MEMGNN_TNM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task
    p.add_argument("--depth",        type=int,   default=4)
    p.add_argument("--max_examples", type=int,   default=16000)
    p.add_argument("--seed",         type=int,   default=11)

    # Model
    p.add_argument("--model",     choices=["baseline", "router_post_gnn", "router_graph_memory"],
                   default="router_graph_memory")
    p.add_argument("--gnn_type",  choices=["GCN", "GIN", "GAT", "GGNN", "GCNII"], default="GCNII")
    p.add_argument("--h_dim",     type=int,   default=512)
    p.add_argument("--num_layers",type=int,   default=None,
                   help="Default: depth+1 for baseline, 16 for router_graph_memory")
    p.add_argument("--dropout",   type=float, default=0.7)
    p.add_argument("--use_fa_layer", action="store_true",
                   help="Replace last GNN layer with FA layer (baseline only)")

    # GCNII params
    p.add_argument("--gcnii_alpha",          type=float, default=0.2)
    p.add_argument("--gcnii_theta",          type=float, default=1.0)
    p.add_argument("--gcnii_shared_weights", type=_bool, default=True, metavar="BOOL")
    p.add_argument("--gcnii_dropout",        type=float, default=None,
                   help="GCNII layer dropout; defaults to --dropout if not set")

    # Router params (router_* models only)
    p.add_argument("--num_routers",      type=int,   default=128)
    p.add_argument("--topk_assign",      type=int,   default=16)
    p.add_argument("--assignment_tau",   type=float, default=1.0)
    p.add_argument("--assign_type",      choices=["dot", "bilinear", "linear_proj"], default="dot")
    p.add_argument("--router_fusion",    choices=["none", "add", "residual", "concat"],
                   default="residual")
    p.add_argument("--residual_gamma",   type=_gamma_or_learnable, default=0.1,
                   help="Residual fusion gamma; -1 or 'learnable' for sigmoid-parameterized learnable gamma")
    p.add_argument("--proto_update",     choices=["grad", "ema"], default="ema")
    p.add_argument("--ema_beta",         type=float, default=0.03)
    p.add_argument("--ema_init",         choices=["random", "sample_h", "farthest_h", "kmeans_h"],
                   default="sample_h",
                   help="Prototype initialization mode for EMA update")
    p.add_argument("--ema_normalize_proto", type=_bool, default=True, metavar="BOOL")
    p.add_argument("--ema_reinit_dead",  type=_bool, default=False, metavar="BOOL")
    p.add_argument("--ema_dead_threshold", type=float, default=1e-4)
    p.add_argument("--ema_reinit_patience", type=int, default=20)
    p.add_argument("--proto_init_mode",  choices=["default", "gaussian_normalized",
                                                   "gaussian_scaled", "qr_orthogonal"],
                   default="default")
    p.add_argument("--m_step_interval",  type=int,   default=1,
                   help="EMA update every N outer epochs (1 = every epoch)")
    p.add_argument("--memory_from_all",  action="store_true",
                   help="Build memory from all nodes (router_graph_memory only)")

    # Disabled regularizers (kept for CLI compatibility, always off)
    p.add_argument("--use_mixture_pi",              type=_bool, default=False, metavar="BOOL")
    p.add_argument("--use_dirichlet_pi",            type=_bool, default=False, metavar="BOOL")
    p.add_argument("--lambda_pi",                   type=float, default=0.0)
    p.add_argument("--use_usage_balance_loss",      type=_bool, default=False, metavar="BOOL")
    p.add_argument("--use_e_step_kl_loss",          type=_bool, default=False, metavar="BOOL")
    p.add_argument("--use_proto_orthogonal_regularizer", type=_bool, default=False, metavar="BOOL")
    p.add_argument("--use_proto_orthogonal_constraint",  type=_bool, default=False, metavar="BOOL")

    # Measure space params
    p.add_argument("--use_measure_space",      action="store_true")
    p.add_argument("--measure_transform_type",
                   choices=["identity", "frozen_qr_orthogonal", "frozen_hadamard_sign"],
                   default="frozen_qr_orthogonal")
    p.add_argument("--measure_apply_mode",
                   choices=["none", "route_only", "context_only", "route_and_context"],
                   default="route_only")
    p.add_argument("--measure_seed", type=int, default=42)

    # Training
    p.add_argument("--epochs",       type=int,   default=50000)
    p.add_argument("--eval_every",   type=int,   default=100,
                   help="Number of mini-batch steps between evaluations")
    p.add_argument("--batch_size",   type=int,   default=1024)
    p.add_argument("--lr",           type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience",     type=int,   default=20,
                   help="EarlyStopping patience (in eval points) on train accuracy")
    p.add_argument("--lr_patience",  type=int,   default=10,
                   help="ReduceLROnPlateau patience (in eval points)")
    p.add_argument("--lr_factor",    type=float, default=0.5)
    p.add_argument("--accum_grad",   type=int,   default=1,
                   help="Gradient accumulation steps")

    # I/O
    p.add_argument("--save_dir",     type=str,   default="results/debug")
    p.add_argument("--device",       type=str,   default="auto")
    p.add_argument("--quiet",        action="store_true")
    p.add_argument("--log_interval", type=int,   default=1,
                   help="Print every N eval points")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args, num_classes: int, in_dim: int) -> nn.Module:
    if args.num_layers is not None:
        num_layers = args.num_layers
    elif args.model == "router_graph_memory":
        num_layers = 16
    else:
        num_layers = args.depth + 1

    gcnii_dropout = args.gcnii_dropout if args.gcnii_dropout is not None else args.dropout
    residual_gamma = None if args.residual_gamma < 0 else args.residual_gamma

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
        dropout=args.dropout,
        num_routers=args.num_routers,
        topk_assign=args.topk_assign,
        assignment_tau=args.assignment_tau,
        assign_type=args.assign_type,
        router_fusion=args.router_fusion,
        residual_gamma=residual_gamma,
        proto_update=args.proto_update,
        ema_beta=args.ema_beta,
        ema_init=args.ema_init,
        ema_normalize_proto=args.ema_normalize_proto,
        ema_reinit_dead=args.ema_reinit_dead,
        ema_dead_threshold=args.ema_dead_threshold,
        ema_reinit_patience=args.ema_reinit_patience,
        proto_init_mode=args.proto_init_mode,
        m_step_interval=args.m_step_interval,
        gcnii_alpha=args.gcnii_alpha,
        gcnii_theta=args.gcnii_theta,
        gcnii_shared_weights=args.gcnii_shared_weights,
        use_measure_space=args.use_measure_space,
        measure_transform_type=args.measure_transform_type,
        measure_apply_mode=args.measure_apply_mode,
        measure_seed=args.measure_seed,
    )

    if args.model == "router_post_gnn":
        # router_post_gnn uses legacy SparseRouterBank interface
        from models.router.router_bank import SparseRouterBank
        return TNMRouterPostGNN(
            num_classes=num_classes,
            in_dim=in_dim,
            h_dim=args.h_dim,
            num_layers=num_layers,
            gnn_type=args.gnn_type,
            num_routers=args.num_routers,
            topk=args.topk_assign,
            tau=args.assignment_tau,
            ema_beta=args.ema_beta,
            update_mode=args.proto_update,
            init_mode=args.proto_init_mode,
            fusion=args.router_fusion,
        )

    if args.model == "router_graph_memory":
        return TNMRouterGraphMemory(
            **router_kwargs,
            memory_from_all=args.memory_from_all,
        )

    raise ValueError(args.model)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_steps(model, loader_iter, loader, optimizer, device, n_steps, accum_grad):
    """Run exactly n_steps mini-batch updates. Returns (total_loss, total_correct, total_n)."""
    model.train()
    total_loss = total_correct = total_n = 0
    optimizer.zero_grad()
    accum_count = 0

    for _ in range(n_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = batch.to(device)
        logits = model(batch)
        loss = nn.functional.cross_entropy(logits, batch.y)
        (loss / accum_grad).backward()

        accum_count += 1

        total_loss += float(loss) * logits.size(0)
        total_correct += int((logits.argmax(-1) == batch.y).sum())
        total_n += logits.size(0)

        if accum_count % accum_grad == 0:
            optimizer.step()
            optimizer.zero_grad()

    if accum_count % accum_grad != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1), loader_iter


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_correct = total_n = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        total_correct += int((logits.argmax(-1) == batch.y).sum())
        total_n += logits.size(0)
    return total_correct / max(total_n, 1)


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

    train_list, test_list, meta = build_tnm_datasets(
        depth=args.depth,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_list,  batch_size=args.batch_size, shuffle=False)

    model = build_model(args, meta["num_classes"], meta["in_dim"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=1e-6,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    epoch_csv_path = os.path.join(args.save_dir, "per_epoch.csv")

    epoch_rows = []
    best_train_acc = 0.0
    best_test_acc  = 0.0
    no_improve     = 0
    t0 = time.time()

    loader_iter = iter(train_loader)
    eval_point = 0

    for outer_epoch in range(1, args.epochs + 1):
        # Notify model of current epoch (for EMA m_step_interval)
        if hasattr(model, "set_epoch"):
            model.set_epoch(outer_epoch)

        train_loss, train_acc, loader_iter = _run_steps(
            model, loader_iter, train_loader, optimizer, device,
            args.eval_every, args.accum_grad,
        )

        # Epoch-level EMA update: full pass over train_loader, one prototype commit.
        if hasattr(model, "epoch_ema_update"):
            model.epoch_ema_update(train_loader, device, quiet=args.quiet)

        test_acc = eval_epoch(model, test_loader, device)
        eval_point += 1

        scheduler.step(train_acc)

        row = dict(
            eval_point=eval_point,
            outer_epoch=outer_epoch,
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

        if not args.quiet and eval_point % args.log_interval == 0:
            print(
                f"[ep {outer_epoch:5d}] loss={train_loss:.4f}  "
                f"train={train_acc:.4f}  test={test_acc:.4f}  "
                f"best_test={best_test_acc:.4f}  no_imp={no_improve}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        if no_improve >= args.patience:
            if not args.quiet:
                print(f"Early stop at outer_epoch {outer_epoch}", flush=True)
            break

    elapsed = time.time() - t0

    with open(epoch_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    # Resolve num_layers for reporting
    if args.num_layers is not None:
        resolved_layers = args.num_layers
    elif args.model == "router_graph_memory":
        resolved_layers = 16
    else:
        resolved_layers = args.depth + 1

    args_dict = {k: v for k, v in vars(args).items()
                 if k not in ("save_dir", "device", "quiet")}
    final = dict(
        best_train_acc=round(best_train_acc, 6),
        best_test_acc=round(best_test_acc, 6),
        final_test_acc=round(epoch_rows[-1]["test_acc"], 6),
        eval_points_run=len(epoch_rows),
        elapsed_s=round(elapsed, 2),
        **args_dict,
    )
    final["num_layers"] = resolved_layers

    with open(os.path.join(args.save_dir, "final_metrics.json"), "w") as f:
        json.dump(final, f, indent=2)

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
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return final


if __name__ == "__main__":
    main()

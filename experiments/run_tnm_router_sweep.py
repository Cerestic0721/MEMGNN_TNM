"""experiments/run_tnm_router_sweep.py — Router hyperparameter sweep.

Pilot mode: depths 3,4,5 x gnn_types GCN,GIN x seeds 0,1,2
Sweeps 5 main router parameters.

Usage:
  python experiments/run_tnm_router_sweep.py --pilot
  python experiments/run_tnm_router_sweep.py --pilot --dry-run
  python experiments/run_tnm_router_sweep.py --model router_graph_memory
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from itertools import product
from typing import List

_PILOT_DEPTHS     = [3, 4, 5]
_PILOT_GNN_TYPES  = ["GCN", "GIN"]
_PILOT_SEEDS      = [0, 1, 2]

# 5 main router parameters to sweep
_SWEEP_GRID = {
    "num_routers": [16, 32, 64],
    "topk_router": [2, 4, 8],
    "tau":         [0.5, 1.0, 2.0],
    "ema_beta":    [0.9, 0.99],
    "fusion":      ["add", "residual", "gate"],
}

_BATCH_SIZE = {2: 64, 3: 64, 4: 64, 5: 64, 6: 32, 7: 16, 8: 8}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      choices=["router_post_gnn", "router_graph_memory"],
                   default="router_graph_memory")
    p.add_argument("--pilot",      action="store_true",
                   help="Use pilot depths/gnn_types/seeds")
    p.add_argument("--depths",     type=str, default=None)
    p.add_argument("--gnn_types",  type=str, default=None)
    p.add_argument("--seeds",      type=str, default=None)
    p.add_argument("--epochs",     type=int, default=500)
    p.add_argument("--h_dim",      type=int, default=32)
    p.add_argument("--max_examples", type=int, default=32000)
    p.add_argument("--results_dir",type=str, default="results/router_sweep")
    p.add_argument("--dry-run",    action="store_true")
    # Fixed router params (non-swept)
    p.add_argument("--update_mode", choices=["grad", "ema"], default="ema")
    p.add_argument("--init_mode",   default="default")
    return p.parse_args()


def build_configs(depths, gnn_types, seeds, args):
    configs = []
    # Default router params (center of sweep grid)
    default_router = dict(
        num_routers=32, topk_router=4, tau=1.0,
        ema_beta=0.9, fusion="residual",
    )

    # One-at-a-time sweep: vary one param, fix others at default
    sweep_configs = [default_router.copy()]  # baseline router config
    for param, values in _SWEEP_GRID.items():
        for val in values:
            if val == default_router[param]:
                continue
            cfg = default_router.copy()
            cfg[param] = val
            sweep_configs.append(cfg)

    for router_cfg, depth, gnn_type, seed in product(
        sweep_configs, depths, gnn_types, seeds
    ):
        num_layers = depth + 1
        batch_size = _BATCH_SIZE.get(depth, 32)
        tag = (
            f"d{depth}_{gnn_type}_s{seed}"
            f"_K{router_cfg['num_routers']}"
            f"_k{router_cfg['topk_router']}"
            f"_tau{router_cfg['tau']}"
            f"_b{router_cfg['ema_beta']}"
            f"_{router_cfg['fusion']}"
        )
        save_dir = os.path.join(args.results_dir, tag)
        cfg = dict(
            depth=depth,
            gnn_type=gnn_type,
            seed=seed,
            num_layers=num_layers,
            batch_size=batch_size,
            h_dim=args.h_dim,
            epochs=args.epochs,
            max_examples=args.max_examples,
            save_dir=save_dir,
            update_mode=args.update_mode,
            init_mode=args.init_mode,
            **router_cfg,
        )
        configs.append(cfg)
    return configs


def run_config(cfg: dict, model: str, python: str = sys.executable) -> dict | None:
    cmd = [
        python, "train_tnm.py",
        "--model",        model,
        "--depth",        str(cfg["depth"]),
        "--gnn_type",     cfg["gnn_type"],
        "--seed",         str(cfg["seed"]),
        "--num_layers",   str(cfg["num_layers"]),
        "--batch_size",   str(cfg["batch_size"]),
        "--h_dim",        str(cfg["h_dim"]),
        "--epochs",       str(cfg["epochs"]),
        "--max_examples", str(cfg["max_examples"]),
        "--num_routers",  str(cfg["num_routers"]),
        "--topk_router",  str(cfg["topk_router"]),
        "--tau",          str(cfg["tau"]),
        "--ema_beta",     str(cfg["ema_beta"]),
        "--fusion",       cfg["fusion"],
        "--update_mode",  cfg["update_mode"],
        "--init_mode",    cfg["init_mode"],
        "--save_dir",     cfg["save_dir"],
    ]
    print(f"  Running: {os.path.basename(cfg['save_dir'])}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [FAILED] {os.path.basename(cfg['save_dir'])}")
        return None

    metrics_path = os.path.join(cfg["save_dir"], "final_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


def write_summary(all_results: List[dict], results_dir: str):
    if not all_results:
        return
    per_run_path = os.path.join(results_dir, "per_run.csv")
    with open(per_run_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nper_run.csv written to {per_run_path} ({len(all_results)} rows)")


def main():
    args = parse_args()

    if args.pilot:
        depths    = _PILOT_DEPTHS
        gnn_types = _PILOT_GNN_TYPES
        seeds     = _PILOT_SEEDS
    else:
        depths    = [int(x) for x in args.depths.split(",")]   if args.depths    else _PILOT_DEPTHS
        gnn_types = args.gnn_types.split(",")                  if args.gnn_types else _PILOT_GNN_TYPES
        seeds     = [int(x) for x in args.seeds.split(",")]    if args.seeds     else _PILOT_SEEDS

    configs = build_configs(depths, gnn_types, seeds, args)
    os.makedirs(args.results_dir, exist_ok=True)

    plan_path = os.path.join(args.results_dir, "config_plan.csv")
    with open(plan_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(configs[0].keys()))
        writer.writeheader()
        writer.writerows(configs)
    print(f"Config plan: {len(configs)} runs -> {plan_path}")

    if args.dry_run:
        print("Dry-run mode: exiting without training.")
        return

    all_results = []
    for cfg in configs:
        metrics = run_config(cfg, args.model)
        if metrics:
            all_results.append(metrics)

    write_summary(all_results, args.results_dir)


if __name__ == "__main__":
    main()

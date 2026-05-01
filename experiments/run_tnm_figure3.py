"""experiments/run_tnm_figure3.py — Reproduce bottleneck Figure 3 baselines.

Sweeps depths 2-8 x GNN types {GCN, GIN, GAT, GGNN}, calling train_tnm.py
as a subprocess for each configuration.

Usage:
  python experiments/run_tnm_figure3.py                        # full sweep
  python experiments/run_tnm_figure3.py --depths 2,3 --gnn_types GCN,GIN
  python experiments/run_tnm_figure3.py --dry-run              # print config plan only
  python experiments/run_tnm_figure3.py --depths 2,3 --dry-run

Outputs:
  results/figure3/config_plan.csv   — all planned configs
  results/figure3/per_run.csv       — aggregated results (one row per run)
  results/figure3/summary.csv       — mean/std per (depth, gnn_type) across seeds
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

# Batch sizes from bottleneck/run-gcn-2-8.py
_BATCH_SIZE = {2: 64, 3: 64, 4: 64, 5: 64, 6: 32, 7: 16, 8: 8}
_DEFAULT_DEPTHS     = [2, 3, 4, 5, 6, 7, 8]
_DEFAULT_GNN_TYPES  = ["GCN", "GIN", "GAT", "GGNN"]
_DEFAULT_SEEDS      = [0, 1, 2, 3, 4]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depths",     type=str, default=None,
                   help="Comma-separated depths, e.g. 2,3,4")
    p.add_argument("--gnn_types",  type=str, default=None,
                   help="Comma-separated GNN types, e.g. GCN,GIN")
    p.add_argument("--seeds",      type=str, default=None,
                   help="Comma-separated seeds, e.g. 0,1,2")
    p.add_argument("--epochs",     type=int, default=500)
    p.add_argument("--h_dim",      type=int, default=32)
    p.add_argument("--max_examples", type=int, default=32000)
    p.add_argument("--results_dir",type=str, default="results/figure3")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print config plan CSV and exit without training")
    p.add_argument("--use_fa_layer", action="store_true")
    return p.parse_args()


def build_configs(depths, gnn_types, seeds, args):
    configs = []
    for depth, gnn_type, seed in product(depths, gnn_types, seeds):
        num_layers = depth + 1
        batch_size = _BATCH_SIZE.get(depth, 32)
        save_dir = os.path.join(
            args.results_dir, f"d{depth}_{gnn_type}_s{seed}"
        )
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
            use_fa_layer=args.use_fa_layer,
        )
        configs.append(cfg)
    return configs


def run_config(cfg: dict, python: str = sys.executable) -> dict | None:
    cmd = [
        python, "train_tnm.py",
        "--model",        "baseline",
        "--depth",        str(cfg["depth"]),
        "--gnn_type",     cfg["gnn_type"],
        "--seed",         str(cfg["seed"]),
        "--num_layers",   str(cfg["num_layers"]),
        "--batch_size",   str(cfg["batch_size"]),
        "--h_dim",        str(cfg["h_dim"]),
        "--epochs",       str(cfg["epochs"]),
        "--max_examples", str(cfg["max_examples"]),
        "--save_dir",     cfg["save_dir"],
    ]
    if cfg.get("use_fa_layer"):
        cmd.append("--use_fa_layer")

    print(f"  Running: depth={cfg['depth']} gnn={cfg['gnn_type']} seed={cfg['seed']}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [FAILED] depth={cfg['depth']} gnn={cfg['gnn_type']} seed={cfg['seed']}")
        return None

    metrics_path = os.path.join(cfg["save_dir"], "final_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


def write_summary(all_results: List[dict], results_dir: str):
    if not all_results:
        return

    # per_run.csv
    per_run_path = os.path.join(results_dir, "per_run.csv")
    with open(per_run_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    # summary.csv: mean/std per (depth, gnn_type)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in all_results:
        key = (r["depth"], r["gnn_type"])
        groups[key].append(r["best_test_acc"])

    summary_rows = []
    for (depth, gnn_type), accs in sorted(groups.items()):
        import statistics
        mean = statistics.mean(accs)
        std  = statistics.stdev(accs) if len(accs) > 1 else 0.0
        summary_rows.append(dict(
            depth=depth, gnn_type=gnn_type,
            n_seeds=len(accs),
            mean_test_acc=round(mean, 4),
            std_test_acc=round(std, 4),
        ))

    summary_path = os.path.join(results_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary written to {summary_path}")
    print(f"{'depth':>6}  {'gnn':>6}  {'mean':>8}  {'std':>8}")
    for row in summary_rows:
        print(f"{row['depth']:>6}  {row['gnn_type']:>6}  "
              f"{row['mean_test_acc']:>8.4f}  {row['std_test_acc']:>8.4f}")


def main():
    args = parse_args()

    depths    = [int(x) for x in args.depths.split(",")]    if args.depths    else _DEFAULT_DEPTHS
    gnn_types = args.gnn_types.split(",")                   if args.gnn_types else _DEFAULT_GNN_TYPES
    seeds     = [int(x) for x in args.seeds.split(",")]     if args.seeds     else _DEFAULT_SEEDS

    configs = build_configs(depths, gnn_types, seeds, args)
    os.makedirs(args.results_dir, exist_ok=True)

    # Write config plan
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
        metrics = run_config(cfg)
        if metrics:
            all_results.append(metrics)

    write_summary(all_results, args.results_dir)


if __name__ == "__main__":
    main()

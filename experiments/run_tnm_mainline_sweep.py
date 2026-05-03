"""experiments/run_tnm_mainline_sweep.py — Pilot sweep for TNM mainline.

Runs a limited set of configurations (no Cartesian product explosion).
Supports --dry-run, --resume, --overwrite, --device, --max-runs, --depths, --seeds.

Usage examples:
  python experiments/run_tnm_mainline_sweep.py --dry-run
  python experiments/run_tnm_mainline_sweep.py --depths 4 5 --device cuda:0
  python experiments/run_tnm_mainline_sweep.py --max-runs 4 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

PYTHON = sys.executable
TRAIN_SCRIPT = str(Path(__file__).parent.parent / "train_tnm.py")

# Shared mainline defaults
MAINLINE_BASE = dict(
    model="router_graph_memory",
    gnn_type="GCNII",
    num_layers=16,
    h_dim=512,
    dropout=0.7,
    gcnii_alpha=0.2,
    gcnii_theta=1.0,
    gcnii_shared_weights="True",
    num_routers=128,
    topk_assign=16,
    assignment_tau=1.0,
    assign_type="dot",
    router_fusion="residual",
    residual_gamma=0.02,
    proto_update="ema",
    ema_beta=0.03,
    ema_normalize_proto="True",
    ema_reinit_dead="False",
    m_step_interval=20,
    proto_init_mode="default",
    max_examples=16000,
    batch_size=1024,
    lr=0.001,
    weight_decay=0.0,
    epochs=50000,
    eval_every=100,
    patience=20,
    lr_patience=10,
    lr_factor=0.5,
)

# Baseline configs
BASELINE_GCN = dict(model="baseline", gnn_type="GCN", num_layers=None,
                    h_dim=32, dropout=0.0)
BASELINE_GCN_FA = dict(model="baseline", gnn_type="GCN", num_layers=None,
                       h_dim=32, dropout=0.0, use_fa_layer=True)
BASELINE_GCNII = dict(model="baseline", gnn_type="GCNII", num_layers=16,
                      h_dim=512, dropout=0.7, gcnii_alpha=0.2, gcnii_theta=1.0)


def _make_configs(depths, seeds):
    """Generate all (name, overrides) pairs for the pilot sweep."""
    configs = []

    for depth in depths:
        for seed in seeds:
            base = dict(depth=depth, seed=seed)

            # --- Baselines ---
            configs.append((
                f"baseline_gcn_d{depth}_s{seed}",
                {**base, **BASELINE_GCN},
            ))
            configs.append((
                f"baseline_gcn_fa_d{depth}_s{seed}",
                {**base, **BASELINE_GCN_FA},
            ))
            configs.append((
                f"baseline_gcnii_d{depth}_s{seed}",
                {**base, **BASELINE_GCNII},
            ))

            # --- Mainline ---
            configs.append((
                f"router_gcnii_d{depth}_s{seed}",
                {**base, **MAINLINE_BASE},
            ))

            # --- Ablation A: tau ---
            for tau_name, tau_val in [("tau_0p2", 0.2), ("tau_5p0", 5.0)]:
                configs.append((
                    f"router_gcnii_{tau_name}_d{depth}_s{seed}",
                    {**base, **MAINLINE_BASE, "assignment_tau": tau_val},
                ))

            # --- Ablation B: router repr ---
            configs.append((
                f"router_gcnii_repr_bilinear_d{depth}_s{seed}",
                {**base, **MAINLINE_BASE, "assign_type": "bilinear"},
            ))

            # --- Ablation C: router capacity ---
            for cap_name, K, topk in [("K32_topk4", 32, 4), ("K64_topk8", 64, 8)]:
                configs.append((
                    f"router_gcnii_{cap_name}_d{depth}_s{seed}",
                    {**base, **MAINLINE_BASE, "num_routers": K, "topk_assign": topk},
                ))

            # --- Ablation D: fusion ---
            for fusion in ["none", "add", "concat"]:
                configs.append((
                    f"router_gcnii_fusion_{fusion}_d{depth}_s{seed}",
                    {**base, **MAINLINE_BASE, "router_fusion": fusion},
                ))

            # --- Ablation E: mstep ---
            for mstep in [5, 100]:
                configs.append((
                    f"router_gcnii_mstep{mstep}_d{depth}_s{seed}",
                    {**base, **MAINLINE_BASE, "m_step_interval": mstep},
                ))

    return configs


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _build_cmd(name: str, overrides: dict, save_root: str, device: str) -> list[str]:
    save_dir = os.path.join(save_root, name)
    cmd = [PYTHON, TRAIN_SCRIPT, "--quiet", "--save_dir", save_dir, "--device", device]
    for k, v in overrides.items():
        if v is None:
            continue
        if isinstance(v, bool) or str(v).lower() in ("true", "false"):
            cmd += [f"--{k}", str(v)]
        elif k == "use_fa_layer" and v:
            cmd.append("--use_fa_layer")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def _is_done(save_dir: str) -> bool:
    return os.path.exists(os.path.join(save_dir, "final_metrics.json"))


def _write_config_plan(configs, save_root):
    plan_path = os.path.join(save_root, "config_plan.csv")
    with open(plan_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "overrides"])
        for name, overrides in configs:
            writer.writerow([name, json.dumps(overrides)])
    return plan_path


def _append_per_run(save_root: str, name: str):
    src = os.path.join(save_root, name, "final_metrics.json")
    if not os.path.exists(src):
        return
    with open(src) as f:
        row = json.load(f)
    row["run_name"] = name
    per_run_path = os.path.join(save_root, "per_run.csv")
    write_header = not os.path.exists(per_run_path)
    with open(per_run_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="TNM mainline pilot sweep")
    p.add_argument("--depths",    type=int, nargs="+", default=[4, 5])
    p.add_argument("--seeds",     type=int, nargs="+", default=[11])
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--save_root", type=str, default="results/tnm_mainline_sweep")
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--resume",    action="store_true",
                   help="Skip runs that already have final_metrics.json")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if results exist")
    p.add_argument("--max-runs",  type=int, default=None,
                   help="Stop after this many runs")
    args = p.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.join(args.save_root, "logs"), exist_ok=True)

    configs = _make_configs(args.depths, args.seeds)
    plan_path = _write_config_plan(configs, args.save_root)
    print(f"Config plan written to {plan_path}")
    print(f"Total configs: {len(configs)}")

    if args.dry_run:
        for name, overrides in configs:
            cmd = _build_cmd(name, overrides, args.save_root, args.device)
            print(f"  [DRY] {name}")
            print(f"        {' '.join(cmd)}")
        return

    run_count = 0
    for name, overrides in configs:
        if args.max_runs is not None and run_count >= args.max_runs:
            print(f"Reached --max-runs={args.max_runs}, stopping.")
            break

        save_dir = os.path.join(args.save_root, name)
        if args.resume and not args.overwrite and _is_done(save_dir):
            print(f"[SKIP] {name} (already done)")
            continue

        cmd = _build_cmd(name, overrides, args.save_root, args.device)
        log_path = os.path.join(args.save_root, "logs", f"{name}.log")
        print(f"[RUN ] {name}")

        t0 = time.time()
        with open(log_path, "w") as log_f:
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  FAILED (rc={result.returncode}) in {elapsed:.1f}s — see {log_path}")
        else:
            _append_per_run(args.save_root, name)
            print(f"  done in {elapsed:.1f}s")

        run_count += 1

    # Write summary
    per_run_path = os.path.join(args.save_root, "per_run.csv")
    if os.path.exists(per_run_path):
        import csv as _csv
        with open(per_run_path) as f:
            rows = list(_csv.DictReader(f))
        summary_path = os.path.join(args.save_root, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            if rows:
                writer = _csv.DictWriter(f, fieldnames=["run_name", "best_train_acc",
                                                         "best_test_acc", "eval_points_run",
                                                         "elapsed_s"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
        print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

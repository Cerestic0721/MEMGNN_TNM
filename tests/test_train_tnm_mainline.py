"""tests/test_train_tnm_mainline.py — Smoke test: router_graph_memory + GCNII runs end-to-end."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from train_tnm import main


def test_router_graph_memory_gcnii_smoke():
    """2 outer epochs, 128 examples, batch_size=16 — just verify no crash."""
    final = main([
        "--model", "router_graph_memory",
        "--gnn_type", "GCNII",
        "--depth", "4",
        "--num_layers", "3",
        "--h_dim", "32",
        "--dropout", "0.0",
        "--num_routers", "8",
        "--topk_assign", "2",
        "--assignment_tau", "1.0",
        "--router_fusion", "residual",
        "--residual_gamma", "0.02",
        "--proto_update", "ema",
        "--ema_beta", "0.03",
        "--m_step_interval", "1",
        "--epochs", "2",
        "--eval_every", "1",
        "--max_examples", "128",
        "--batch_size", "16",
        "--patience", "100",
        "--lr_patience", "100",
        "--device", "cpu",
        "--quiet",
        "--save_dir", "results/test_mainline_smoke",
    ])
    assert "best_train_acc" in final
    assert "best_test_acc" in final


def test_baseline_gcn_smoke():
    final = main([
        "--model", "baseline",
        "--gnn_type", "GCN",
        "--depth", "2",
        "--num_layers", "3",
        "--h_dim", "16",
        "--dropout", "0.0",
        "--epochs", "2",
        "--eval_every", "1",
        "--max_examples", "64",
        "--batch_size", "16",
        "--patience", "100",
        "--lr_patience", "100",
        "--device", "cpu",
        "--quiet",
        "--save_dir", "results/test_baseline_smoke",
    ])
    assert "best_train_acc" in final

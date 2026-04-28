"""
main.py — Entry Point
=======================
Runs all three experiments with the full ablation study, generates
dashboards, regret curves, CSV exports, and scientific conclusions.

Usage:
    python main.py
"""

import os
import sys
import time
import torch
import numpy as np
import argparse

import config as cfg
from ablation import (run_ablation_study, plot_comparative_bars,
                      print_scientific_conclusions)
from evaluation import export_csv, compute_pvalues, plot_regret_curve


def main():
    parser = argparse.ArgumentParser(description="Multi-Neuromodulated Modular RL Suite")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], help="Only run ablation study for a specific experiment (1, 2, or 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Multi-Neuromodulated Modular RL Architecture")
    print("  Research Experiment Suite")
    print("=" * 60)
    print(f"  Device:  {cfg.DEVICE}")
    print(f"  Seed:    {cfg.SEED}")
    print(f"  Output:  {cfg.RESULTS_DIR}")
    if args.exp:
        print(f"  Target:  Experiment {args.exp}")
    print("=" * 60)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # Set global seeds
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    start = time.time()

    # ── Run ablation study (specific experiment or all) ──
    all_results = run_ablation_study(seed=cfg.SEED, exp_id=args.exp)

    # ── Comparative bar charts ──
    print("\n> Generating comparative bar charts...")
    plot_comparative_bars(all_results)

    # ── Regret curves for Full Model ──
    print("\n> Generating regret curves...")
    if "Full Model" in all_results["Experiment_1"]:
        plot_regret_curve(
            all_results["Experiment_1"]["Full Model"],
            optimal_reward=cfg.EXP1_REWARD_MU_HI,
            exp_name="Experiment_1")
    if "Full Model" in all_results["Experiment_2"]:
        plot_regret_curve(
            all_results["Experiment_2"]["Full Model"],
            optimal_reward=cfg.EXP2_SAFE_REWARD,  # Safe optimal
            exp_name="Experiment_2")

    # ── CSV export ──
    print("\n> Exporting CSV results...")
    export_csv(all_results)

    # ── P-values ──
    print("\n> Computing p-values (Welch's t-test)...")
    compute_pvalues(all_results)

    # ── Scientific conclusions ──
    print_scientific_conclusions(all_results)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  All experiments completed in {elapsed:.1f}s")
    print(f"  Results saved to: {cfg.RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

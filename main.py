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
from ablation import (run_multiseed_study, run_multiseed_study_parallel,
                      plot_comparative_bars_multiseed,
                      print_scientific_conclusions_multiseed)
from evaluation import (summarize_multiseed, compute_multiseed_pvalues,
                        plot_experiment_1, plot_experiment_2,
                        plot_experiment_3, plot_regret_curve)


def main():
    parser = argparse.ArgumentParser(description="Multi-Neuromodulated Modular RL Suite")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], help="Only run ablation study for a specific experiment (1, 2, or 3)")
    parser.add_argument("--merge", action="store_true", help="Merge all experiment ablation charts into one file (default: separate)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes for the multi-seed study (>1 enables parallelism; ~= CPU cores)")
    parser.add_argument("--gpu", action="store_true", help="Keep parallel workers on the default device (GPU) instead of forcing CPU")
    args = parser.parse_args()

    print("=" * 60)
    print("  Multi-Neuromodulated Modular RL Architecture")
    print("  Research Experiment Suite")
    print("=" * 60)
    print(f"  Device:  {cfg.DEVICE}")
    print(f"  Seeds:   {cfg.SEEDS}")
    print(f"  Output:  {cfg.RESULTS_DIR}")
    if args.exp:
        print(f"  Target:  Experiment {args.exp}")
    print("=" * 60)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    start = time.time()

    # ── Run the multi-seed ablation study (statistical backbone) ──
    print("\n> Running multi-seed study "
          f"({len(cfg.SEEDS)} seeds × {len(cfg.ABLATION_CONFIGS)} configs, "
          f"workers={args.workers})...")
    if args.workers > 1:
        all_ms = run_multiseed_study_parallel(seeds=cfg.SEEDS, exp_id=args.exp,
                                              n_workers=args.workers,
                                              force_cpu=not args.gpu)
    else:
        all_ms = run_multiseed_study(seeds=cfg.SEEDS, exp_id=args.exp)

    # ── Dashboards from the first seed (representative visuals) ──
    print("\n> Generating per-config dashboards (seed = %d)..." % cfg.SEEDS[0])
    _plotters = {"Experiment_1": plot_experiment_1,
                 "Experiment_2": plot_experiment_2,
                 "Experiment_3": plot_experiment_3}
    for exp_key, configs in all_ms.items():
        for label, res_list in configs.items():
            if not res_list:
                continue
            sfx = f"_{label.replace(' ', '_').lower()}"
            _plotters[exp_key](res_list[0], suffix=sfx)

    # ── Comparative bar charts (mean ± 95% CI across seeds) ──
    print("\n> Generating comparative bar charts...")
    plot_comparative_bars_multiseed(all_ms, merge=args.merge)

    # ── Regret curves for Full Model (first seed) ──
    print("\n> Generating regret curves...")
    if all_ms.get("Experiment_1", {}).get("Full Model"):
        plot_regret_curve(all_ms["Experiment_1"]["Full Model"][0],
                          optimal_reward=cfg.EXP1_REWARD_MU_HI,
                          exp_name="Experiment_1")
    if all_ms.get("Experiment_2", {}).get("Full Model"):
        plot_regret_curve(all_ms["Experiment_2"]["Full Model"][0],
                          optimal_reward=cfg.EXP2_SAFE_REWARD,  # safe optimal
                          exp_name="Experiment_2")

    # ── Statistics: mean ± CI summary + paired significance tests ──
    # Suffix per-experiment runs so successive `--exp N` calls don't overwrite
    # each other's CSVs (a full run keeps the plain filenames).
    tag = f"_exp{args.exp}" if args.exp else ""
    print("\n> Summarizing metrics across seeds...")
    summarize_multiseed(all_ms, filename=f"summary_multiseed{tag}.csv")
    print("\n> Computing paired significance tests (Full vs each config)...")
    compute_multiseed_pvalues(all_ms, filename=f"pvalues{tag}.csv")

    # ── Scientific conclusions ──
    print_scientific_conclusions_multiseed(all_ms)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  All experiments completed in {elapsed:.1f}s")
    print(f"  Results saved to: {cfg.RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

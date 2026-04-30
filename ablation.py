"""
ablation.py — Ablation Study Framework (Section 9)
=====================================================
Runs all three experiments under four configurations:
    1. Full Model  [DA + NA + 5HT]
    2. Ablated NA  [DA + 5HT]      — tests adaptation/reset ability
    3. Ablated 5-HT [DA + NA]      — tests harm aversion
    4. Static Baseline              — fixed hyperparams, no modulators

Produces comparative bar charts and scientific conclusion summaries.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from experiments import run_experiment_1, run_experiment_2, run_experiment_3
from evaluation import (plot_experiment_1, plot_experiment_2,
                        plot_experiment_3, export_csv, compute_pvalues)


def run_ablation_study(seed: int = cfg.SEED, exp_id: int = None) -> dict:
    """Run experiments across all ablation configs.

    Args:
        seed: Random seed.
        exp_id: If provided (1, 2, or 3), only runs that specific experiment.

    Returns:
        Nested dict: {experiment_name: {config_label: results_dict}}
    """
    exp_keys = [f"Experiment_{i}" for i in ([exp_id] if exp_id else [1, 2, 3])]
    all_results = {k: {} for k in exp_keys}

    for config_label, abl_cfg in cfg.ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Configuration: {config_label}")
        print(f"{'='*60}")
        sfx = f"_{config_label.replace(' ', '_').lower()}"

        # Experiment 1
        if exp_id is None or exp_id == 1:
            print(f"  > Running Experiment 1 (Volatile Bandit)...")
            r1 = run_experiment_1(ablation_cfg=abl_cfg, seed=seed, label=config_label)
            all_results["Experiment_1"][config_label] = r1
            plot_experiment_1(r1, suffix=sfx)
            print(f"    Adaptation Latency: {r1['adaptation_latency']} steps")

        # Experiment 2
        if exp_id is None or exp_id == 2:
            print(f"  > Running Experiment 2 (High-Stakes Foraging)...")
            r2 = run_experiment_2(ablation_cfg=abl_cfg, seed=seed, label=config_label)
            all_results["Experiment_2"][config_label] = r2
            plot_experiment_2(r2, suffix=sfx)
            print(f"    Deaths: {r2['death_count']}, Total Reward: {r2['total_reward']:.1f}")

        # Experiment 3
        if exp_id is None or exp_id == 3:
            print(f"  > Running Experiment 3 (CartPole Adaptation)...")
            r3 = run_experiment_3(ablation_cfg=abl_cfg, seed=seed, label=config_label)
            all_results["Experiment_3"][config_label] = r3
            plot_experiment_3(r3, suffix=sfx)
            print(f"    Recovery Time: {r3['recovery_time']} episodes")

    return all_results


def plot_comparative_bars(all_results: dict, merge: bool = False):
    """Comparative bar chart: Survival Rate & Adaptation Latency (Sec 9B)."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    configs = list(cfg.ABLATION_CONFIGS.keys())
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"]
    
    active_exps = [k for k in all_results.keys() if all_results[k]]
    if not active_exps: return

    def draw_bar(ax, exp_key):
        if exp_key == "Experiment_1":
            vals = [all_results["Experiment_1"][c]["adaptation_latency"] for c in configs]
            ax.set_ylabel("Steps")
            ax.set_title("Exp 1: Adaptation Latency")
        elif exp_key == "Experiment_2":
            vals = [all_results["Experiment_2"][c]["death_count"] for c in configs]
            ax.set_ylabel("Deaths")
            ax.set_title("Exp 2: Death Count")
        elif exp_key == "Experiment_3":
            vals = [all_results["Experiment_3"][c]["recovery_time"] for c in configs]
            ax.set_ylabel("Episodes")
            ax.set_title("Exp 3: Recovery Time")
        
        ax.bar(configs, vals, color=colors)
        ax.tick_params(axis="x", rotation=25)

    if merge:
        n_plots = len(active_exps)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6), squeeze=False)
        fig.suptitle("Ablation Study — Comparative Results (Merged)", fontsize=16, fontweight="bold")
        for idx, exp_key in enumerate(active_exps):
            draw_bar(axes[0, idx], exp_key)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fname = os.path.join(cfg.RESULTS_DIR, "ablation_comparison_merged.png")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {fname}")
    else:
        for exp_key in active_exps:
            fig, ax = plt.subplots(figsize=(8, 6))
            draw_bar(ax, exp_key)
            plt.tight_layout()
            fname = os.path.join(cfg.RESULTS_DIR, f"ablation_comparison_{exp_key.lower()}.png")
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            print(f"  [Saved] {fname}")


def print_scientific_conclusions(all_results: dict):
    """Print summary explaining why Full Model outperforms ablations."""
    print("\n" + "=" * 60)
    print("  SCIENTIFIC CONCLUSIONS")
    print("=" * 60)

    # Experiment 1
    if "Experiment_1" in all_results and all_results["Experiment_1"]:
        e1 = all_results["Experiment_1"]
        full_lat = e1["Full Model"]["adaptation_latency"]
        na_lat = e1["Ablated NA"]["adaptation_latency"]
        print(f"\n[Exp 1 - Volatile Bandit]")
        print(f"  Full Model adaptation latency: {full_lat} steps")
        print(f"  Ablated NA adaptation latency: {na_lat} steps")
        if full_lat < na_lat:
            pct = (na_lat - full_lat) / max(na_lat, 1) * 100
            print(f"  -> The absence of NA resulted in {pct:.0f}% slower "
                  f"adaptation to the reward distribution switch.")
        else:
            print(f"  -> NA ablation did not worsen adaptation in this run.")

    # Experiment 2
    if "Experiment_2" in all_results and all_results["Experiment_2"]:
        e2 = all_results["Experiment_2"]
        full_d = e2["Full Model"]["death_count"]
        ht_d = e2["Ablated 5-HT"]["death_count"]
        print(f"\n[Exp 2 - High-Stakes Foraging]")
        print(f"  Full Model deaths: {full_d}")
        print(f"  Ablated 5-HT deaths: {ht_d}")
        if full_d < ht_d:
            pct = (ht_d - full_d) / max(ht_d, 1) * 100
            print(f"  -> The absence of 5-HT resulted in {pct:.0f}% more "
                  f"Death resets, confirming its role in harm aversion.")
        else:
            print(f"  -> 5-HT ablation did not increase deaths in this run.")

    # Experiment 3
    if "Experiment_3" in all_results and all_results["Experiment_3"]:
        e3 = all_results["Experiment_3"]
        full_r = e3["Full Model"]["recovery_time"]
        static_r = e3["Static Baseline"]["recovery_time"]
        print(f"\n[Exp 3 - CartPole Physics Adaptation]")
        print(f"  Full Model recovery: {full_r} episodes")
        print(f"  Static Baseline recovery: {static_r} episodes")
        if full_r < static_r:
            pct = (static_r - full_r) / max(static_r, 1) * 100
            print(f"  -> Dynamic neuromodulation achieved {pct:.0f}% faster "
                  f"recovery after physics perturbation.")
        else:
            print(f"  -> Static baseline matched or outperformed in this run.")
    print()

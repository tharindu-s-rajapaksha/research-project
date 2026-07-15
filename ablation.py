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

import config as cfg
from experiments import run_experiment_1, run_experiment_2, run_experiment_3
from evaluation import _metric_vectors, _mean_ci, _LOWER_IS_BETTER

_RUNNERS = {1: run_experiment_1, 2: run_experiment_2, 3: run_experiment_3}


# ======================================================================
# Multi-seed study (statistical backbone)
# ======================================================================
def run_multiseed_study(seeds: list = None, exp_id: int = None) -> dict:
    """Run every config on every seed, collecting per-seed result dicts.

    Returns:
        {experiment_name: {config_label: [res_seed0, res_seed1, ...]}}

    Every config sees the SAME set of seeds, so metric vectors are paired
    by seed for the downstream paired significance tests.
    """
    seeds = seeds if seeds is not None else cfg.SEEDS
    exp_ids = [exp_id] if exp_id else [1, 2, 3]
    exp_keys = [f"Experiment_{i}" for i in exp_ids]
    results = {k: {c: [] for c in cfg.ABLATION_CONFIGS} for k in exp_keys}

    n_total = len(cfg.ABLATION_CONFIGS) * len(seeds) * len(exp_ids)
    done = 0
    for config_label, abl_cfg in cfg.ABLATION_CONFIGS.items():
        for seed in seeds:
            for i in exp_ids:
                res = _RUNNERS[i](ablation_cfg=abl_cfg, seed=seed,
                                  label=config_label)
                results[f"Experiment_{i}"][config_label].append(res)
                done += 1
                print(f"    [{done}/{n_total}] {config_label} | "
                      f"Exp {i} | seed {seed}")
    return results


def _worker_init():
    """Pin each worker to a single BLAS/torch thread so N processes do not
    oversubscribe the CPU (they parallelise across runs, not within one)."""
    import torch
    torch.set_num_threads(1)


def _run_task(task):
    """Run ONE (experiment, config, seed) job. Top-level so it is picklable."""
    exp_id, config_label, seed = task
    res = _RUNNERS[exp_id](ablation_cfg=cfg.ABLATION_CONFIGS[config_label],
                           seed=seed, label=config_label)
    return exp_id, config_label, seed, res


def run_multiseed_study_parallel(seeds: list = None, exp_id: int = None,
                                 n_workers: int = 4,
                                 force_cpu: bool = True) -> dict:
    """Parallel version of run_multiseed_study across independent runs.

    Every (experiment, config, seed) is an independent job dispatched to a
    process pool.  Results are reassembled in sorted-seed order per config so
    the downstream PAIRED significance tests stay correctly aligned by seed.

    Args:
        n_workers: number of worker processes (≈ number of CPU cores).
        force_cpu: run workers on CPU (recommended — these nets are tiny and
                   CPU avoids GPU launch overhead and memory contention).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    seeds = seeds if seeds is not None else cfg.SEEDS
    exp_ids = [exp_id] if exp_id else [1, 2, 3]
    exp_keys = [f"Experiment_{i}" for i in exp_ids]

    if force_cpu:
        # Children inherit this env at spawn → config.DEVICE resolves to CPU.
        os.environ["NEUROMOD_FORCE_CPU"] = "1"

    tasks = [(i, c, s) for c in cfg.ABLATION_CONFIGS
             for s in seeds for i in exp_ids]
    # Collect keyed by seed to preserve pairing regardless of completion order.
    collected = {k: {c: {} for c in cfg.ABLATION_CONFIGS} for k in exp_keys}

    done, total = 0, len(tasks)
    print(f"    Dispatching {total} runs across {n_workers} workers "
          f"({'CPU' if force_cpu else 'default device'})...")
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_worker_init) as ex:
        for exp_i, clabel, seed, res in ex.map(_run_task, tasks):
            collected[f"Experiment_{exp_i}"][clabel][seed] = res
            done += 1
            print(f"    [{done}/{total}] {clabel} | Exp {exp_i} | seed {seed}")

    # Rebuild ordered per-seed lists (sorted seed order → aligned pairing).
    results = {k: {} for k in exp_keys}
    for k in exp_keys:
        for clabel, by_seed in collected[k].items():
            results[k][clabel] = [by_seed[s] for s in seeds if s in by_seed]
    return results


def _agg_metric(res_list, metric):
    """(mean, ci95, n) for one metric across a config's per-seed runs."""
    return _mean_ci(_metric_vectors(res_list).get(metric, []))


def plot_comparative_bars_multiseed(multiseed_results: dict,
                                    merge: bool = False):
    """Comparative bars with 95% CI error bars, aggregated across seeds."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    configs = list(cfg.ABLATION_CONFIGS.keys())
    _palette = ["#2ecc71", "#9b59b6", "#3498db", "#e74c3c", "#95a5a6",
                "#f39c12", "#1abc9c", "#34495e"]
    colors = [_palette[i % len(_palette)] for i in range(len(configs))]

    # (metric, y-label, title stem). Direction differs per metric, so the
    # y-label states it explicitly and the ★best marker respects _LOWER_IS_BETTER.
    _metric_for = {"Experiment_1": ("Adaptation_Latency", "Steps  (↓ better)",
                                    "Exp 1: Adaptation Latency"),
                   "Experiment_2": ("Death_Count", "Deaths  (↓ better)",
                                    "Exp 2: Death Count"),
                   "Experiment_3": ("Total_Reward", "Cumulative reward  (↑ better)",
                                    "Exp 3: Cumulative Reward (capstone)")}
    active = [k for k in multiseed_results if multiseed_results[k]
              and k in _metric_for]
    if not active:
        return

    def draw(ax, exp_key):
        metric, ylab, title = _metric_for[exp_key]
        lower_better = metric in _LOWER_IS_BETTER
        means, errs = [], []
        for c in configs:
            m, ci, _ = _agg_metric(multiseed_results[exp_key].get(c, []), metric)
            means.append(m)
            errs.append(0.0 if np.isnan(ci) else ci)
        bars = ax.bar(configs, means, yerr=errs, capsize=4, color=colors)
        ax.set_ylabel(ylab)
        # Mark the best config, respecting the metric's direction.
        valid = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
        if valid:
            best_i = (min if lower_better else max)(valid, key=lambda t: t[1])[0]
        else:
            best_i = None
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        for j, (bar, m) in enumerate(zip(bars, means)):
            if np.isnan(m):
                continue
            star = "  ★best" if j == best_i else ""
            ax.annotate(f"{m:.1f}{star}",
                        xy=(bar.get_x() + bar.get_width() / 2, m),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        if best_i is not None:
            bars[best_i].set_edgecolor("black")
            bars[best_i].set_linewidth(2.5)

    if merge:
        fig, axes = plt.subplots(1, len(active), figsize=(6 * len(active), 6),
                                 squeeze=False)
        fig.suptitle("Ablation Study — Comparative Results (mean ± 95% CI)",
                     fontsize=16, fontweight="bold")
        for idx, exp_key in enumerate(active):
            draw(axes[0, idx], exp_key)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fname = os.path.join(cfg.RESULTS_DIR, "ablation_comparison_merged.png")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {fname}")
    else:
        for exp_key in active:
            fig, ax = plt.subplots(figsize=(9, 6))
            draw(ax, exp_key)
            plt.tight_layout()
            fname = os.path.join(cfg.RESULTS_DIR,
                                 f"ablation_comparison_{exp_key.lower()}.png")
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            print(f"  [Saved] {fname}")


def print_scientific_conclusions_multiseed(multiseed_results: dict):
    """Report the headline comparisons with means, CIs, and direction, using
    the aggregated per-seed data (NaN-aware for undefined CartPole recovery).
    """
    print("\n" + "=" * 64)
    print("  SCIENTIFIC CONCLUSIONS  (mean ± 95% CI across seeds)")
    print("=" * 64)

    def line(exp_key, metric, full_label, other_label, unit):
        if exp_key not in multiseed_results or not multiseed_results[exp_key]:
            return
        exp = multiseed_results[exp_key]
        if full_label not in exp or other_label not in exp:
            return
        fm, fci, fn = _agg_metric(exp[full_label], metric)
        om, oci, on = _agg_metric(exp[other_label], metric)
        print(f"\n[{exp_key}] metric = {metric}")
        print(f"  {full_label:<16}: {fm:.2f} ± {fci:.2f}  (n={fn})")
        print(f"  {other_label:<16}: {om:.2f} ± {oci:.2f}  (n={on})")
        if np.isnan(fm) or np.isnan(om):
            print("  -> Metric undefined for one config (e.g. never competent).")
            return
        lower_better = metric in _LOWER_IS_BETTER
        full_better = (fm < om) if lower_better else (fm > om)
        denom = abs(om) if om != 0 else 1.0
        pct = abs(om - fm) / denom * 100
        verb = "better" if full_better else "worse"
        print(f"  -> Full Model is {pct:.0f}% {verb} than {other_label}.")

    line("Experiment_1", "Adaptation_Latency", "Full Model", "Ablated NA", "steps")
    line("Experiment_2", "Death_Count", "Full Model", "Ablated 5-HT", "deaths")
    # Capstone: removing ANY single hormone should lower cumulative reward.
    for other in ["Ablated NA", "Ablated DA", "Ablated 5-HT", "Static Baseline"]:
        line("Experiment_3", "Total_Reward", "Full Model", other, "reward")
    print()

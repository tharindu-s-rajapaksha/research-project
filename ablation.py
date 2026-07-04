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
from scipy import stats

import config as cfg
from experiments import run_experiment_1, run_experiment_2, run_experiment_3
from evaluation import (plot_experiment_1, plot_experiment_2,
                        plot_experiment_3, export_csv, compute_pvalues)


# Primary scalar metric per experiment (used for bars, stats, conclusions).
PRIMARY_METRIC = {
    "Experiment_1": "adaptation_latency",
    "Experiment_2": "death_count",
    "Experiment_3": "recovery_time",
}


def _extract_metrics(exp_id: int, res: dict) -> dict:
    """Pull the scalar summary metrics from one experiment run."""
    m = {"total_reward": float(sum(res.get("rewards", [])))}
    if exp_id == 1:
        m["adaptation_latency"] = float(res["adaptation_latency"])
    elif exp_id == 2:
        m["death_count"] = float(res["death_count"])
        sv = res.get("survival_steps", [0]) or [0]
        m["survival_rate"] = float(np.mean(sv))
    elif exp_id == 3:
        m["recovery_time"] = float(res["recovery_time"])
    return m


def run_ablation_study(seeds=None, exp_id: int = None, seed: int = None):
    """Run experiments across all ablation configs over multiple seeds.

    Args:
        seeds:  List of independent random seeds (defaults to cfg.EXP_SEEDS).
        exp_id: If provided (1, 2, or 3), only runs that specific experiment.
        seed:   Back-compat single seed; used only if ``seeds`` is None.

    Returns:
        (representative, metrics_by_config):
          • representative: {exp_key: {config_label: results_dict}} using the
            first seed — full time-series for dashboards / regret curves.
          • metrics_by_config: {exp_key: {config_label: {metric: np.array}}}
            with one value per seed (plus a 'seed' array), for stats / CSV.
    """
    if seeds is None:
        seeds = [seed] if seed is not None else cfg.EXP_SEEDS

    exp_ids = [exp_id] if exp_id else [1, 2, 3]
    exp_keys = [f"Experiment_{i}" for i in exp_ids]
    representative = {k: {} for k in exp_keys}
    metrics_by_config = {k: {} for k in exp_keys}

    runners = {1: run_experiment_1, 2: run_experiment_2, 3: run_experiment_3}
    plotters = {1: plot_experiment_1, 2: plot_experiment_2, 3: plot_experiment_3}
    names = {1: "Volatile Bandit", 2: "High-Stakes Foraging", 3: "CartPole Adaptation"}

    for config_label, abl_cfg in cfg.ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Configuration: {config_label}   (seeds={seeds})")
        print(f"{'='*60}")
        sfx = f"_{config_label.replace(' ', '_').lower()}"

        for i in exp_ids:
            ek = f"Experiment_{i}"
            print(f"  > Running Experiment {i} ({names[i]})...")
            per_seed_metrics = {}
            per_seed_results = []
            for sd in seeds:
                res = runners[i](ablation_cfg=abl_cfg, seed=sd, label=config_label)
                per_seed_results.append(res)
                m = _extract_metrics(i, res)
                m["seed"] = float(sd)
                for name, val in m.items():
                    per_seed_metrics.setdefault(name, []).append(val)

            metrics_by_config[ek][config_label] = {
                name: np.array(vals, dtype=float)
                for name, vals in per_seed_metrics.items()
            }
            # First seed is the representative run for dashboards.
            representative[ek][config_label] = per_seed_results[0]
            plotters[i](per_seed_results[0], suffix=sfx)

            metric = PRIMARY_METRIC[ek]
            arr = metrics_by_config[ek][config_label][metric]
            print(f"    {metric}: mean={arr.mean():.1f} ± {arr.std():.1f} "
                  f"(n={len(arr)})")

    return representative, metrics_by_config


def plot_comparative_bars(metrics_by_config: dict, merge: bool = False):
    """Comparative bar chart of each experiment's primary metric (Sec 9B).

    Bars show the mean over seeds with std error bars.
    """
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    active_exps = [k for k in metrics_by_config.keys() if metrics_by_config[k]]
    if not active_exps:
        return

    # Use whichever configs are actually present (keeps order from cfg).
    configs = [c for c in cfg.ABLATION_CONFIGS.keys()
               if any(c in metrics_by_config[k] for k in active_exps)]
    palette = sns.color_palette("husl", len(configs))

    metric_meta = {
        "Experiment_1": ("adaptation_latency", "Steps", "Exp 1: Adaptation Latency"),
        "Experiment_2": ("death_count", "Deaths", "Exp 2: Death Count"),
        "Experiment_3": ("recovery_time", "Episodes", "Exp 3: Recovery Time"),
    }

    def draw_bar(ax, exp_key):
        metric, ylabel, title = metric_meta[exp_key]
        means, stds = [], []
        for c in configs:
            arr = metrics_by_config[exp_key].get(c, {}).get(metric, np.array([]))
            means.append(float(np.mean(arr)) if len(arr) else 0.0)
            stds.append(float(np.std(arr)) if len(arr) > 1 else 0.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        bars = ax.bar(configs, means, yerr=stds, capsize=4, color=palette)
        ax.tick_params(axis="x", rotation=25)

        for bar, mean in zip(bars, means):
            ax.annotate(f'{mean:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

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


def _mean(metrics_by_config, exp, config, metric):
    arr = metrics_by_config.get(exp, {}).get(config, {}).get(metric)
    return float(np.mean(arr)) if arr is not None and len(arr) else None


def _pval(metrics_by_config, exp, a, b, metric):
    arr_a = metrics_by_config.get(exp, {}).get(a, {}).get(metric)
    arr_b = metrics_by_config.get(exp, {}).get(b, {}).get(metric)
    if arr_a is not None and arr_b is not None and len(arr_a) > 1 and len(arr_b) > 1:
        if np.std(arr_a) == 0 and np.std(arr_b) == 0:
            # Both constant across seeds: t-test undefined.
            return 1.0 if np.mean(arr_a) == np.mean(arr_b) else 0.0
        _, p = stats.ttest_ind(arr_a, arr_b, equal_var=False)
        return float(p)
    return None


def print_scientific_conclusions(metrics_by_config: dict):
    """Print summary explaining why the Full Model outperforms ablations.

    Uses seed-averaged metrics and (where ≥2 seeds) cross-seed Welch p-values.
    """
    print("\n" + "=" * 60)
    print("  SCIENTIFIC CONCLUSIONS  (means over seeds)")
    print("=" * 60)

    def _sig(p):
        if p is None:
            return "(p=n/a: need ≥2 seeds)"
        return f"(p={p:.3f}{'*' if p < 0.05 else ''})"

    # Experiment 1
    if metrics_by_config.get("Experiment_1"):
        full = _mean(metrics_by_config, "Experiment_1", "Full Model", "adaptation_latency")
        na = _mean(metrics_by_config, "Experiment_1", "Ablated NA", "adaptation_latency")
        p = _pval(metrics_by_config, "Experiment_1", "Full Model", "Ablated NA", "adaptation_latency")
        print(f"\n[Exp 1 - Volatile Bandit]")
        print(f"  Full Model adaptation latency: {full:.1f} steps")
        if na is not None:
            print(f"  Ablated NA adaptation latency: {na:.1f} steps  {_sig(p)}")
            if full < na:
                pct = (na - full) / max(na, 1) * 100
                print(f"  -> Removing NA slowed adaptation to the reward switch "
                      f"by {pct:.0f}%.")
            else:
                print(f"  -> NA ablation did not worsen adaptation in this run.")

    # Experiment 2
    if metrics_by_config.get("Experiment_2"):
        full = _mean(metrics_by_config, "Experiment_2", "Full Model", "death_count")
        ht = _mean(metrics_by_config, "Experiment_2", "Ablated 5-HT", "death_count")
        p = _pval(metrics_by_config, "Experiment_2", "Full Model", "Ablated 5-HT", "death_count")
        print(f"\n[Exp 2 - High-Stakes Foraging]")
        print(f"  Full Model deaths: {full:.1f}")
        if ht is not None:
            print(f"  Ablated 5-HT deaths: {ht:.1f}  {_sig(p)}")
            if full < ht:
                pct = (ht - full) / max(ht, 1) * 100
                print(f"  -> Removing 5-HT produced {pct:.0f}% more death resets, "
                      f"confirming its role in harm aversion.")
            else:
                print(f"  -> 5-HT ablation did not increase deaths in this run.")

    # Experiment 3
    if metrics_by_config.get("Experiment_3"):
        full = _mean(metrics_by_config, "Experiment_3", "Full Model", "recovery_time")
        static = _mean(metrics_by_config, "Experiment_3", "Static Baseline", "recovery_time")
        p = _pval(metrics_by_config, "Experiment_3", "Full Model", "Static Baseline", "recovery_time")
        print(f"\n[Exp 3 - CartPole Physics Adaptation]")
        print(f"  Full Model recovery: {full:.1f} episodes")
        if static is not None:
            print(f"  Static Baseline recovery: {static:.1f} episodes  {_sig(p)}")
            if full < static:
                pct = (static - full) / max(static, 1) * 100
                print(f"  -> Dynamic neuromodulation recovered {pct:.0f}% faster "
                      f"after the physics perturbation.")
            else:
                print(f"  -> Static baseline matched or outperformed in this run.")
    print()

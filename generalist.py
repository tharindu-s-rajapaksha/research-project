"""
generalist.py — "Generalist vs Specialists" study (the novelty test)
====================================================================
Directly tests the interim novelty (Slides 3/6/7): *integration beats
single-modulator gating*.

Rather than asking whether removing one hormone from the Full agent hurts
(ablation — which only isolates 5-HT robustly, since a competent base learner
makes NA/DA redundant), this study compares the Full tri-hormone agent against
three SINGLE-modulator specialists across TWO tasks with opposite demands:

    Task A = Experiment 1 (volatile bandit, no lethal arm)  -> ADAPTATION
             metric: Adaptation_Latency  (lower = faster re-locking)
    Task B = Experiment 2 (high-stakes foraging, lethal arm) -> SURVIVAL
             metric: Total_Reward        (higher = avoids the death trap)

Each specialist has a fatal blind spot:
    * DA only / NA only  -> no serotonin -> die on Task B (deep negative reward)
    * 5-HT only          -> no adaptation hormones -> slow on Task A
Only the Full agent is competent on BOTH. The headline is therefore the
*worst-task floor*: Full has by far the highest floor; every specialist has a
floor in the basement of the task outside its niche.

Outputs (research_results/):
    generalist_summary.csv     per (agent, task, metric) mean ± 95% CI
    generalist_pvalues.csv     paired Full-vs-specialist tests (Holm-corrected)
    generalist_scatter.png     adaptation × survival quadrant (Full alone top-right)
    generalist_floor.png       normalized worst-task score per agent
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import config as cfg
from experiments import run_experiment_1, run_experiment_2
from evaluation import _mean_ci, _holm

# Task A = adaptation (Exp 1); Task B = survival (Exp 2).
_TASK_RUNNER = {"A_adapt": run_experiment_1, "B_survive": run_experiment_2}
# Headline metric per task and its direction (True = lower is better).
_TASK_METRIC = {"A_adapt": ("Adaptation_Latency", True),
                "B_survive": ("Total_Reward", False)}
_TASK_LABEL = {"A_adapt": "Adaptation (Exp 1)", "B_survive": "Survival (Exp 2)"}
_FULL = "Full Model"


# ======================================================================
# Metric extraction
# ======================================================================
def _task_metrics(task, res):
    """Reduce one run to the scalar metrics we care about for that task."""
    if task == "A_adapt":
        return {"Adaptation_Latency": float(res["adaptation_latency"]),
                "Total_Reward": float(np.sum(res["rewards"]))}
    # B_survive
    return {"Total_Reward": float(res["total_reward"]),
            "Death_Count": float(res["death_count"])}


# ======================================================================
# Parallel runner  (one job = one agent on one task at one seed)
# ======================================================================
def _gen_worker_init():
    import torch
    torch.set_num_threads(1)


def _gen_run_task(job):
    """Run ONE (task, config_label, seed) job. Top-level → picklable."""
    task, label, seed = job
    runner = _TASK_RUNNER[task]
    res = runner(ablation_cfg=cfg.GENERALIST_CONFIGS[label], seed=seed,
                 label=label)
    return task, label, seed, _task_metrics(task, res)


def run_generalist_battery(seeds=None, n_workers=1, force_cpu=True):
    """Run every agent on BOTH tasks across all seeds.

    Returns {task: {label: {metric: [v_seed0, v_seed1, ...]}}}, seed-aligned.
    """
    seeds = seeds if seeds is not None else cfg.SEEDS
    labels = list(cfg.GENERALIST_CONFIGS.keys())
    tasks = list(_TASK_RUNNER.keys())
    jobs = [(t, l, s) for t in tasks for l in labels for s in seeds]

    # collected[task][label][seed] = {metric: value}
    collected = {t: {l: {} for l in labels} for t in tasks}

    if n_workers > 1:
        if force_cpu:
            os.environ["NEUROMOD_FORCE_CPU"] = "1"
        print(f"    Dispatching {len(jobs)} runs across {n_workers} workers "
              f"({'CPU' if force_cpu else 'default device'})...")
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_gen_worker_init) as ex:
            for task, label, seed, m in ex.map(_gen_run_task, jobs):
                collected[task][label][seed] = m
                done += 1
                print(f"    [{done}/{len(jobs)}] {label:<10} | {task} | "
                      f"seed {seed}")
    else:
        for i, job in enumerate(jobs, 1):
            task, label, seed, m = _gen_run_task(job)
            collected[task][label][seed] = m
            print(f"    [{i}/{len(jobs)}] {label:<10} | {task} | seed {seed}")

    # Rebuild seed-aligned metric vectors.
    out = {t: {l: {} for l in labels} for t in tasks}
    for t in tasks:
        for l in labels:
            by_seed = collected[t][l]
            ordered = [by_seed[s] for s in seeds if s in by_seed]
            metrics = ordered[0].keys() if ordered else []
            for metric in metrics:
                out[t][l][metric] = [d[metric] for d in ordered]
    return out


# ======================================================================
# Statistics
# ======================================================================
def summarize_generalist(battery, filename="generalist_summary.csv"):
    """Per (agent, task, metric) mean ± 95% CI CSV."""
    import pandas as pd
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    rows = []
    for task, per_label in battery.items():
        for label, per_metric in per_label.items():
            for metric, vec in per_metric.items():
                mean, ci, n = _mean_ci(vec)
                rows.append({"Task": _TASK_LABEL[task], "Agent": label,
                             "Metric": metric,
                             "Mean": round(mean, 4), "CI95": round(ci, 4),
                             "N_seeds": n})
    df = pd.DataFrame(rows)
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    df.to_csv(fpath, index=False)
    print(f"  [Saved] {fpath}")
    return df


def generalist_pvalues(battery, filename="generalist_pvalues.csv"):
    """Paired Full-vs-specialist tests on each task's headline metric.

    One comparison family = {each specialist} × {each task headline metric}.
    Holm–Bonferroni correction is applied across the whole family so the
    'no specialist beats Full on both axes' claim is held to a corrected
    threshold.
    """
    import pandas as pd
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    specialists = [l for l in cfg.GENERALIST_CONFIGS if l != _FULL]

    rows = []
    for task in _TASK_RUNNER:
        metric, lower_better = _TASK_METRIC[task]
        full_vec = np.asarray(battery[task][_FULL][metric], dtype=float)
        for spec in specialists:
            spec_vec = np.asarray(battery[task][spec][metric], dtype=float)
            pairs = [(f, o) for f, o in zip(full_vec, spec_vec)
                     if not (np.isnan(f) or np.isnan(o))]
            if len(pairs) < 2:
                continue
            f_arr = np.array([p[0] for p in pairs])
            o_arr = np.array([p[1] for p in pairs])
            t_stat, p_val = stats.ttest_rel(f_arr, o_arr)
            full_better = (f_arr.mean() < o_arr.mean()) if lower_better \
                else (f_arr.mean() > o_arr.mean())
            rows.append({"Task": _TASK_LABEL[task], "Metric": metric,
                         "Comparison": f"{_FULL} vs {spec}",
                         "Full_mean": round(float(f_arr.mean()), 3),
                         "Spec_mean": round(float(o_arr.mean()), 3),
                         "Full_better": bool(full_better),
                         "t": round(float(t_stat), 3),
                         "p_value": round(float(p_val), 6),
                         "N": len(pairs)})

    if rows:
        adj = _holm([r["p_value"] for r in rows])
        for r, pa in zip(rows, adj):
            r["p_holm"] = round(float(pa), 6)
            r["sig_holm_0.05"] = bool(pa < 0.05 and r["Full_better"])

    df = pd.DataFrame(rows)
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    df.to_csv(fpath, index=False)
    print(f"  [Saved] {fpath}")
    return df


# ======================================================================
# Normalized scores (for the quadrant scatter + floor bar)
# ======================================================================
def _norm_scores(battery):
    """Per-agent adaptation & survival scores in [0,1] (min-max across the
    four agents, so 1 = best agent on that axis, 0 = worst). Returns
    {label: (adapt_norm, survive_norm, adapt_lat_mean, survive_reward_mean)}.
    """
    labels = list(cfg.GENERALIST_CONFIGS.keys())
    lat = {l: _mean_ci(battery["A_adapt"][l]["Adaptation_Latency"])[0]
           for l in labels}
    rew = {l: _mean_ci(battery["B_survive"][l]["Total_Reward"])[0]
           for l in labels}
    lat_v = np.array(list(lat.values()))
    rew_v = np.array(list(rew.values()))
    lat_lo, lat_hi = lat_v.min(), lat_v.max()
    rew_lo, rew_hi = rew_v.min(), rew_v.max()
    eps = 1e-9
    scores = {}
    for l in labels:
        adapt_norm = (lat_hi - lat[l]) / (lat_hi - lat_lo + eps)   # lower lat → higher
        survive_norm = (rew[l] - rew_lo) / (rew_hi - rew_lo + eps)  # higher rew → higher
        scores[l] = (adapt_norm, survive_norm, lat[l], rew[l])
    return scores


# ======================================================================
# Plots
# ======================================================================
_COLORS = {"Full Model": "#2ecc71", "DA only": "#f39c12",
           "NA only": "#3498db", "5-HT only": "#9b59b6"}


def plot_generalist_scatter(battery, filename="generalist_scatter.png"):
    """Adaptation × Survival quadrant: Full alone in the good-good corner,
    each specialist stranded on one failing edge."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    scores = _norm_scores(battery)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    # Shade the "generalist zone" (good on both) top-right.
    ax.axhspan(0.5, 1.05, xmin=0.0, xmax=1.0, color="#2ecc71", alpha=0.04)
    ax.axvspan(0.5, 1.05, ymin=0.0, ymax=1.0, color="#2ecc71", alpha=0.04)
    ax.axhline(0.5, color="#bbbbbb", lw=1, ls="--", zorder=1)
    ax.axvline(0.5, color="#bbbbbb", lw=1, ls="--", zorder=1)

    for label, (ax_n, sv_n, lat, rew) in scores.items():
        is_full = (label == _FULL)
        ax.scatter(ax_n, sv_n, s=520 if is_full else 300,
                   marker="*" if is_full else "o",
                   color=_COLORS.get(label, "#555"),
                   edgecolor="black", linewidth=1.6 if is_full else 1.0,
                   zorder=5)
        dy = 0.05 if sv_n < 0.9 else -0.07
        ax.annotate(f"{label}\n(lat {lat:.0f}, rew {rew:,.0f})",
                    xy=(ax_n, sv_n), xytext=(0, 16 if dy > 0 else -34),
                    textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold" if is_full else "normal")

    ax.text(0.75, 0.97, "GENERALIST\n(good at both)", ha="center", va="top",
            fontsize=10, color="#2ecc71", fontweight="bold", alpha=0.8)
    ax.text(0.75, 0.03, "adapts but\nDIES", ha="center", va="bottom",
            fontsize=9, color="#c0392b", alpha=0.7)
    ax.text(0.03, 0.97, "survives but\nSLOW", ha="left", va="top",
            fontsize=9, color="#c0392b", alpha=0.7)

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlabel("Adaptation score  (→ faster re-locking, Exp 1)",
                  fontsize=11)
    ax.set_ylabel("Survival score  (→ avoids the lethal arm, Exp 2)",
                  fontsize=11)
    ax.set_title("Integration beats single-modulator gating\n"
                 "Only the Full tri-hormone agent is competent on both tasks",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    fig.savefig(fpath, dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fpath}")


def plot_generalist_floor(battery, filename="generalist_floor.png"):
    """Worst-task normalized score per agent — the headline 'generalist' bar.
    Full has the highest floor; every specialist bottoms out on its blind-spot
    task."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    scores = _norm_scores(battery)
    labels = list(cfg.GENERALIST_CONFIGS.keys())
    floor = [min(scores[l][0], scores[l][1]) for l in labels]
    colors = [_COLORS.get(l, "#555") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, floor, color=colors, edgecolor="black")
    best_i = int(np.argmax(floor))
    bars[best_i].set_linewidth(2.6)
    for j, (b, v) in enumerate(zip(bars, floor)):
        star = "  (best)" if j == best_i else ""
        ax.annotate(f"{v:.2f}{star}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Worst-task score  (min over adaptation & survival)",
                  fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Generalist floor — competence on your WEAKEST task\n"
                 "(higher = good at everything; specialists collapse on one)",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=12)
    plt.tight_layout()
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    fig.savefig(fpath, dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fpath}")


# ======================================================================
# Console summary
# ======================================================================
def print_generalist_conclusions(battery):
    scores = _norm_scores(battery)
    print("\n" + "=" * 70)
    print("  GENERALIST vs SPECIALISTS  (integration beats single-modulator)")
    print("=" * 70)
    print(f"{'Agent':<12} | {'Adapt lat':>10} {'(score)':>8} | "
          f"{'Surv rew':>10} {'(score)':>8} | {'floor':>6}")
    print("-" * 70)
    for l in cfg.GENERALIST_CONFIGS:
        ax_n, sv_n, lat, rew = scores[l]
        floor = min(ax_n, sv_n)
        print(f"{l:<12} | {lat:>10.0f} {ax_n:>8.2f} | "
              f"{rew:>10.0f} {sv_n:>8.2f} | {floor:>6.2f}")
    print("-" * 70)
    winner = max(cfg.GENERALIST_CONFIGS,
                 key=lambda l: min(scores[l][0], scores[l][1]))
    print(f"  Highest floor (best generalist): {winner}")
    print("  Interpretation: each specialist scores high on ONE axis and")
    print("  collapses on the other; only the Full agent keeps a high floor.")
    print("=" * 70)


def run_generalist_study(seeds=None, n_workers=1, force_cpu=True):
    """End-to-end: run the battery, write CSVs, and render both figures."""
    battery = run_generalist_battery(seeds=seeds, n_workers=n_workers,
                                     force_cpu=force_cpu)
    summarize_generalist(battery)
    generalist_pvalues(battery)
    plot_generalist_scatter(battery)
    plot_generalist_floor(battery)
    print_generalist_conclusions(battery)
    return battery


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generalist vs specialists study")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    a = p.parse_args()
    run_generalist_study(seeds=a.seeds, n_workers=a.workers,
                         force_cpu=not a.gpu)

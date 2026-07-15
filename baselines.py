"""
baselines.py — Fair-baseline battery: "does the agent beat STANDARD RL?"
========================================================================
The ablation's Static/Vanilla baseline is a near-greedy fixed ε=0.01 DQN with no
exploration schedule and a Huber loss that gradient-clips the −500 catastrophe.
That is a *weak* reference. The honest form of the research claim is that a
self-regulating agent beats the **best static configuration**, so this module
pits the Full tri-hormone agent against a spread of properly-tuned standard DQNs
(see ``config.BASELINE_CONFIGS``) across both task niches:

    Task A = adaptation (Exp 1 volatile bandit)  → metric Adaptation_Latency ↓
             tests C2: does NA beat the best FIXED / ANNEALED exploration, not
             merely ε=0.01?
    Task B = survival  (Exp 2 high-stakes forage) → metric Total_Reward ↑
             tests H2: does 5-HT beat a value function that ISN'T crippled by
             Huber under-weighting the −500 (MSE / reward-scaled DQN)?

Honest-outcome principle: report whichever way it comes out. If a value-corrected
DQN survives Exp 2 without 5-HT, that weakens the 5-HT *necessity* claim and is
stated plainly — the purpose here is fairness, not a bigger win.

Outputs (research_results/):
    baselines_summary.csv     per (agent, task, metric) mean ± 95% CI
    baselines_pvalues.csv     paired Full-vs-baseline tests (t + Wilcoxon, Holm)
    baselines_headtohead.png  Full vs every standard-DQN baseline, raw units
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from experiments import run_experiment_1, run_experiment_2
from evaluation import _mean_ci, _holm, _paired_p

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
def _bl_worker_init():
    import torch
    torch.set_num_threads(1)


def _bl_run_task(job):
    """Run ONE (task, config_label, seed) job. Top-level → picklable."""
    task, label, seed = job
    runner = _TASK_RUNNER[task]
    res = runner(ablation_cfg=cfg.BASELINE_CONFIGS[label], seed=seed, label=label)
    return task, label, seed, _task_metrics(task, res)


def run_baseline_battery(seeds=None, n_workers=1, force_cpu=True):
    """Run every agent on BOTH tasks across all seeds.

    Returns {task: {label: {metric: [v_seed0, v_seed1, ...]}}}, seed-aligned.
    """
    seeds = seeds if seeds is not None else cfg.SEEDS
    labels = list(cfg.BASELINE_CONFIGS.keys())
    tasks = list(_TASK_RUNNER.keys())
    jobs = [(t, l, s) for t in tasks for l in labels for s in seeds]

    collected = {t: {l: {} for l in labels} for t in tasks}

    if n_workers > 1:
        if force_cpu:
            os.environ["NEUROMOD_FORCE_CPU"] = "1"
        print(f"    Dispatching {len(jobs)} runs across {n_workers} workers "
              f"({'CPU' if force_cpu else 'default device'})...")
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_bl_worker_init) as ex:
            for task, label, seed, m in ex.map(_bl_run_task, jobs):
                collected[task][label][seed] = m
                done += 1
                print(f"    [{done}/{len(jobs)}] {label:<18} | {task} | seed {seed}")
    else:
        for i, job in enumerate(jobs, 1):
            task, label, seed, m = _bl_run_task(job)
            collected[task][label][seed] = m
            print(f"    [{i}/{len(jobs)}] {label:<18} | {task} | seed {seed}")

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
def summarize_baselines(battery, filename="baselines_summary.csv"):
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


def baseline_pvalues(battery, filename="baselines_pvalues.csv"):
    """Paired Full-vs-baseline tests on each task's headline metric.

    Family = {each standard-DQN baseline} × {each task headline metric}. Both a
    paired t-test and a Wilcoxon signed-rank are reported (the latter is the
    defensible test for the non-normal metrics), each Holm–Bonferroni corrected
    across the whole family so 'Full beats every standard DQN' is held to a
    corrected threshold.
    """
    import pandas as pd
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    baselines = [l for l in cfg.BASELINE_CONFIGS if l != _FULL]

    rows = []
    for task in _TASK_RUNNER:
        metric, lower_better = _TASK_METRIC[task]
        full_vec = np.asarray(battery[task][_FULL][metric], dtype=float)
        for base in baselines:
            base_vec = np.asarray(battery[task][base][metric], dtype=float)
            pairs = [(f, o) for f, o in zip(full_vec, base_vec)
                     if not (np.isnan(f) or np.isnan(o))]
            if len(pairs) < 2:
                continue
            f_arr = np.array([p[0] for p in pairs])
            o_arr = np.array([p[1] for p in pairs])
            t_stat, p_t, p_w = _paired_p(f_arr, o_arr)
            full_better = (f_arr.mean() < o_arr.mean()) if lower_better \
                else (f_arr.mean() > o_arr.mean())
            rows.append({"Task": _TASK_LABEL[task], "Metric": metric,
                         "Comparison": f"{_FULL} vs {base}",
                         "Full_mean": round(float(f_arr.mean()), 3),
                         "Base_mean": round(float(o_arr.mean()), 3),
                         "Full_better": bool(full_better),
                         "t": round(t_stat, 3),
                         "p_ttest": round(p_t, 6),
                         "p_wilcoxon": round(p_w, 6),
                         "N": len(pairs)})

    if rows:
        adj_t = _holm([r["p_ttest"] for r in rows])
        adj_w = _holm([r["p_wilcoxon"] for r in rows])
        for r, pt, pw in zip(rows, adj_t, adj_w):
            r["p_ttest_holm"] = round(float(pt), 6)
            r["p_wilcoxon_holm"] = round(float(pw), 6)
            # Conservative: significant only if BOTH tests survive Holm AND Full wins.
            r["sig_holm_0.05"] = bool(pt < 0.05 and pw < 0.05 and r["Full_better"])

    df = pd.DataFrame(rows)
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    df.to_csv(fpath, index=False)
    print(f"  [Saved] {fpath}")
    if not df.empty:
        print(df.to_string(index=False))
    return df


def _star(p):
    return ("***" if p < 0.001 else "**" if p < 0.01
            else "*" if p < 0.05 else "ns")


def _sig_stars(battery):
    """Holm-corrected (Wilcoxon) significance of Full vs each baseline on each
    task's headline metric. Returns {(task, label): stars}."""
    baselines = [l for l in cfg.BASELINE_CONFIGS if l != _FULL]
    entries, pvals = [], []
    for task in _TASK_RUNNER:
        metric = _TASK_METRIC[task][0]
        fv = np.asarray(battery[task][_FULL][metric], dtype=float)
        for base in baselines:
            bv = np.asarray(battery[task][base][metric], dtype=float)
            _, _, p_w = _paired_p(fv, bv)
            entries.append((task, base))
            pvals.append(p_w)
    adj = _holm(pvals)
    return {e: _star(pa) for e, pa in zip(entries, adj)}


# ======================================================================
# Plot
# ======================================================================
_FULL_COLOR = "#2ecc71"
_BASE_COLOR = "#7f8c8d"


def plot_baselines_headtohead(battery, filename="baselines_headtohead.png"):
    """Full vs every standard-DQN baseline, in each task's own raw units, with
    95% CI whiskers and Holm-corrected (Wilcoxon) significance stars.
    Left = adaptation latency (↓ better); right = survival total reward (↑ better)."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    stars = _sig_stars(battery)
    labels = list(cfg.BASELINE_CONFIGS.keys())
    colors = [_FULL_COLOR if l == _FULL else _BASE_COLOR for l in labels]

    def stats_for(task, metric):
        means, cis = [], []
        for l in labels:
            m, ci, _ = _mean_ci(battery[task][l][metric])
            means.append(m)
            cis.append(0.0 if np.isnan(ci) else ci)
        return means, cis

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.5))

    # ── Panel A: adaptation latency (lower is better) ──
    lat, lat_ci = stats_for("A_adapt", "Adaptation_Latency")
    barsA = axA.bar(labels, lat, yerr=lat_ci, capsize=5, color=colors,
                    edgecolor="black", error_kw={"elinewidth": 1.4})
    barsA[int(np.argmin(lat))].set_linewidth(2.8)   # best (lowest) latency
    ytopA = max(l + c for l, c in zip(lat, lat_ci)) * 1.30
    for l, b, v, c in zip(labels, barsA, lat, lat_ci):
        axA.annotate(f"{v:.0f}", xy=(b.get_x() + b.get_width() / 2, v + c),
                     xytext=(0, 3), textcoords="offset points", ha="center",
                     va="bottom", fontsize=8, fontweight="bold")
        if l != _FULL:
            axA.annotate(stars[("A_adapt", l)],
                         xy=(b.get_x() + b.get_width() / 2, ytopA * 0.93),
                         ha="center", va="center", fontsize=11,
                         color="#c0392b", fontweight="bold")
    axA.set_ylim(0, ytopA)
    axA.set_ylabel("Adaptation latency (steps)", fontsize=11)
    axA.set_title("Adaptation (Exp 1) — re-lock speed\n"
                  "(lower = faster;  ↓ better)", fontsize=12, fontweight="bold")
    axA.tick_params(axis="x", rotation=25)

    # ── Panel B: survival total reward (higher is better) ──
    rew, rew_ci = stats_for("B_survive", "Total_Reward")
    barsB = axB.bar(labels, rew, yerr=rew_ci, capsize=5, color=colors,
                    edgecolor="black", error_kw={"elinewidth": 1.4})
    barsB[int(np.argmax(rew))].set_linewidth(2.8)   # best (highest) reward
    axB.axhline(0, color="#c0392b", lw=1.4, ls="--", zorder=1)
    lo = min(r - c for r, c in zip(rew, rew_ci))
    hi = max(r + c for r, c in zip(rew, rew_ci))
    rng = hi - lo if hi > lo else 1.0
    top, bot, star_y = hi + rng * 0.30, lo - rng * 0.10, hi + rng * 0.20
    for l, b, v, c in zip(labels, barsB, rew, rew_ci):
        va, off = ("bottom", 4) if v >= 0 else ("top", -4)
        axB.annotate(f"{v:,.0f}",
                     xy=(b.get_x() + b.get_width() / 2, v + (c if v >= 0 else -c)),
                     xytext=(0, off), textcoords="offset points", ha="center",
                     va=va, fontsize=8, fontweight="bold")
        if l != _FULL:
            axB.annotate(stars[("B_survive", l)],
                         xy=(b.get_x() + b.get_width() / 2, star_y),
                         ha="center", va="center", fontsize=11,
                         color="#c0392b", fontweight="bold")
    axB.set_ylim(bot, top)
    axB.set_ylabel("Survival total reward", fontsize=11)
    axB.set_title("Survival (Exp 2) — avoids the lethal arm\n"
                  "(negative = died repeatedly;  ↑ better)",
                  fontsize=12, fontweight="bold")
    axB.tick_params(axis="x", rotation=25)

    fig.suptitle("Full tri-hormone agent vs well-tuned STANDARD DQN baselines",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             f"bars = mean ± 95% CI across {len(cfg.SEEDS)} seeds;  stars = "
             "Holm-corrected Wilcoxon vs Full  (*** p<.001, ** p<.01, * p<.05, "
             "ns = not sig.);  ε-swept + ε-decay test adaptation fairness (C2), "
             "MSE + reward-scaled test survival fairness (H2)",
             ha="center", fontsize=7.5, color="#888", style="italic")
    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    fig.savefig(fpath, dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fpath}")


# ======================================================================
# Console summary
# ======================================================================
def print_baseline_conclusions(battery):
    # NOTE: ASCII only in console output — Windows cp1252 can't encode ↑/↓.
    print("\n" + "=" * 76)
    print("  FAIR BASELINES - does the Full agent beat well-tuned STANDARD RL?")
    print("=" * 76)
    print(f"{'Agent':<20} | {'Adapt lat (lo=better)':>21} | "
          f"{'Survival rew (hi=better)':>24}")
    print("-" * 76)
    for l in cfg.BASELINE_CONFIGS:
        lat, _, _ = _mean_ci(battery["A_adapt"][l]["Adaptation_Latency"])
        rew, _, _ = _mean_ci(battery["B_survive"][l]["Total_Reward"])
        print(f"{l:<20} | {lat:>11.1f} | {rew:>17,.0f}")
    print("-" * 76)
    best_adapt = min((l for l in cfg.BASELINE_CONFIGS if l != _FULL),
                     key=lambda l: _mean_ci(battery["A_adapt"][l]["Adaptation_Latency"])[0])
    best_surv = max((l for l in cfg.BASELINE_CONFIGS if l != _FULL),
                    key=lambda l: _mean_ci(battery["B_survive"][l]["Total_Reward"])[0])
    print(f"  Best standard-DQN baseline on adaptation: {best_adapt}")
    print(f"  Best standard-DQN baseline on survival:   {best_surv}")
    print("  Claim holds iff Full beats BOTH of these (see baselines_pvalues.csv).")
    print("=" * 76)


def run_baseline_study(seeds=None, n_workers=1, force_cpu=True):
    """End-to-end: run the battery, write CSVs, render the head-to-head figure."""
    battery = run_baseline_battery(seeds=seeds, n_workers=n_workers,
                                   force_cpu=force_cpu)
    summarize_baselines(battery)
    baseline_pvalues(battery)
    plot_baselines_headtohead(battery)
    print_baseline_conclusions(battery)
    return battery


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fair-baseline battery vs standard DQN")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    a = p.parse_args()
    run_baseline_study(seeds=a.seeds, n_workers=a.workers, force_cpu=not a.gpu)

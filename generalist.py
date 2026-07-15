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

Scoring method (task-anchored ABSOLUTE competence, not cross-agent min-max)
--------------------------------------------------------------------------
Each task is scored in [0,1] against its OWN theoretical worst case — a task
constant, never the other agents — so the numbers are honest and non-circular
(an earlier min-max normalization forced the worst agent to exactly 0 on each
axis, which manufactured a 0-vs-1 bar chart and hid the real magnitudes):

    adaptation competence = 1 - latency / mean_phase_length   (re-lock window
                            is the exact worst-case latency; Exp 1 ~875 steps)
    survival   competence = 1 - deaths  / (steps x death_prob) (max death
                            exposure = always pulling the lethal arm)

Outputs (research_results/):
    generalist_summary.csv       per (agent, task, metric) mean ± 95% CI
    generalist_pvalues.csv       paired Full-vs-specialist tests (Holm-corrected)
    generalist_headtohead.png    single-modulator vs Full in RAW units (latency
                                 & survival reward) with 95% CI + significance
    generalist_scatter.png       adaptation × survival competence quadrant
    generalist_floor.png         per-task competence + worst-task floor per agent
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
            t_stat, p_val, p_w = _paired_p(f_arr, o_arr)
            full_better = (f_arr.mean() < o_arr.mean()) if lower_better \
                else (f_arr.mean() > o_arr.mean())
            rows.append({"Task": _TASK_LABEL[task], "Metric": metric,
                         "Comparison": f"{_FULL} vs {spec}",
                         "Full_mean": round(float(f_arr.mean()), 3),
                         "Spec_mean": round(float(o_arr.mean()), 3),
                         "Full_better": bool(full_better),
                         "t": round(float(t_stat), 3),
                         "p_value": round(float(p_val), 6),
                         "p_wilcoxon": round(float(p_w), 6),
                         "N": len(pairs)})

    if rows:
        adj = _holm([r["p_value"] for r in rows])
        adj_w = _holm([r["p_wilcoxon"] for r in rows])
        for r, pa, pwa in zip(rows, adj, adj_w):
            r["p_holm"] = round(float(pa), 6)
            r["p_wilcoxon_holm"] = round(float(pwa), 6)
            r["sig_holm_0.05"] = bool(pa < 0.05 and r["Full_better"])
            r["sig_wilcoxon_holm_0.05"] = bool(pwa < 0.05 and r["Full_better"])

    df = pd.DataFrame(rows)
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    df.to_csv(fpath, index=False)
    print(f"  [Saved] {fpath}")
    return df


# ======================================================================
# Normalized scores (for the quadrant scatter + floor bar)
# ======================================================================
def _adapt_denom():
    """Worst-case mean re-lock latency for Exp 1 = the mean phase length.
    An agent that never re-locks after a switch is capped at the phase length,
    so the mean phase length is the exact theoretical worst-case mean latency —
    a task constant, independent of any agent."""
    sw = sorted(cfg.EXP1_SWITCH_STEPS)
    bounds = sw + [cfg.EXP1_TOTAL_STEPS]
    phase_lens = [bounds[i + 1] - bounds[i] for i in range(len(sw))]
    return float(np.mean(phase_lens))


def _survive_denom():
    """Maximum death exposure for Exp 2 = pulling the lethal arm every step
    (expected deaths = steps × death_prob). A task constant, not an agent."""
    return float(cfg.EXP2_TOTAL_STEPS * cfg.EXP2_RISKY_DEATH_P)


def _competence(battery):
    """Per-agent ABSOLUTE competence on each task, scored against the task's own
    theoretical worst case (see module docstring) — NOT min-max across agents.

    Returns {label: {adapt, adapt_ci, survive, survive_ci, floor,
                     lat, lat_ci, rew, rew_ci, deaths, deaths_ci}} where the
    competence CIs are the raw-metric CIs mapped through the (affine) scoring.
    """
    labels = list(cfg.GENERALIST_CONFIGS.keys())
    d_adapt, d_surv = _adapt_denom(), _survive_denom()
    out = {}
    for l in labels:
        lat, lat_ci, _ = _mean_ci(battery["A_adapt"][l]["Adaptation_Latency"])
        rew, rew_ci, _ = _mean_ci(battery["B_survive"][l]["Total_Reward"])
        dth, dth_ci, _ = _mean_ci(battery["B_survive"][l]["Death_Count"])
        adapt = float(np.clip(1.0 - lat / d_adapt, 0.0, 1.0))
        survive = float(np.clip(1.0 - dth / d_surv, 0.0, 1.0))
        out[l] = {"adapt": adapt, "adapt_ci": lat_ci / d_adapt,
                  "survive": survive, "survive_ci": dth_ci / d_surv,
                  "floor": min(adapt, survive),
                  "lat": lat, "lat_ci": lat_ci,
                  "rew": rew, "rew_ci": rew_ci,
                  "deaths": dth, "deaths_ci": dth_ci}
    return out


def _star(p):
    """Significance marker for a (Holm-adjusted) p-value."""
    return ("***" if p < 0.001 else "**" if p < 0.01
            else "*" if p < 0.05 else "ns")


def _sig_stars(battery):
    """Holm-corrected significance of Full vs each specialist on each task's
    headline raw metric. Returns {(task, label): stars}."""
    specialists = [l for l in cfg.GENERALIST_CONFIGS if l != _FULL]
    entries, pvals = [], []
    for task in _TASK_RUNNER:
        metric = _TASK_METRIC[task][0]
        fv = np.asarray(battery[task][_FULL][metric], dtype=float)
        for spec in specialists:
            sv = np.asarray(battery[task][spec][metric], dtype=float)
            _, p = stats.ttest_rel(fv, sv)
            entries.append((task, spec))
            pvals.append(p)
    adj = _holm(pvals)
    return {e: _star(pa) for e, pa in zip(entries, adj)}


# ======================================================================
# Plots
# ======================================================================
_COLORS = {"Full Model": "#2ecc71", "DA only": "#f39c12",
           "NA only": "#3498db", "5-HT only": "#9b59b6"}


def plot_generalist_scatter(battery, filename="generalist_scatter.png"):
    """Adaptation × Survival competence quadrant (absolute [0,1] scores): Full
    alone in the good-good corner, each specialist stranded on one failing
    edge. Axes are task-anchored competence, so a point near 0 means a genuine
    collapse on that task, not merely 'worst of these four agents'."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    comp = _competence(battery)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    # Shade the "generalist zone" (competent on both) top-right.
    ax.axhspan(0.5, 1.05, xmin=0.0, xmax=1.0, color="#2ecc71", alpha=0.04)
    ax.axvspan(0.5, 1.05, ymin=0.0, ymax=1.0, color="#2ecc71", alpha=0.04)
    ax.axhline(0.5, color="#bbbbbb", lw=1, ls="--", zorder=1)
    ax.axvline(0.5, color="#bbbbbb", lw=1, ls="--", zorder=1)

    for label, c in comp.items():
        is_full = (label == _FULL)
        ax_n, sv_n = c["adapt"], c["survive"]
        # 95% CI whiskers (per-seed spread, mapped into competence units).
        ax.errorbar(ax_n, sv_n, xerr=c["adapt_ci"], yerr=c["survive_ci"],
                    fmt="none", ecolor=_COLORS.get(label, "#555"),
                    elinewidth=1.4, capsize=3, capthick=1.4, alpha=0.55,
                    zorder=3)
        ax.scatter(ax_n, sv_n, s=520 if is_full else 300,
                   marker="*" if is_full else "o",
                   color=_COLORS.get(label, "#555"),
                   edgecolor="black", linewidth=1.6 if is_full else 1.0,
                   zorder=5)
        ax.annotate(f"{label}\n(lat {c['lat']:.0f}, deaths {c['deaths']:.0f})",
                    xy=(ax_n, sv_n), xytext=(0, 16 if sv_n < 0.9 else -34),
                    textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold" if is_full else "normal")

    ax.text(0.60, 0.66, "GENERALIST\n(good at both)", ha="center", va="center",
            fontsize=10, color="#2ecc71", fontweight="bold", alpha=0.8)
    ax.text(0.50, 0.30, "adapts but\nDIES", ha="center", va="center",
            fontsize=9, color="#c0392b", alpha=0.7)
    ax.text(0.03, 0.66, "survives but\nSLOW", ha="left", va="center",
            fontsize=9, color="#c0392b", alpha=0.7)

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlabel("Adaptation competence  (1 − latency/window, Exp 1)",
                  fontsize=11)
    ax.set_ylabel("Survival competence  (1 − deaths/max-exposure, Exp 2)",
                  fontsize=11)
    ax.set_title("Integration beats single-modulator gating\n"
                 "Only the Full tri-hormone agent is competent on both tasks",
                 fontsize=12, fontweight="bold")
    ax.text(0.99, -0.075, f"whiskers = 95% CI across {len(cfg.SEEDS)} seeds",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#888", style="italic")
    plt.tight_layout()
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    fig.savefig(fpath, dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fpath}")


def plot_generalist_headtohead(battery, filename="generalist_headtohead.png"):
    """The supervisor's ask: a DIRECT single-modulator-baseline vs Full-model
    comparison, in each task's own RAW units, with 95% CI whiskers and
    Holm-corrected significance stars. Left = adaptation latency (↓ better);
    right = survival total reward (↑ better, with a death line at 0)."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    comp = _competence(battery)
    stars = _sig_stars(battery)
    labels = list(cfg.GENERALIST_CONFIGS.keys())
    colors = [_COLORS.get(l, "#555") for l in labels]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6))

    # ── Panel A: adaptation latency (lower is better) ──
    lat = [comp[l]["lat"] for l in labels]
    lat_ci = [comp[l]["lat_ci"] for l in labels]
    barsA = axA.bar(labels, lat, yerr=lat_ci, capsize=6, color=colors,
                    edgecolor="black", error_kw={"elinewidth": 1.6})
    barsA[int(np.argmin(lat))].set_linewidth(2.8)
    ytopA = max(l + c for l, c in zip(lat, lat_ci)) * 1.32
    star_yA = ytopA * 0.93
    for l, b, v, c in zip(labels, barsA, lat, lat_ci):
        axA.annotate(f"{v:.0f}", xy=(b.get_x() + b.get_width() / 2, v + c),
                     xytext=(0, 3), textcoords="offset points", ha="center",
                     va="bottom", fontsize=9, fontweight="bold")
        if l != _FULL:
            axA.annotate(stars[("A_adapt", l)],
                         xy=(b.get_x() + b.get_width() / 2, star_yA),
                         ha="center", va="center", fontsize=12,
                         color="#c0392b", fontweight="bold")
    axA.set_ylim(0, ytopA)
    axA.set_ylabel("Adaptation latency (steps)", fontsize=11)
    axA.set_title("Adaptation (Exp 1) — re-lock speed\n(lower = faster;  ↓ better)",
                  fontsize=12, fontweight="bold")
    axA.tick_params(axis="x", rotation=12)

    # ── Panel B: survival total reward (higher is better) ──
    rew = [comp[l]["rew"] for l in labels]
    rew_ci = [comp[l]["rew_ci"] for l in labels]
    barsB = axB.bar(labels, rew, yerr=rew_ci, capsize=6, color=colors,
                    edgecolor="black", error_kw={"elinewidth": 1.6})
    barsB[int(np.argmax(rew))].set_linewidth(2.8)
    axB.axhline(0, color="#c0392b", lw=1.4, ls="--", zorder=1)
    lo = min(r - c for r, c in zip(rew, rew_ci))
    hi = max(r + c for r, c in zip(rew, rew_ci))
    rng = hi - lo
    top = hi + rng * 0.28   # headroom for a clean star row above every label
    bot = lo - rng * 0.10
    star_yB = hi + rng * 0.20
    for l, b, v, c in zip(labels, barsB, rew, rew_ci):
        va, off = ("bottom", 4) if v >= 0 else ("top", -4)
        axB.annotate(f"{v:,.0f}",
                     xy=(b.get_x() + b.get_width() / 2, v + (c if v >= 0 else -c)),
                     xytext=(0, off), textcoords="offset points", ha="center",
                     va=va, fontsize=9, fontweight="bold")
        if l != _FULL:
            axB.annotate(stars[("B_survive", l)],
                         xy=(b.get_x() + b.get_width() / 2, star_yB),
                         ha="center", va="center", fontsize=12,
                         color="#c0392b", fontweight="bold")
    axB.set_ylim(bot, top)
    axB.set_ylabel("Survival total reward", fontsize=11)
    axB.set_title("Survival (Exp 2) — avoids the lethal arm\n"
                  "(negative = died repeatedly;  ↑ better)",
                  fontsize=12, fontweight="bold")
    axB.tick_params(axis="x", rotation=12)

    fig.suptitle("Single-modulator baselines vs the Full tri-hormone agent",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             f"bars = mean ± 95% CI across {len(cfg.SEEDS)} seeds;  "
             "stars = Holm-corrected paired t-test vs Full  "
             "(*** p<.001, ** p<.01, * p<.05, ns = not sig.)",
             ha="center", fontsize=8, color="#888", style="italic")
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    fig.savefig(fpath, dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fpath}")


def plot_generalist_floor(battery, filename="generalist_floor.png"):
    """Two-panel headline. Left: per-task ABSOLUTE competence (adaptation vs
    survival) side by side, so you can see which task each specialist fails.
    Right: the worst-task FLOOR (min of the two) — the single generalist number.
    Scores are task-anchored (see _competence), so a low bar is a real collapse,
    not an artefact of normalizing against the other agents."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    comp = _competence(battery)
    labels = list(cfg.GENERALIST_CONFIGS.keys())
    colors = [_COLORS.get(l, "#555") for l in labels]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # ── Left: grouped per-task competence ──
    x = np.arange(len(labels))
    w = 0.38
    adapt = [comp[l]["adapt"] for l in labels]
    surv = [comp[l]["survive"] for l in labels]
    adapt_ci = [comp[l]["adapt_ci"] for l in labels]
    surv_ci = [comp[l]["survive_ci"] for l in labels]
    axL.bar(x - w / 2, adapt, w, yerr=adapt_ci, capsize=4, color="#5dade2",
            edgecolor="black", label="Adaptation (Exp 1)")
    axL.bar(x + w / 2, surv, w, yerr=surv_ci, capsize=4, color="#e59866",
            edgecolor="black", label="Survival (Exp 2)")
    for xi, (a, s, ac, sc) in enumerate(zip(adapt, surv, adapt_ci, surv_ci)):
        axL.annotate(f"{a:.2f}", xy=(xi - w / 2, a + ac), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=8)
        axL.annotate(f"{s:.2f}", xy=(xi + w / 2, s + sc), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=8)
    axL.set_xticks(x)
    axL.set_xticklabels(labels, rotation=12)
    axL.set_ylim(0, 1.18)
    axL.set_ylabel("Task competence  (1 = solves the task)", fontsize=11)
    axL.set_title("Competence per task\n(each specialist collapses on one)",
                  fontsize=12, fontweight="bold")
    axL.legend(loc="upper center", fontsize=9, ncol=2)

    # ── Right: the worst-task floor ──
    floor = [comp[l]["floor"] for l in labels]
    bars = axR.bar(labels, floor, color=colors, edgecolor="black")
    best_i = int(np.argmax(floor))
    bars[best_i].set_linewidth(2.8)
    for j, (b, v) in enumerate(zip(bars, floor)):
        tag = "  (best)" if j == best_i else ""
        axR.annotate(f"{v:.2f}{tag}",
                     xy=(b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axR.set_ylabel("Worst-task competence  (min over both tasks)", fontsize=11)
    axR.set_ylim(0, 1.12)
    axR.set_title("Generalist floor\n(competence on your WEAKEST task)",
                  fontsize=12, fontweight="bold")
    axR.tick_params(axis="x", rotation=12)

    fig.suptitle("Only the Full agent stays competent on its weakest task",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             f"absolute task-anchored competence, mean across {len(cfg.SEEDS)} "
             "seeds  (adaptation = 1 − latency/window;  survival = 1 − deaths/"
             "max-exposure)",
             ha="center", fontsize=8, color="#888", style="italic")
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    fig.savefig(fpath, dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fpath}")


# ======================================================================
# Console summary
# ======================================================================
def print_generalist_conclusions(battery):
    comp = _competence(battery)
    print("\n" + "=" * 74)
    print("  GENERALIST vs SPECIALISTS  (integration beats single-modulator)")
    print("  competence = task-anchored absolute score in [0,1] (not min-max)")
    print("=" * 74)
    print(f"{'Agent':<12} | {'Adapt lat':>9} {'(comp)':>7} | "
          f"{'Deaths':>7} {'(comp)':>7} | {'floor':>6}")
    print("-" * 74)
    for l in cfg.GENERALIST_CONFIGS:
        c = comp[l]
        print(f"{l:<12} | {c['lat']:>9.0f} {c['adapt']:>7.2f} | "
              f"{c['deaths']:>7.0f} {c['survive']:>7.2f} | {c['floor']:>6.2f}")
    print("-" * 74)
    winner = max(cfg.GENERALIST_CONFIGS, key=lambda l: comp[l]["floor"])
    print(f"  Highest floor (best generalist): {winner}")
    print("  Interpretation: each specialist is competent on ONE task and")
    print("  collapses on the other; only the Full agent keeps a high floor.")
    print("=" * 74)


def run_generalist_study(seeds=None, n_workers=1, force_cpu=True):
    """End-to-end: run the battery, write CSVs, and render both figures."""
    battery = run_generalist_battery(seeds=seeds, n_workers=n_workers,
                                     force_cpu=force_cpu)
    summarize_generalist(battery)
    generalist_pvalues(battery)
    plot_generalist_headtohead(battery)
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

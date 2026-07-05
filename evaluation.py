"""
evaluation.py — Visualization Pipeline & Statistics (Section 8)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import config as cfg

sns.set_theme(style="darkgrid", palette="deep", font_scale=1.1,
              rc={"figure.dpi": cfg.PLOT_DPI, "savefig.dpi": cfg.PLOT_DPI,
                  "font.family": "serif"})


def _ensure_dir():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)


def _rolling(data, window=cfg.ROLLING_WINDOW):
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        window = max(1, len(arr))
    return pd.Series(arr).rolling(window, min_periods=1).mean().values


def plot_experiment_1(results: dict, suffix: str = ""):
    """4-panel dashboard for Exp 1 — Volatile Bandit."""
    _ensure_dir()
    label = results["label"]
    fig, axes = plt.subplots(2, 2, figsize=cfg.FIG_SIZE)
    fig.suptitle(f"Experiment 1 — Volatile Bandit  [{label}]",
                 fontsize=16, fontweight="bold")
    steps = np.arange(len(results["rewards"]))

    # Panel 1: Reward
    ax = axes[0, 0]
    ax.plot(steps, np.cumsum(results["rewards"]), alpha=0.3,
            label="Cumulative", color="steelblue")
    ax.plot(steps, _rolling(results["rewards"]), linewidth=2,
            label=f"{cfg.ROLLING_WINDOW}-step avg", color="navy")
    for s in cfg.EXP1_SWITCH_STEPS:
        ax.axvline(s, color="red", ls="--", alpha=0.8)
    ax.set(xlabel="Step", ylabel="Reward", title="Reward Dynamics")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Hormones
    ax = axes[0, 1]
    h = np.arange(len(results["hormones_da"]))
    ax.plot(h, results["hormones_da_eff"], label="DA(eff)", color="gold")
    ax.plot(h, results["hormones_na"], label="NA", color="crimson")
    ax.plot(h, results["hormones_ht"], label="5-HT", color="seagreen")
    for s in cfg.EXP1_SWITCH_STEPS:
        ax.axvline(s, color="red", ls="--", alpha=0.5)
    ax.set(xlabel="Step", ylabel="Conc.", title="Neuromodulator Levels")
    ax.legend(fontsize=8)

    # Panel 3: Hyperparams
    ax = axes[1, 0]
    hp = np.arange(len(results["alpha"]))
    ax.plot(hp, results["alpha"], label="α", color="purple")
    ax.plot(hp, results["epsilon"], label="ε", color="orange")
    ax.plot(hp, results["gamma"], label="γ", color="teal")
    for s in cfg.EXP1_SWITCH_STEPS:
        ax.axvline(s, color="red", ls="--", alpha=0.5)
    ax.set(xlabel="Step", ylabel="Value", title="Hyperparameter Dynamics")
    ax.legend(fontsize=8)

    # Panel 4: Action heatmap
    ax = axes[1, 1]
    actions = np.array(results["actions"])
    ws = 50
    nw = len(actions) // ws
    hist = np.zeros((cfg.EXP1_N_ARMS, nw))
    for w in range(nw):
        c = actions[w * ws:(w + 1) * ws]
        for a in range(cfg.EXP1_N_ARMS):
            hist[a, w] = np.sum(c == a) / ws
    sns.heatmap(hist, ax=ax, cmap="YlOrRd",
                yticklabels=[f"Arm {i}" for i in range(cfg.EXP1_N_ARMS)],
                cbar_kws={"label": "Freq"})
    for s in cfg.EXP1_SWITCH_STEPS:
        ax.axvline(s // ws, color="cyan", lw=2, ls="--")
    ax.set(xlabel=f"Window ({ws} steps)", title="Action Distribution")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join(cfg.RESULTS_DIR, f"exp1_dashboard{suffix}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fname}")


def plot_experiment_2(results: dict, suffix: str = ""):
    """4-panel dashboard for Exp 2 — High-Stakes Foraging."""
    _ensure_dir()
    label = results["label"]
    fig, axes = plt.subplots(2, 2, figsize=cfg.FIG_SIZE)
    fig.suptitle(f"Experiment 2 — High-Stakes Foraging  [{label}]",
                 fontsize=16, fontweight="bold")
    steps = np.arange(len(results["rewards"]))

    ax = axes[0, 0]
    ax.plot(steps, results["cumulative_rewards"], color="steelblue", lw=1.5)
    for d in results["death_events"]:
        ax.axvline(d, color="red", alpha=0.3, lw=0.5)
    ax.set(xlabel="Step", ylabel="Cumul. Reward",
           title=f"Score (Deaths={results['death_count']})")

    ax = axes[0, 1]
    h = np.arange(len(results["hormones_da"]))
    ax.plot(h, results["hormones_da_eff"], label="DA(eff)", color="gold")
    ax.plot(h, results["hormones_na"], label="NA", color="crimson")
    ax.plot(h, results["hormones_ht"], label="5-HT", color="seagreen")
    ax.set(xlabel="Step", ylabel="Conc.", title="Neuromodulator Levels")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    hp = np.arange(len(results["alpha"]))
    ax.plot(hp, results["alpha"], label="α", color="purple")
    ax.plot(hp, results["epsilon"], label="ε", color="orange")
    ax.plot(hp, results["gamma"], label="γ", color="teal")
    ax.set(xlabel="Step", ylabel="Value", title="Hyperparameter Dynamics")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    actions = np.array(results["actions"])
    ws = 100
    nw = len(actions) // ws
    safe_pct = [np.sum(actions[w*ws:(w+1)*ws] == 0) / ws for w in range(nw)]
    ax.fill_between(range(nw), safe_pct, alpha=0.4, color="seagreen",
                    label="Safe %")
    ax.fill_between(range(nw), [1-s for s in safe_pct], [1]*nw,
                    alpha=0.4, color="crimson", label="Risky %")
    ax.set(xlabel=f"Window ({ws} steps)", ylabel="Proportion",
           title="Safe vs. Risky")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join(cfg.RESULTS_DIR, f"exp2_dashboard{suffix}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fname}")


def plot_experiment_3(results: dict, suffix: str = ""):
    """4-panel dashboard for Exp 3 — CartPole."""
    _ensure_dir()
    label = results["label"]
    fig, axes = plt.subplots(2, 2, figsize=cfg.FIG_SIZE)
    fig.suptitle(f"Experiment 3 — CartPole Adaptation  [{label}]",
                 fontsize=16, fontweight="bold")
    eps = np.arange(len(results["episode_lengths"]))

    ax = axes[0, 0]
    ax.plot(eps, results["episode_lengths"], alpha=0.4, color="steelblue")
    ax.plot(eps, _rolling(results["episode_lengths"], 10), lw=2,
            color="navy", label="10-ep avg")
    ax.axvline(results["perturb_episode"], color="red", ls="--",
               label="Perturbation")
    ax.axhline(cfg.EXP3_RECOVERY_TARGET, color="green", ls=":", alpha=0.5)
    ax.set(xlabel="Episode", ylabel="Steps", title="Episode Duration")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    h = np.arange(len(results["hormones_da"]))
    s = max(1, len(h) // 2000)
    ax.plot(h[::s], np.array(results["hormones_da_eff"])[::s],
            label="DA(eff)", color="gold", lw=0.8)
    ax.plot(h[::s], np.array(results["hormones_na"])[::s],
            label="NA", color="crimson", lw=0.8)
    ax.plot(h[::s], np.array(results["hormones_ht"])[::s],
            label="5-HT", color="seagreen", lw=0.8)
    ax.set(xlabel="Timestep", ylabel="Conc.", title="Neuromodulators")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    hp = np.arange(len(results["alpha"]))
    ax.plot(hp[::s], np.array(results["alpha"])[::s],
            label="α", color="purple", lw=0.8)
    ax.plot(hp[::s], np.array(results["epsilon"])[::s],
            label="ε", color="orange", lw=0.8)
    ax.plot(hp[::s], np.array(results["gamma"])[::s],
            label="γ", color="teal", lw=0.8)
    ax.set(xlabel="Timestep", ylabel="Value", title="Hyperparameters")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(eps, results["rewards"], color="steelblue", alpha=0.5, width=1.0)
    ax.axvline(results["perturb_episode"], color="red", ls="--")
    ax.set(xlabel="Episode", ylabel="Reward", title="Episode Reward")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join(cfg.RESULTS_DIR, f"exp3_dashboard{suffix}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fname}")


def plot_regret_curve(results: dict, optimal_reward: float,
                      exp_name: str, suffix: str = ""):
    """Cumulative regret vs theoretical optimum."""
    _ensure_dir()
    rewards = np.array(results["rewards"])
    regret = np.cumsum(np.full_like(rewards, optimal_reward) - rewards)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(regret, color="firebrick", lw=1.5)
    ax.fill_between(range(len(regret)), regret, alpha=0.2, color="firebrick")
    ax.set(xlabel="Step", ylabel="Cumulative Regret",
           title=f"Regret — {exp_name} [{results['label']}]")
    fname = os.path.join(cfg.RESULTS_DIR, f"{exp_name}_regret{suffix}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fname}")


def export_csv(all_results: dict, filename: str = "experiment_results.csv"):
    """Export results to CSV for statistical testing."""
    _ensure_dir()
    rows = []
    for exp_name, configs in all_results.items():
        for cfg_label, res in configs.items():
            row = {"Experiment": exp_name, "Configuration": cfg_label,
                   "Total_Reward": sum(res.get("rewards", []))}
            if "adaptation_latency" in res:
                row["Adaptation_Latency"] = res["adaptation_latency"]
            if "death_count" in res:
                row["Death_Count"] = res["death_count"]
                row["Survival_Rate"] = np.mean(res.get("survival_steps", [0]))
            if "recovery_time" in res:
                row["Recovery_Time"] = res["recovery_time"]
            rows.append(row)
    df = pd.DataFrame(rows)
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    df.to_csv(fpath, index=False)
    print(f"  [Saved] {fpath}")
    return df


# ─────────────────────────────────────────────────────────────────────
# Multi-seed statistics
# ─────────────────────────────────────────────────────────────────────
# Metric directionality: is a LOWER value the BETTER agent?
_LOWER_IS_BETTER = {"Adaptation_Latency", "Death_Count", "Recovery_Time"}


def scalar_metrics(res: dict) -> dict:
    """Reduce one experiment run to its headline scalar metrics."""
    m = {"Total_Reward": float(np.sum(res.get("rewards", []) or [0.0]))}
    if "adaptation_latency" in res:
        m["Adaptation_Latency"] = float(res["adaptation_latency"])
    if "death_count" in res:
        m["Death_Count"] = float(res["death_count"])
        m["Survival_Rate"] = float(np.mean(res.get("survival_steps", [0])))
    if "recovery_time" in res:
        m["Recovery_Time"] = float(res["recovery_time"])  # may be NaN
    return m


def _metric_vectors(res_list: list) -> dict:
    """Stack per-seed scalar metrics into {metric: [v_seed0, v_seed1, ...]}."""
    vecs = {}
    for res in res_list:
        for k, v in scalar_metrics(res).items():
            vecs.setdefault(k, []).append(v)
    return vecs


def _holm(pvals):
    """Holm–Bonferroni step-down adjusted p-values (returned in input order).

    Controls the family-wise error rate across the multiple Full-vs-config
    comparisons within an experiment, without assuming independence. Less
    conservative than plain Bonferroni.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])  # enforce monotonic step-down
        adj[i] = min(running, 1.0)
    return adj


def _mean_ci(vec):
    """Mean and 95% (t-based) CI half-width, ignoring NaNs."""
    a = np.array([v for v in vec if not np.isnan(v)], dtype=float)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(a.mean())
    if n == 1:
        return mean, float("nan"), 1
    sem = float(a.std(ddof=1) / np.sqrt(n))
    ci = float(stats.t.ppf(0.975, n - 1) * sem)
    return mean, ci, n


def summarize_multiseed(multiseed_results: dict,
                        filename: str = "summary_multiseed.csv"):
    """Per-(experiment, config, metric) mean ± 95% CI across seeds."""
    _ensure_dir()
    rows = []
    for exp_name, configs in multiseed_results.items():
        for cfg_label, res_list in configs.items():
            for metric, vec in _metric_vectors(res_list).items():
                mean, ci, n = _mean_ci(vec)
                # Recovery_Time is only defined for pre-shock-competent seeds, and
                # DA suppresses competence, so some configs have too few competent
                # seeds to report a meaningful mean. Suppress the point estimate
                # below MIN_RECOVERY_SEEDS (keep the true N so it is visibly
                # undefined, not silently missing).
                if (metric == "Recovery_Time"
                        and n < cfg.MIN_RECOVERY_SEEDS):
                    mean, ci = float("nan"), float("nan")
                rows.append({"Experiment": exp_name, "Configuration": cfg_label,
                             "Metric": metric,
                             "Mean": (round(mean, 4) if not np.isnan(mean)
                                      else float("nan")),
                             "CI95": (round(ci, 4) if not np.isnan(ci)
                                      else float("nan")),
                             "N_seeds": n})
    df = pd.DataFrame(rows)
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    df.to_csv(fpath, index=False)
    print(f"  [Saved] {fpath}")
    return df


def compute_multiseed_pvalues(multiseed_results: dict,
                              filename: str = "pvalues.csv"):
    """Paired significance tests: Full Model vs each other config, per metric.

    Seeds are matched across configs, so we use a paired t-test on the
    per-seed metric vectors (dropping any seed where either side is NaN,
    e.g. an undefined CartPole recovery). This replaces the earlier
    single-seed window-chunking test, which was pseudoreplication.

    Because each experiment runs many comparisons (Full vs each config × each
    metric), we additionally report **Holm–Bonferroni adjusted p-values**
    (`p_holm`, `sig_holm_0.05`) computed within each experiment family, so the
    marginal results are held to the corrected threshold. Recovery_Time
    comparisons with fewer than MIN_RECOVERY_SEEDS competent pairs are skipped
    (too few competent seeds to be meaningful).
    """
    _ensure_dir()
    rows = []
    for exp_name, configs in multiseed_results.items():
        if "Full Model" not in configs:
            continue
        full_vecs = _metric_vectors(configs["Full Model"])
        for cfg_label, res_list in configs.items():
            if cfg_label == "Full Model":
                continue
            other_vecs = _metric_vectors(res_list)
            for metric, full_vec in full_vecs.items():
                other_vec = other_vecs.get(metric)
                if other_vec is None:
                    continue
                pairs = [(f, o) for f, o in zip(full_vec, other_vec)
                         if not (np.isnan(f) or np.isnan(o))]
                # Recovery_Time needs enough competent-in-both seeds to be
                # meaningful; other metrics only need a valid pair.
                min_pairs = (cfg.MIN_RECOVERY_SEEDS
                             if metric == "Recovery_Time" else 2)
                if len(pairs) < min_pairs:
                    continue
                f_arr = np.array([p[0] for p in pairs])
                o_arr = np.array([p[1] for p in pairs])
                if np.allclose(f_arr, o_arr):
                    t, p = 0.0, 1.0
                else:
                    t, p = stats.ttest_rel(f_arr, o_arr)
                mean_diff = float(f_arr.mean() - o_arr.mean())
                lower_better = metric in _LOWER_IS_BETTER
                full_better = (mean_diff < 0) if lower_better else (mean_diff > 0)
                rows.append({"Experiment": exp_name,
                             "Comparison": f"Full vs {cfg_label}",
                             "Metric": metric,
                             "Full_Mean": round(float(f_arr.mean()), 3),
                             "Other_Mean": round(float(o_arr.mean()), 3),
                             "t_stat": round(float(t), 4),
                             "p_value": round(float(p), 6),
                             "sig_0.05": bool(p < 0.05),
                             "Full_Better": bool(full_better),
                             "N_pairs": len(pairs)})

    # ── Holm–Bonferroni correction WITHIN each experiment's family of tests ──
    # Each experiment runs several Full-vs-config comparisons across several
    # metrics; correcting per experiment guards against false positives from the
    # multiple comparisons while keeping the families scientifically coherent
    # (one family = one experiment's hypothesis tests).
    from collections import defaultdict
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[r["Experiment"]].append(r)
    for exp_rows in by_exp.values():
        adj = _holm([r["p_value"] for r in exp_rows])
        for r, pa in zip(exp_rows, adj):
            r["p_holm"] = round(float(pa), 6)
            r["sig_holm_0.05"] = bool(pa < 0.05)

    df = pd.DataFrame(rows)
    fpath = os.path.join(cfg.RESULTS_DIR, filename)
    df.to_csv(fpath, index=False)
    print(f"  [Saved] {fpath}")
    if not df.empty:
        print(df.to_string(index=False))
    return df

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
    ax.axvline(cfg.EXP1_SWITCH_STEP, color="red", ls="--",
               label="Switch")
    ax.set(xlabel="Step", ylabel="Reward", title="Reward Dynamics")
    ax.legend(fontsize=8)

    # Panel 2: Hormones
    ax = axes[0, 1]
    h = np.arange(len(results["hormones_da"]))
    ax.plot(h, results["hormones_da_eff"], label="DA(eff)", color="gold")
    ax.plot(h, results["hormones_na"], label="NA", color="crimson")
    ax.plot(h, results["hormones_ht"], label="5-HT", color="seagreen")
    ax.axvline(cfg.EXP1_SWITCH_STEP, color="red", ls="--", alpha=0.5)
    ax.set(xlabel="Step", ylabel="Conc.", title="Neuromodulator Levels")
    ax.legend(fontsize=8)

    # Panel 3: Hyperparams
    ax = axes[1, 0]
    hp = np.arange(len(results["alpha"]))
    ax.plot(hp, results["alpha"], label="α", color="purple")
    ax.plot(hp, results["tau"], label="τ", color="orange")
    ax.plot(hp, results["gamma"], label="γ", color="teal")
    ax.axvline(cfg.EXP1_SWITCH_STEP, color="red", ls="--", alpha=0.5)
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
    ax.axvline(cfg.EXP1_SWITCH_STEP // ws, color="cyan", lw=2, ls="--")
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
    ax.plot(hp, results["tau"], label="τ", color="orange")
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
    ax.plot(hp[::s], np.array(results["tau"])[::s],
            label="τ", color="orange", lw=0.8)
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


def compute_pvalues(all_results: dict):
    """Welch's t-test between Full Model and each ablation."""
    _ensure_dir()
    rows = []
    for exp_name, configs in all_results.items():
        if "Full Model" not in configs:
            continue
        full = np.array(configs["Full Model"]["rewards"])
        for cfg_label, res in configs.items():
            if cfg_label == "Full Model":
                continue
            abl = np.array(res["rewards"])
            w = 50
            n = min(len(full), len(abl))
            fc = [full[i:i+w].mean() for i in range(0, n-w, w)]
            ac = [abl[i:i+w].mean() for i in range(0, n-w, w)]
            if len(fc) > 1 and len(ac) > 1:
                t, p = stats.ttest_ind(fc, ac, equal_var=False)
            else:
                t, p = 0.0, 1.0
            rows.append({"Experiment": exp_name,
                         "Comparison": f"Full vs {cfg_label}",
                         "t_stat": round(t, 4), "p_value": round(p, 6),
                         "sig_0.05": p < 0.05})
    df = pd.DataFrame(rows)
    fpath = os.path.join(cfg.RESULTS_DIR, "pvalues.csv")
    df.to_csv(fpath, index=False)
    print(f"  [Saved] {fpath}")
    print(df.to_string(index=False))
    return df

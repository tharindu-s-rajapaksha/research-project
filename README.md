# Multi-Neuromodulated Modular RL Architecture

A Bio-inspired Reinforcement Learning framework implementing dynamic neuromodulation and differentiable plasticity to solve complex adaptation tasks.

## 🧠 Project Concept

This project implements a **Multi-Neuromodulated Modular RL Architecture** designed to mimic biological learning mechanisms in the brain. Unlike traditional RL agents with static hyperparameters, this architecture uses a **Hormonal Meta-Agent** to dynamically regulate learning based on environmental cues.

### Hormone Engine (Section 3)
Three neuromodulators with homeostatic spike-and-decay dynamics:
- **Dopamine (DA)**: TD-error → plasticity gate (0 at rest, rises on surprise)
- **Noradrenaline (NA)**: Reward-drop detector → ε-greedy exploration rate (0.01–0.5)
- **Serotonin (5-HT)**: Aversive events → loss aversion gain + behavioral inhibition

### Meta-Agent (Section 4B)
Maps hormone levels to real-time hyperparameter modulation:
- α (learning rate): DA-driven surprise boost (capped 2×)
- ε (exploration): NA-driven rate (ε_base=0.01 → ε_max=0.5)
- γ (discount): 5-HT-driven horizon (γ_base=0.99 → γ_max=0.999)
- Harm-aversion gain: 5-HT multiplier (1 → 4 on losses) + behavioural-inhibition penalty (weight 5)

### Worker (Section 2B)
`LocalRLWorker`: ε-greedy DQN with differentiable Hebbian plasticity (`NeuromodulatedLinear`).

---

## 🚀 Getting Started

### Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch numpy gymnasium matplotlib seaborn pygame
```

## Usage Commands

### Full Multi-Seed Study (10 seeds, all configs, parallel)
```bash
# All 3 experiments
python main.py --workers 12

# Single experiment (outputs: summary_multiseed_exp{N}.csv, pvalues_exp{N}.csv)
python main.py --exp 1 --workers 12
python main.py --exp 2 --workers 12
python main.py --exp 3 --workers 12
```
#### Generalist vs. Specialists Study (10 seeds)
```bash
python main.py --generalist --workers 12
```

#### Fair-Baseline Battery — "beats STANDARD RL?" (10 seeds)
Compares the Full agent against **well-tuned standard DQNs** (swept-ε, ε-decay, and
value-corrected MSE / reward-scaled), not just the near-greedy ε=0.01 baseline. This is the
honest "beats standard RL" test — see `RESEARCH_NOTES.md` §6.6.
```bash
python main.py --baselines --workers 12
```

#### Regenerate result tables + sensitivity probes
```bash
python tools/make_tables.py                       # tables from CSVs -> RESULTS_TABLES.md
python tools/bandit_gamma0_probe.py               # NA robust to gamma=0? (sensitivity)
python tools/da_strong_probe.py                   # DA-strong (exploratory / future work)
```
Workers run on CPU by default (faster for tiny nets than GPU). Results: `research_results/*.csv`, dashboards, regret curves.

*Note: Add `--merge` to any of the above to combine results into a single chart file.*

#### 3. Interactive Simulation Engine
To visualize experiments or run specific analyses. Both arguments are **required**:
```bash
python simulation_engine.py --exp <1|2|3> --mode <live|fast|ablation>
```

**Examples:**
```bash
python simulation_engine.py --exp 3 --mode ablation
python simulation_engine.py --exp 3 --mode live
python simulation_engine.py --exp 3 --mode fast
```

- **Live Mode (`--mode live`)**: Normal speed playback with Pygame visuals.
    - *Controls*: `SPACE` to pause, `UP/DOWN` to adjust speed, `R` to restart simulation.
    - *Live Ablation*: Toggle the **DA, NA, and 5HT checkboxes** at the top of the UI to enable/disable neuromodulation on-the-fly.
    - *Static Baseline*: To run the true Static Baseline (Standard DQN), uncheck all three hormones and press **`R`** to restart.
    - *UI Indicators*: A live status bar indicates whether the **DYNAMIC RL** or **STATIC BASELINE** model is currently running.
- **Fast Mode (`--mode fast`)**: Train a single Full Model in the background at maximum speed.
- **Ablation Mode (`--mode ablation`)**: Run the full 4-config study for the selected experiment.

---

## 📌 Honest results summary

Every number traces to a committed CSV (`research_results/`, regenerate tables with
`python tools/make_tables.py`). Full details + statistics in `RESEARCH_NOTES.md`.

- **Adaptation (Exp 1) — the strongest, cleanest win.** The Full agent re-locks after a reward
  switch ~58% faster than a standard DQN, and faster than the best **swept-ε / ε-decay** DQN
  (fair-baseline study). The ablation isolates this to **noradrenaline** (removing NA nearly
  doubles latency, Holm p=1.7×10⁻⁴; removing dopamine changes nothing).
- **Survival (Exp 2) — a real but *bounded* win.** vs a Huber-loss DQN, 5-HT behavioural
  inhibition cuts deaths ~13×. **Honest bound:** a *value-corrected* DQN (MSE / reward-scaled)
  also avoids the trap without serotonin, so 5-HT **repairs a known DQN loss-function pathology**
  rather than being universally necessary.
- **Integration / novelty (generalist study).** Only the tri-hormone agent is competent across
  *both* an adaptation and a survival niche — **generalist coverage**, not synergy.
- **Dopamine / differentiable plasticity — a documented negative.** Neutral-to-harmful on every
  task tested; the DA-strong probe is an exploratory attempt to find where it helps.

## 🧪 Experiments

### Exp 1: Volatile Multi-Armed Bandit (NA Test)
Tests the agent's ability to detect shifts in reward distributions. Noradrenaline (NA) spikes during volatility to reset internal representations.

### Exp 2: High-Stakes Foraging (5-HT Test)
Tests survival and harm aversion. Serotonin (5-HT) spikes during "near-death" or high-risk scenarios to enforce a safer policy.

### Exp 3: Volatile Risky Foraging (CAPSTONE — integration + survival)
An integrative task that **fuses Exp 1 and Exp 2**: a contextual cue→action mapping that **reverses** over time (volatility → NA/DA), plus a tempting but occasionally **lethal** arm (→ 5-HT). Headline metric: **cumulative reward**. Honest result (see `RESEARCH_NOTES.md` §6.3): the Full agent **massively beats standard RL**, but the win is driven by **5-HT survival** (not dying), with DA giving a marginal re-adaptation-latency benefit and NA no measurable effect here; overall accuracy ≈0.41, so the agent only *partially* learns the reversal mapping. It is an **integration + survival** result, **not** an "all three hormones cooperating" result.

### (Legacy) CartPole Physics Adaptation — secondary / negative result
Kept via `experiments.run_experiment_cartpole` to reproduce the honest finding that DA-gated plasticity *helps* discrete re-mapping but *hurts* stable continuous control. Not part of the default suite.

---

## 📊 Project Structure

- `main.py`: Entry point for full research suite.
- `simulation_engine.py`: Interactive Pygame simulation and visual runner.
- `config.py`: Global hyperparameters and experiment settings.
- `neuromodulators.py`: Core logic for hormone accumulation and decay.
- `meta_agent.py`: Implementation of the Hormonal Meta-Agent.
- `plasticity.py`: Neuromodulated linear layers and Hebbian trace logic.
- `worker.py`: DQN implementation and baseline agents.
- `environments.py`: Custom Gymnasium environments for Bandit and Foraging.
- `ablation.py`: Framework for running comparative (multi-seed) studies.
- `generalist.py`: "Generalist vs specialists" novelty study.
- `baselines.py`: Fair-baseline battery (Full vs well-tuned standard DQNs).
- `evaluation.py`: Statistics (paired t + Wilcoxon, Holm) and dashboard generation.
- `tools/`: `make_tables.py` (regenerate tables from CSVs), `bandit_gamma0_probe.py`, `da_strong_probe.py`.
- `research_results/`: Directory where all dashboards, CSVs, and `RESULTS_TABLES.md` are saved.

---

## 📈 Visualizations

The system generates high-quality dashboards for each run, including:
- Reward curves and survival rates.
- Hormone concentration timelines.
- Dynamic hyperparameter adjustments.
- Comparative bar charts for ablation studies.

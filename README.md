# Multi-Neuromodulated Modular RL Architecture

A Bio-inspired Reinforcement Learning framework implementing dynamic neuromodulation and differentiable plasticity to solve complex adaptation tasks.

## 🧠 Project Concept

This project implements a **Multi-Neuromodulated Modular RL Architecture** designed to mimic biological learning mechanisms in the brain. Unlike traditional RL agents with static hyperparameters, this architecture uses a **Hormonal Meta-Agent** to dynamically regulate learning based on environmental cues.

### Key Components:
- **Hormone Engine**: Simulates the dynamics of three key neuromodulators:
    - **Dopamine (DA)**: Encodes TD-error and scales the three-factor Hebbian learning rule.
    - **Noradrenaline (NA)**: Responds to environmental volatility, increasing exploration and learning rates during shifts.
    - **Serotonin (5-HT)**: Responds to high-risk/aversive events, increasing harm aversion and survival probability.
- **Differentiable Plasticity**: Uses a `NeuromodulatedLinear` layer with per-synapse Hebbian eligibility traces, allowing for rapid weight adaptation beyond standard gradient descent.
- **Hormonal Meta-Agent**: The "conductor" that monitors performance and modulates $\alpha$ (learning rate), $\tau$ (softmax temperature), and $\gamma$ (discount factor) in real-time.

---

## 🚀 Getting Started

### Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch numpy gymnasium matplotlib seaborn pygame
```

### Usage Commands

#### 1. Full Research Suite
To run the entire ablation study (all 3 experiments across all 7 configurations, over multiple
seeds) and generate the full report:
```bash
python main.py                 # default seeds (config.EXP_SEEDS = 3 seeds)
python main.py --seeds 5       # run 5 independent seeds for stronger statistics
```
The ablation configurations are: **Full Model, Ablated DA, Ablated NA, Ablated 5-HT, No Plasticity,
No Modulation** (same plastic+softmax architecture with hormones clamped — the clean modulation
control), and **Static Baseline** (classic ε-greedy DQN). Cross-seed Welch t-tests are written to
`research_results/pvalues.csv`, with per-seed metrics in `experiment_results.csv` and a mean/std
roll-up in `experiment_results_summary.csv`.

*Note: 7 configs × seeds × 3 experiments can be slow (Experiment 3 dominates). Use `--exp N` and/or
`--seeds 1` to scope down during development.*

#### 2. Single Experiment Analysis
To run a full ablation analysis (Full Model, Ablated NA, Ablated 5-HT, and Static Baseline) for just one specific experiment:
```bash
python main.py --exp 1   # Volatile Bandit
python main.py --exp 2   # High-Stakes Foraging
python main.py --exp 3   # CartPole Adaptation
```
*Note: Add `--merge` to combine results into a single chart file, and `--seeds N` to set the number
of independent seeds. Both flags also work on the single-experiment commands above.*

#### 3. Interactive Simulation Engine
To visualize experiments or run specific analyses. Both arguments are **required**:
```bash
python simulation_engine.py --exp <1|2|3> --mode <live|fast|ablation>
```

**Examples:**
```bash
python simulation_engine.py --exp 1 --mode ablation
python simulation_engine.py --exp 1 --mode live
python simulation_engine.py --exp 1 --mode fast
```

- **Live Mode (`--mode live`)**: Normal speed playback with Pygame visuals.
    - *Controls*: `SPACE` to pause, `UP/DOWN` to adjust speed, `R` to restart simulation.
    - *Live Ablation*: Toggle the **DA, NA, and 5HT checkboxes** at the top of the UI to enable/disable neuromodulation on-the-fly.
    - *Static Baseline*: To run the true Static Baseline (Standard DQN), uncheck all three hormones and press **`R`** to restart.
    - *UI Indicators*: A live status bar indicates whether the **DYNAMIC RL** or **STATIC BASELINE** model is currently running.
- **Fast Mode (`--mode fast`)**: Train a single Full Model in the background at maximum speed.
- **Ablation Mode (`--mode ablation`)**: Run the full 4-config study for the selected experiment.

---

## 🧪 Experiments

### Exp 1: Volatile Multi-Armed Bandit (NA Test)
Tests the agent's ability to detect shifts in reward distributions. Noradrenaline (NA) spikes during volatility to reset internal representations.

### Exp 2: High-Stakes Foraging (5-HT Test)
Tests survival and harm aversion. Serotonin (5-HT) spikes during "near-death" or high-risk scenarios to enforce a safer policy.

### Exp 3: CartPole Physics Adaptation (DA/NA Test)
Tests the agent's ability to adapt to sudden physical changes (e.g., 3x gravity). Requires rapid re-learning of balancing dynamics.

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
- `ablation.py`: Framework for running comparative studies.
- `evaluation.py`: Statistics and dashboard generation.
- `research_results/`: Directory where all dashboards and CSVs are saved.

---

## 📈 Visualizations

The system generates high-quality dashboards for each run, including:
- Reward curves and survival rates.
- Hormone concentration timelines.
- Dynamic hyperparameter adjustments.
- Comparative bar charts for ablation studies.

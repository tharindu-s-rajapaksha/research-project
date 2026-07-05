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
Workers run on CPU by default (faster for tiny nets than GPU). Results: `research_results/*.csv`, dashboards, regret curves.

*Note: Add `--merge` to any of the above to combine results into a single chart file.*

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

### Exp 3: CartPole Physics Adaptation (DA Test)
Tests the agent's ability to adapt to a sudden physics change (2× gravity, applied after a pre-training phase). Requires rapid re-learning of balancing dynamics.

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

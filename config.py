"""
config.py — Global hyperparameters and constants for the
Multi-Neuromodulated Modular RL Architecture.

All mathematical symbols reference the specification document.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────────────
# General
# ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CPU or GPU
SEED = 42
# Seeds for the statistical study. Every config is run on EVERY seed, so the
# per-seed metric vectors are paired by seed. ≥10 recommended for the final
# report; trim for quick iteration (heavy: len(SEEDS)×n_configs×3 experiments).
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_results")

# ─────────────────────────────────────────────────────────────────────
# Hormone Engine — Accumulation & Decay  (Section 3A)
# C_{t+1} = B + (C_t - B) * exp(-k) + SpikeValue   (Homeostatic Decay)
# ─────────────────────────────────────────────────────────────────────
HORMONE_BASELINE = 1.0          # Resting concentration
HORMONE_DECAY_DA  = 0.1         # k for Dopamine
HORMONE_DECAY_NA  = 0.08        # k for Noradrenaline  (slower decay)
HORMONE_DECAY_5HT = 0.03        # k for Serotonin      (slowest decay)

DA_SPIKE_SCALE   = 1.0          # Scaling factor for TD-error → DA spike
NA_SPIKE_SCALE   = 2.0          # Scaling factor for volatility → NA spike
HT_SPIKE_SCALE   = 3.0          # Scaling factor for aversive event → 5-HT spike

# Opponent processing  (Section 3B):  DA_eff = DA_raw * (1 - σ(5HT))
# σ is torch.sigmoid, centred at HORMONE_BASELINE

# ─────────────────────────────────────────────────────────────────────
# Volatility / Risk Trackers  (Section 2A)
# ─────────────────────────────────────────────────────────────────────
VOLATILITY_WINDOW = 200          # History buffer for change detection
VOLATILITY_THRESHOLD = 2.0       # Z-score threshold: recent mean vs baseline mean
RISK_PENALTY_THRESHOLD = -50.0  # Reward below this triggers 5-HT spike
DEATH_PENALTY = -500.0          # Canonical "death" penalty value

# ─────────────────────────────────────────────────────────────────────
# Differentiable Plasticity  (Section 4A)
# ─────────────────────────────────────────────────────────────────────
ETA_DECAY = 0.05                # η_decay  — trace decay rate
ETA_TRACE = 0.01                # η_trace  — trace accumulation rate

# ─────────────────────────────────────────────────────────────────────
# Worker DQN  (Section 2B + 4B)
# ─────────────────────────────────────────────────────────────────────
ALPHA_BASE   = 1e-3             # α_base (DA)  — base learning rate
TAU_BASE     = 1.0              # τ_base (NA)  — base softmax exploration temperature
GAMMA_BASE   = 0.99             # γ_base (5HT) — base discount factor (agent AT REST)
GAMMA_MAX    = 0.999            # γ ceiling — horizon when 5-HT is saturated (survival mode)

HIDDEN_DIM   = 128              # Hidden layer width
REPLAY_SIZE  = 500              # Experience-replay buffer capacity (CHANGED FROM 10000 to 500)
BATCH_SIZE   = 64               # Mini-batch size
TARGET_UPDATE_FREQ = 100        # Steps between target-network syncs
EPSILON_MIN  = 0.01             # Floor for ε (static baseline)

# ─────────────────────────────────────────────────────────────────────
# Experiment 1 — Volatile Multi-Armed Bandit  (Section 5)
# ─────────────────────────────────────────────────────────────────────
EXP1_N_ARMS        = 5
EXP1_TOTAL_STEPS   = 4_000
EXP1_SWITCH_STEPS  = [500, 1100, 1800, 3000]
EXP1_REWARD_MU_HI  = 10.0
EXP1_REWARD_MU_LO  = 2.0
EXP1_REWARD_SIGMA  = 1.0

# ─────────────────────────────────────────────────────────────────────
# Experiment 2 — High-Stakes Foraging  (Section 6)
# ─────────────────────────────────────────────────────────────────────
EXP2_TOTAL_STEPS    = 5_000
EXP2_SAFE_REWARD    = 5.0
EXP2_RISKY_REWARD   = 50.0
EXP2_RISKY_DEATH_P  = 0.10      # 10% death probability
EXP2_DEATH_PENALTY  = -500.0

# ─────────────────────────────────────────────────────────────────────
# Experiment 3 — CartPole Physics Adaptation  (Section 7)
# ─────────────────────────────────────────────────────────────────────
EXP3_TRAIN_EPISODES    = 300    # Pre-perturbation training
EXP3_PERTURB_EPISODE   = 300    # Episode at which the physics changes
EXP3_POST_EPISODES     = 300    # Post-perturbation episodes
EXP3_NEW_GRAVITY       = 29.4   # 3× the CartPole default gravity (9.8)
EXP3_FORCE_SCALE       = 0.5    # Actuator force_mag multiplier (halves push
                                # strength) — proxy for a changed-dynamics shock
EXP3_RECOVERY_TARGET   = 400    # Steps sustained to count as "recovered"
EXP3_COMPETENCE_TARGET = 450    # Pre-perturb rolling mean needed to be "competent"
EXP3_COMPETENCE_WINDOW = 20     # Episodes averaged for the competence check

# ─────────────────────────────────────────────────────────────────────
# Ablation Study  (Section 9)
# ─────────────────────────────────────────────────────────────────────
# Each config runs on the SAME plastic LocalRLWorker so that the ONLY thing
# that differs is which neuromodulator is active — a fair, single-variable
# ablation.  "Static Baseline" is that identical architecture with all
# hormones frozen at baseline (α/τ/γ constant at their base values), i.e. the
# static agent the hypothesis claims to beat.  "Vanilla DQN" is a separate
# plain-MLP ε-greedy reference (marked with "vanilla") — NOT a clean ablation,
# kept only as an external sanity anchor.
ABLATION_CONFIGS = {
    "Full Model":      {"DA": True,  "NA": True,  "5HT": True},
    "Ablated DA":      {"DA": False, "NA": True,  "5HT": True},   # isolate DA/plasticity
    "Ablated NA":      {"DA": True,  "NA": False, "5HT": True},   # isolate adaptation
    "Ablated 5-HT":    {"DA": True,  "NA": True,  "5HT": False},  # isolate harm aversion
    "Static Baseline": {"DA": False, "NA": False, "5HT": False},  # same arch, frozen
    "Vanilla DQN":     {"DA": False, "NA": False, "5HT": False, "vanilla": True},
}

# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────
ROLLING_WINDOW = 100            # Rolling average window for reward plots
PLOT_DPI = 300                  # Publication-quality DPI
FIG_SIZE = (16, 12)             # Default multi-panel figure size

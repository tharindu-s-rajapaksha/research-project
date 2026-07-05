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
# Device. Set NEUROMOD_FORCE_CPU=1 to force CPU — used by the parallel study
# runner, since these tiny nets run faster per-process on CPU than on a
# contended GPU, and N CPU workers give near-linear throughput.
if os.environ.get("NEUROMOD_FORCE_CPU", "0") == "1":
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
# Seeds for the statistical study. Every config is run on EVERY seed, so the
# per-seed metric vectors are paired by seed. 10 seeds is the reported setting
# (heavy: len(SEEDS)×n_configs×3 experiments). Each runner reseeds torch, numpy
# AND the stdlib `random` from this value, so every run is fully reproducible.
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
VOLATILITY_THRESHOLD = 3.5       # Z-score threshold: recent mean vs baseline mean.
                                 # Raised 2.0→3.5: at 2.0 the NA detector fired on
                                 # ordinary ε-greedy reward noise (~450×/run) rather
                                 # than only the genuine distribution switches (~4×),
                                 # so NA lost its selectivity and its feedback loop
                                 # (NA↑→ε↑→more noise→NA↑) inflated exploration.
RISK_PENALTY_THRESHOLD = -50.0  # Reward below this triggers 5-HT spike
DEATH_PENALTY = -500.0          # Canonical "death" penalty value

# ─────────────────────────────────────────────────────────────────────
# Differentiable Plasticity  (Section 4A)
# ─────────────────────────────────────────────────────────────────────
ETA_DECAY = 0.05                # η_decay  — trace decay rate
ETA_TRACE = 0.01                # η_trace  — trace accumulation rate
PLASTIC_ALPHA_INIT = 0.002      # initial per-synapse plastic coefficient α
                                # (small: strong fast-weights destabilised
                                #  stable-control learning)

# ─────────────────────────────────────────────────────────────────────
# Worker DQN  (Section 2B + 4B)
# ─────────────────────────────────────────────────────────────────────
ALPHA_BASE   = 1e-3             # α_base (DA)  — base learning rate
ALPHA_MAX_SCALE = 2.0           # α ceiling = α_base × this (DA surprise boost, capped)
GAMMA_BASE   = 0.99             # γ_base (5HT) — base discount factor (agent AT REST)
GAMMA_MAX    = 0.999            # γ ceiling — horizon when 5-HT is saturated (survival mode)

# Exploration is ε-greedy (scale-invariant), with NA modulating the rate ε.
# Boltzmann/softmax temperature proved uncompetitive on near-equal-Q tasks
# (e.g. CartPole), so NA is routed through ε instead of τ.
EPSILON_BASE = 0.01             # ε at rest. Lowered 0.1→0.01 over tuning: the base
                                # rate is the STEADY-STATE exploration floor, and NA
                                # supplies switch-time exploration by opening ε up to
                                # EPSILON_MAX, so the base can be very low. Measured on
                                # the bandit, lowering it raised BOTH the cumulative
                                # optimal-pull rate (0.1→0.03→0.01 gave 36→76→84%) and
                                # the locked steady-state (→94.5%), and cut re-lock
                                # time; on foraging it also cut forced-random deaths
                                # (43→39). It leans harder on NA — which is the point.
EPSILON_MAX  = 0.5              # ε when NA is saturated (a strong nudge, not
                                # near-random — 0.9 wrecked stable control)

# Serotonin harm-aversion pathway (two mechanisms, both gated by 5-HT):
#  1. Punishment-sensitive learning: up-weight the loss from negative-reward
#     transitions so harmful actions lose value faster.
#  2. Behavioural inhibition: at action selection, subtract a penalty from
#     actions with a learned "harm history", so 5-HT actively WITHHOLDS
#     risky actions (Cools 2011; Crockett 2009) — this breaks the
#     exploration trap where the agent stays hooked on a high-EV lethal
#     action and never samples the safe one.
HT_PUNISHMENT_GAIN     = 4.0    # max loss up-weight on losses when 5-HT saturates
RISK_INHIBITION_WEIGHT = 5.0    # scales the 5-HT behavioural-inhibition penalty.
                                # Raised 1.0→5.0: at 1.0 the penalty subtracted from
                                # a harmful action's Q-value was too small to overcome
                                # the risky arm's frequent +50, so the greedy policy
                                # still chose it ~9% of the time. 5.0 decisively
                                # withholds actions with a death history (safe-rate
                                # 83→93%, deaths 92→46). Only active when 5-HT is
                                # elevated, so Exp 1/3 are unaffected.
HARM_EMA_DECAY         = 0.90   # EMA decay for per-action harm estimate. Lowered
                                # 0.99→0.90: at 0.99 the per-action harm estimate
                                # took ~100 deaths to build, but only ~90 deaths
                                # occur, so behavioural inhibition never became
                                # strong enough. 0.90 builds a usable harm signal
                                # within a handful of deaths.

HIDDEN_DIM   = 128              # Hidden layer width
# Replay-buffer capacity is set PER EXPERIMENT (see EXP*_REPLAY_SIZE): a volatile
# bandit wants a SMALL buffer so stale pre-switch rewards are forgotten quickly,
# whereas foraging/control want a LARGE buffer so rare (10%) death transitions are
# retained long enough for the value function to learn they are catastrophic.
REPLAY_SIZE  = 500              # SAFE fallback default when a caller does not pass a
                                # per-experiment size (small, so it cannot silently
                                # break the bandit as a large default once did)
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
EXP1_REPLAY_SIZE   = 500        # small → forget stale pre-switch reward stats fast

# ─────────────────────────────────────────────────────────────────────
# Experiment 2 — High-Stakes Foraging  (Section 6)
# ─────────────────────────────────────────────────────────────────────
EXP2_TOTAL_STEPS    = 5_000
EXP2_SAFE_REWARD    = 5.0
EXP2_RISKY_REWARD   = 50.0
EXP2_RISKY_DEATH_P  = 0.10      # 10% death probability
EXP2_DEATH_PENALTY  = -500.0
EXP2_REPLAY_SIZE    = 5000      # large → retain rare death transitions for learning

# ─────────────────────────────────────────────────────────────────────
# Experiment 3 — CartPole Physics Adaptation  (Section 7)
# ─────────────────────────────────────────────────────────────────────
EXP3_TRAIN_EPISODES    = 400    # Pre-perturbation training (300→400: give the
                                # plastic DA-on configs more room to reach competence)
EXP3_PERTURB_EPISODE   = 400    # Episode at which the physics changes
EXP3_POST_EPISODES     = 300    # Post-perturbation episodes
EXP3_NEW_GRAVITY       = 19.6   # 2× the CartPole default gravity (9.8). Was 3×
                                # (29.4): combined with a halved force that made the
                                # post-shock task near-unsolvable, so "recovery" was
                                # unmeasurable. 2× gravity is a clear but recoverable
                                # dynamics shock.
EXP3_FORCE_SCALE       = 1.0    # Actuator force_mag multiplier. Restored 0.5→1.0:
                                # halving control authority ON TOP of 3× gravity was
                                # double-jeopardy; the gravity change alone is the
                                # cleaner single-variable dynamics perturbation.
EXP3_RECOVERY_TARGET   = 300    # Steps sustained to count as "recovered" (60% of max)
EXP3_COMPETENCE_TARGET = 350    # Pre-perturb rolling mean needed to be "competent" (70%)
EXP3_COMPETENCE_WINDOW = 20     # Episodes averaged for the competence check
EXP3_REPLAY_SIZE       = 10000  # standard DQN buffer for continuous control
MIN_RECOVERY_SEEDS     = 5      # Min competent seeds required to report a
                                # Recovery_Time point estimate / run its paired
                                # test; below this the metric is left undefined
                                # (too few competent seeds to be meaningful)

# ─────────────────────────────────────────────────────────────────────
# Ablation Study  (Section 9)
# ─────────────────────────────────────────────────────────────────────
# Each config runs on the SAME plastic LocalRLWorker so that the ONLY thing
# that differs is which neuromodulator is active — a fair, single-variable
# ablation.  "Static Baseline" is that identical architecture with all
# hormones frozen at baseline (α/ε/γ constant at their base values), i.e. the
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

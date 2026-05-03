"""
meta_agent.py — Hormonal Meta-Agent (The Conductor)
=====================================================
Monitors global performance and calculates concentration levels for
digital hormones (DA, NA, 5-HT). Outputs a "Hormonal Vector" that
modulates the Local RL Worker's hyperparameters in real time.

Responsibilities (Section 2A):
    • Volatility Tracker  → drives Noradrenaline
    • Risk Tracker         → drives Serotonin
    • TD-Error relay       → drives Dopamine
"""

import numpy as np
from collections import deque

import config as cfg
from neuromodulators import HormoneEngine


class HormonalMetaAgent:
    """The Conductor — computes a hormonal modulation vector each step.

    The Meta-Agent wraps the HormoneEngine and adds higher-level tracking:
        • a rolling window of prediction errors to detect volatility shifts,
        • a death/penalty counter to track risk exposure.

    It translates the raw hormone levels into dynamic hyperparameters
    for the Local Worker (Section 4B):
        α_t  = α_base × DA_concentration         (Learning Rate)
        τ_t  = τ_base × (1 / NA_concentration)   (Softmax Temperature)
        γ_t  = γ_base × σ(5HT)                   (Discount Factor)
    """

    def __init__(self, enable_da: bool = True, enable_na: bool = True,
                 enable_5ht: bool = True):
        self.engine = HormoneEngine(enable_da, enable_na, enable_5ht)

        # Performance trackers
        self._recent_rewards = deque(maxlen=cfg.VOLATILITY_WINDOW)
        self._death_count = 0
        self._total_steps = 0

        # Logging buffers for hyperparameter dynamics
        self.history_alpha = []
        self.history_tau = []
        self.history_gamma = []

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def step(self, td_error: float, reward: float,
             done: bool) -> dict:
        """Process one environment step.

        Args:
            td_error: Worker's temporal-difference error.
            reward:   Scalar reward from environment.
            done:     Episode termination flag.

        Returns:
            dict with modulated hyperparameters:
                'alpha', 'tau', 'gamma', and raw hormone levels.
        """
        self._total_steps += 1
        self._recent_rewards.append(reward)

        if done and reward <= cfg.RISK_PENALTY_THRESHOLD:
            self._death_count += 1

        # Update hormone concentrations
        hormones = self.engine.step(td_error, reward, done)

        # Compute modulated hyperparameters  (Section 4B)
        alpha = self._modulate_lr(hormones["DA_eff"])
        tau   = self._modulate_temperature(hormones["NA"])
        gamma = self._modulate_discount(hormones["5HT"])

        # Log
        self.history_alpha.append(alpha)
        self.history_tau.append(tau)
        self.history_gamma.append(gamma)

        return {
            "alpha" : alpha,
            "tau"   : tau,
            "gamma" : gamma,
            **hormones,
        }

    def reset_episode(self):
        """Called at the start of each new episode."""
        # Do NOT reset hormone engine — hormones persist across episodes
        # to simulate sustained mood effects (e.g., lingering serotonin
        # after a death). Only reset the per-episode trackers.
        pass

    def hard_reset(self):
        """Full reset including hormone levels (for new experiment run)."""
        self.engine.reset()
        self._recent_rewards.clear()
        self._death_count = 0
        self._total_steps = 0
        self.history_alpha.clear()
        self.history_tau.clear()
        self.history_gamma.clear()
        self.engine.history_da.clear()
        self.engine.history_na.clear()
        self.engine.history_ht.clear()
        self.engine.history_da_eff.clear()

    @property
    def death_count(self) -> int:
        return self._death_count

    # ──────────────────────────────────────────────────────────────────
    # Hyperparameter modulation  (Section 4B)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _modulate_lr(da_eff: float) -> float:
        """α_t = α_base × clip(DA_eff, 0.1, 5.0).

        DA_eff close to 0 → very low learning (no surprise);
        DA_eff high → fast learning (reward surprise).
        """
        return cfg.ALPHA_BASE * float(np.clip(da_eff, 0.1, 5.0))

    @staticmethod
    def _modulate_temperature(na: float) -> float:
        """τ_t = τ_base × clip(NA, 0.1, 10.0).

        High NA (uncertainty) → high temperature → more random exploration.
        (Higher temperature in softmax → more uniform distribution.)
        """
        return cfg.TAU_BASE * float(np.clip(na, 0.1, 10.0))

    @staticmethod
    def _modulate_discount(ht: float) -> float:
        """γ_t = γ_base × σ(5HT - baseline).

        High 5-HT → sigmoid > 0.5 → agent values long-term survival;
        Low  5-HT → sigmoid < 0.5 → agent is more short-sighted.
        """
        sigmoid = 1.0 / (1.0 + np.exp(-(ht - cfg.HORMONE_BASELINE)))
        return cfg.GAMMA_BASE * float(sigmoid)

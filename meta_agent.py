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

import config as cfg
from neuromodulators import HormoneEngine


class HormonalMetaAgent:
    """The Conductor — computes a hormonal modulation vector each step.

    The Meta-Agent wraps the HormoneEngine and adds higher-level tracking:
        • a rolling window of prediction errors to detect volatility shifts,
        • a death/penalty counter to track risk exposure.

    It translates the raw hormone levels into dynamic hyperparameters
    for the Local Worker (Section 4B):
        α_t   = α_base × (1 + |DA_eff − rest|)              (Learning Rate)
        ε_t   = ε_base + (ε_max − ε_base) × excess(NA)      (Exploration Rate)
        γ_t   = γ_base + (γ_max − γ_base) × excess(5HT)     (Discount Factor)
        g_t   = 1 + (punish_gain − 1) × excess(5HT)         (Loss Aversion)

    NOTE on exploration: noradrenaline is routed through the ε-greedy rate
    ε (higher NA → more exploration), NOT a softmax temperature.  Boltzmann
    temperature is scale-sensitive and was uncompetitive on near-equal-Q
    tasks (CartPole); ε is scale-invariant.  Every mapping reduces to its
    base value at rest (all hormones = HORMONE_BASELINE), so the modulated
    agent with no spikes is identical to the static baseline.
    """

    def __init__(self, enable_da: bool = True, enable_na: bool = True,
                 enable_5ht: bool = True,
                 volatility_threshold: float = cfg.VOLATILITY_THRESHOLD):
        self.engine = HormoneEngine(enable_da, enable_na, enable_5ht,
                                    volatility_threshold=volatility_threshold)

        # Performance trackers
        self._death_count = 0
        self._total_steps = 0

        # Logging buffers for hyperparameter dynamics
        self.history_alpha = []
        self.history_epsilon = []
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
                'alpha', 'epsilon', 'gamma', 'punish_gain', and hormone levels.
        """
        self._total_steps += 1

        if done and reward <= cfg.RISK_PENALTY_THRESHOLD:
            self._death_count += 1

        # Update hormone concentrations
        hormones = self.engine.step(td_error, reward, done)

        # Compute modulated hyperparameters  (Section 4B)
        alpha   = self._modulate_lr(hormones["DA_eff"])
        epsilon = self._modulate_epsilon(hormones["NA"])
        gamma   = self._modulate_discount(hormones["5HT"])
        punish  = self._punishment_gain(hormones["5HT"])

        # Log
        self.history_alpha.append(alpha)
        self.history_epsilon.append(epsilon)
        self.history_gamma.append(gamma)

        return {
            "alpha"       : alpha,
            "epsilon"     : epsilon,
            "gamma"       : gamma,
            "punish_gain" : punish,
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
        self._death_count = 0
        self._total_steps = 0
        self.history_alpha.clear()
        self.history_epsilon.clear()
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
    def _excess(conc: float) -> float:
        """How far a hormone sits ABOVE its resting baseline, in [0, 1].

        excess = clip(2·(σ(conc − baseline) − 0.5), 0, 1) → 0 at/below
        baseline, → 1 as the concentration saturates.  Shared by every
        modulation so each one reduces to its base value at rest.
        """
        sigmoid = 1.0 / (1.0 + np.exp(-(conc - cfg.HORMONE_BASELINE)))
        return float(np.clip(2.0 * (sigmoid - 0.5), 0.0, 1.0))

    @staticmethod
    def _modulate_lr(da_eff: float) -> float:
        """α_t = α_base × (1 + |DA_eff − DA_eff_rest|), capped at α_max.

        Learning rate scales with the MAGNITUDE of surprise, not direction.
        At rest α = α_base.  The ceiling is tightened to ALPHA_MAX_SCALE
        (was 5×) because large DA-driven α spikes destabilised value
        learning on CartPole, where the modulated agent underperformed the
        frozen baseline.
        """
        # DA_eff at rest: DA=1.0, 5HT=1.0 → DA_eff = 1.0 × (1 - σ(0)) = 0.5
        da_eff_rest = cfg.HORMONE_BASELINE * 0.5
        surprise = 1.0 + abs(da_eff - da_eff_rest)
        return cfg.ALPHA_BASE * float(np.clip(surprise, 1.0,
                                              cfg.ALPHA_MAX_SCALE))

    @classmethod
    def _modulate_epsilon(cls, na: float) -> float:
        """ε_t = ε_base + (ε_max − ε_base) × excess(NA).

        Noradrenaline routes through the ε-greedy exploration RATE (not a
        softmax temperature).  At rest (NA = baseline) ε = ε_base, matching
        the vanilla-DQN baseline; a volatility-driven NA spike raises ε
        toward ε_max → more exploration exactly when the world changes.
        ε is scale-invariant, so this works even where Q-gaps are tiny
        (CartPole), unlike Boltzmann temperature.
        """
        return cfg.EPSILON_BASE + (cfg.EPSILON_MAX - cfg.EPSILON_BASE) * cls._excess(na)

    @classmethod
    def _modulate_discount(cls, ht: float) -> float:
        """γ_t = γ_base + (γ_max − γ_base) × excess(5HT).

        At rest (5-HT = baseline) γ = γ_base (== static baseline); a
        serotonin spike can only *lengthen* the horizon toward γ_max, never
        shorten it.  Fixes the earlier `γ_base × σ(5HT−baseline)` form,
        which collapsed γ to ≈0.5·γ_base at rest and crippled long-horizon
        tasks (e.g. CartPole).
        """
        return cfg.GAMMA_BASE + (cfg.GAMMA_MAX - cfg.GAMMA_BASE) * cls._excess(ht)

    @classmethod
    def _punishment_gain(cls, ht: float) -> float:
        """Loss-aversion gain g_t = 1 + (HT_PUNISHMENT_GAIN − 1) × excess(5HT).

        Serotonin's harm-aversion pathway.  Elevated 5-HT amplifies the
        magnitude of NEGATIVE rewards in the value target (losses loom
        larger), which lowers the learned value of harmful/high-variance
        actions and biases the policy toward safety — the serotonin-as
        -punishment-sensitivity account (Daw 2002; Cools 2011).  At rest
        g = 1 (no distortion), so the static baseline is unaffected.
        """
        return 1.0 + (cfg.HT_PUNISHMENT_GAIN - 1.0) * cls._excess(ht)

"""
environments.py — Custom Environments for Experiments 1 & 2
=============================================================
Implements:
    1. VolatileBandit     — 5-armed bandit with mid-run reward switch (Exp 1)
    2. HighStakesForaging — Safe vs. Risky foraging task (Exp 2)

Both follow a minimal Gymnasium-like interface:
    reset() → state
    step(action) → (state, reward, done, truncated, info)
"""

import numpy as np
import config as cfg


class VolatileBandit:
    """5-armed Gaussian bandit with a sudden reward distribution switch.

    Stationary Phase (steps 0..499):
        Arm 0 → N(μ=10, σ=1), Arms 1-4 → N(μ=2, σ=1)
    Post-switch (steps 500+):
        Arm 0 → N(μ=0, σ=1), Arm 4 → N(μ=10, σ=1), others unchanged

    Observation space: one-hot encoding of the last chosen arm (dim=5).
    Action space: {0, 1, 2, 3, 4}.
    """

    def __init__(self, n_arms: int = cfg.EXP1_N_ARMS,
                 switch_step: int = cfg.EXP1_SWITCH_STEP,
                 seed: int = cfg.SEED):
        self.n_arms = n_arms
        self.switch_step = switch_step
        self.rng = np.random.RandomState(seed)
        self._step = 0

        # Reward means — pre-switch
        self.means_pre = np.full(n_arms, cfg.EXP1_REWARD_MU_LO)
        self.means_pre[0] = cfg.EXP1_REWARD_MU_HI

        # Reward means — post-switch
        self.means_post = np.full(n_arms, cfg.EXP1_REWARD_MU_LO)
        self.means_post[0] = 0.0
        self.means_post[4] = cfg.EXP1_REWARD_MU_HI

        self.sigma = cfg.EXP1_REWARD_SIGMA
        self._state = np.zeros(n_arms, dtype=np.float32)

    @property
    def observation_dim(self) -> int:
        return self.n_arms

    @property
    def action_dim(self) -> int:
        return self.n_arms

    def reset(self) -> np.ndarray:
        self._step = 0
        self._state = np.zeros(self.n_arms, dtype=np.float32)
        return self._state.copy()

    def step(self, action: int):
        """Execute one bandit pull.

        Returns:
            (state, reward, done, truncated, info)
        """
        means = (self.means_pre if self._step < self.switch_step
                 else self.means_post)
        reward = float(self.rng.normal(means[action], self.sigma))

        # State: one-hot of last action
        self._state = np.zeros(self.n_arms, dtype=np.float32)
        self._state[action] = 1.0

        self._step += 1
        done = self._step >= cfg.EXP1_TOTAL_STEPS
        info = {"step": self._step, "switched": self._step >= self.switch_step,
                "optimal_arm": 0 if self._step <= self.switch_step else 4}

        return self._state.copy(), reward, done, False, info


class HighStakesForaging:
    """Two-option foraging task with asymmetric risk (Experiment 2).

    Actions:
        0 → Safe Source:  deterministic +5
        1 → Risky Source: 90% chance of +50, 10% chance of Death (−500 + reset)

    Observation: [cumulative_score_normalized, steps_since_death_normalized]
    """

    def __init__(self, total_steps: int = cfg.EXP2_TOTAL_STEPS,
                 seed: int = cfg.SEED):
        self.total_steps = total_steps
        self.rng = np.random.RandomState(seed)
        self._step = 0
        self._cumulative = 0.0
        self._steps_since_death = 0
        self._death_count = 0

    @property
    def observation_dim(self) -> int:
        return 2

    @property
    def action_dim(self) -> int:
        return 2

    def reset(self) -> np.ndarray:
        self._step = 0
        self._cumulative = 0.0
        self._steps_since_death = 0
        self._death_count = 0
        return self._get_obs()

    def step(self, action: int):
        self._step += 1
        self._steps_since_death += 1

        if action == 0:
            # Safe source
            reward = cfg.EXP2_SAFE_REWARD
            self._cumulative += reward
            done = self._step >= self.total_steps
            info = {"death": False}
        else:
            # Risky source
            if self.rng.random() < cfg.EXP2_RISKY_DEATH_P:
                # DEATH
                reward = cfg.EXP2_DEATH_PENALTY
                self._cumulative = 0.0  # Reset total score
                self._steps_since_death = 0
                self._death_count += 1
                done = self._step >= self.total_steps
                info = {"death": True}
            else:
                reward = cfg.EXP2_RISKY_REWARD
                self._cumulative += reward
                done = self._step >= self.total_steps
                info = {"death": False}

        info["cumulative"] = self._cumulative
        info["death_count"] = self._death_count
        info["step"] = self._step

        return self._get_obs(), reward, done, False, info

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self._cumulative / 1000.0,          # Normalized score
            self._steps_since_death / 500.0,    # Normalized survival
        ], dtype=np.float32)

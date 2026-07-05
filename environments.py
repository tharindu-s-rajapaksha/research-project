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
                 switch_steps: list = cfg.EXP1_SWITCH_STEPS,
                 seed: int = cfg.SEED):
        self.n_arms = n_arms
        self.switch_steps = sorted(switch_steps)
        self.rng = np.random.RandomState(seed)
        self._step = 0
        self.sigma = cfg.EXP1_REWARD_SIGMA
        self._state = np.zeros(n_arms, dtype=np.float32)

        # Define sequence of optimal arms to cycle through
        self.optimal_sequence = [0, 4, 1, 3, 2]
        # Extend sequence to cover all phases (number of switches + 1)
        while len(self.optimal_sequence) <= len(self.switch_steps):
            self.optimal_sequence.extend([0, 4, 1, 3, 2])
            
        self.current_optimal_arm = self.optimal_sequence[0]
        self.current_means = np.full(n_arms, cfg.EXP1_REWARD_MU_LO)
        self.current_means[self.current_optimal_arm] = cfg.EXP1_REWARD_MU_HI

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
        # Determine current phase based on step count
        phase = sum(1 for s in self.switch_steps if self._step >= s)
        self.current_optimal_arm = self.optimal_sequence[phase]
        
        self.current_means = np.full(self.n_arms, cfg.EXP1_REWARD_MU_LO)
        self.current_means[self.current_optimal_arm] = cfg.EXP1_REWARD_MU_HI
        
        reward = float(self.rng.normal(self.current_means[action], self.sigma))

        # State: one-hot of last action
        self._state = np.zeros(self.n_arms, dtype=np.float32)
        self._state[action] = 1.0

        self._step += 1
        done = self._step >= cfg.EXP1_TOTAL_STEPS
        
        info = {
            "step": self._step, 
            "switched": self._step in self.switch_steps,
            "optimal_arm": self.current_optimal_arm,
            "phase": phase
        }

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


class VolatileRiskyForaging:
    """Capstone task (Experiment 3) — fuses volatility (Exp 1) with lethal
    risk (Exp 2) so that ALL THREE neuromodulators are needed at once.

    Arms 0..n_safe-1 are SAFE: exactly one is "good" (μ_hi), the rest are
    meagre (μ_lo).  The good safe arm MOVES at each switch (volatility).
    Arm n_safe is the RISKY arm: it pays μ_risky on most pulls but has a
    p_death chance of Death (death_penalty + score reset).  Because a rare
    −500 outweighs the frequent +50, the risky arm's true EV is NEGATIVE — a
    trap that a greedy agent gets hooked on.

    Necessary role of each hormone (removing any one → worse cumulative reward):
        • NA  — detect the switch (reward drops) and re-explore.
        • DA  — rapidly re-lock the NEW good safe arm (plastic fast-weights).
        • 5-HT — resist the tempting lethal arm (behavioural inhibition).

    Observation: one-hot of the last action (dim = n_safe + 1).
    """

    def __init__(self, seed: int = cfg.SEED):
        self.n_safe = cfg.VRF_N_SAFE_ARMS
        self.risky_arm = self.n_safe               # lethal arm = last index
        self.n_actions = self.n_safe + 1
        self.switch_steps = sorted(cfg.VRF_SWITCH_STEPS)
        self.rng = np.random.RandomState(seed)
        self.sigma = cfg.VRF_SIGMA

        # Sequence of "good safe arm" to cycle through (a fixed permutation of
        # the safe arms), extended to cover every phase.
        base = [0, 2, 4, 1, 3][:self.n_safe] or list(range(self.n_safe))
        self.optimal_sequence = list(base)
        while len(self.optimal_sequence) <= len(self.switch_steps):
            self.optimal_sequence.extend(base)

        self._step = 0
        self._cumulative = 0.0
        self._steps_since_death = 0
        self._death_count = 0
        self._state = np.zeros(self.n_actions, dtype=np.float32)

    @property
    def observation_dim(self) -> int:
        return self.n_actions

    @property
    def action_dim(self) -> int:
        return self.n_actions

    def reset(self) -> np.ndarray:
        self._step = 0
        self._cumulative = 0.0
        self._steps_since_death = 0
        self._death_count = 0
        self._state = np.zeros(self.n_actions, dtype=np.float32)
        return self._state.copy()

    def step(self, action: int):
        """Execute one pull. Returns (state, reward, done, truncated, info)."""
        # Current phase → which safe arm is good right now.
        phase = sum(1 for s in self.switch_steps if self._step >= s)
        good_arm = self.optimal_sequence[phase]

        death = False
        if action == self.risky_arm:
            if self.rng.random() < cfg.VRF_RISKY_DEATH_P:
                reward = cfg.VRF_DEATH_PENALTY
                self._cumulative = 0.0          # reset accumulated score
                self._death_count += 1
                death = True
            else:
                reward = float(self.rng.normal(cfg.VRF_RISKY_REWARD, self.sigma))
                self._cumulative += reward
        else:
            mu = cfg.VRF_MU_HI if action == good_arm else cfg.VRF_MU_LO
            reward = float(self.rng.normal(mu, self.sigma))
            self._cumulative += reward

        # State: one-hot of last action.
        self._state = np.zeros(self.n_actions, dtype=np.float32)
        self._state[action] = 1.0

        self._step += 1
        self._steps_since_death = 0 if death else self._steps_since_death + 1
        done = self._step >= cfg.VRF_TOTAL_STEPS

        info = {
            "step": self._step,
            "switched": self._step in self.switch_steps,
            "optimal_arm": good_arm,        # the good SAFE arm (never the risky one)
            "risky_arm": self.risky_arm,
            "phase": phase,
            "death": death,
            "cumulative": self._cumulative,
            "death_count": self._death_count,
        }
        return self._state.copy(), reward, done, False, info

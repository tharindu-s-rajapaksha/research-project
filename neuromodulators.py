"""
neuromodulators.py — Digital Hormone Engine
============================================
Implements the "Chemical Engine" that translates environment signals into
hormonal concentrations for Dopamine (DA), Noradrenaline (NA), and
Serotonin (5-HT).

Key equations (Section 3):
    C_{t+1} = C_t · exp(-k) + SpikeValue          (Accumulation & Decay)
    DA_eff  = DA_raw · (1 − σ(5HT))               (Opponent Processing)
"""

import math
import torch
import numpy as np
from collections import deque

import config as cfg


class HormoneEngine:
    """Manages concentrations and dynamics of three digital neuromodulators.

    Each hormone:
        • spikes in response to a specific environmental signal,
        • decays exponentially toward HORMONE_BASELINE,
        • interacts with other hormones via opponent processing.
    """

    def __init__(self, enable_da: bool = True, enable_na: bool = True,
                 enable_5ht: bool = True):
        """
        Args:
            enable_da:  If False, DA is clamped to baseline (ablation).
            enable_na:  If False, NA is clamped to baseline (ablation).
            enable_5ht: If False, 5-HT is clamped to baseline (ablation).
        """
        self.enable_da  = enable_da
        self.enable_na  = enable_na
        self.enable_5ht = enable_5ht

        # Current concentrations
        self.da  = cfg.HORMONE_BASELINE
        self.na  = cfg.HORMONE_BASELINE
        self.ht  = cfg.HORMONE_BASELINE   # 5-HT

        # Volatility tracker — moving window of prediction errors
        self._error_history = deque(maxlen=cfg.VOLATILITY_WINDOW)
        self._prev_volatility = 0.0

        # Logging buffers (for visualization)
        self.history_da  = []
        self.history_na  = []
        self.history_ht  = []
        self.history_da_eff = []

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def step(self, td_error: float, reward: float, done: bool) -> dict:
        """Advance hormone dynamics by one timestep.

        Args:
            td_error: Temporal Difference error δ_TD.
            reward:   Scalar reward from the environment.
            done:     Whether the episode terminated (death / reset).

        Returns:
            dict with keys 'DA', 'NA', '5HT', 'DA_eff' — current levels.
        """
        # 1. Compute raw spikes ------------------------------------------
        da_spike = self._compute_da_spike(td_error)
        na_spike = self._compute_na_spike(td_error)
        ht_spike = self._compute_ht_spike(reward, done)

        # 2. Accumulation + Decay ----------------------------------------
        self.da = self._decay(self.da, cfg.HORMONE_DECAY_DA, da_spike,
                              self.enable_da)
        self.na = self._decay(self.na, cfg.HORMONE_DECAY_NA, na_spike,
                              self.enable_na)
        self.ht = self._decay(self.ht, cfg.HORMONE_DECAY_5HT, ht_spike,
                              self.enable_5ht)

        # 3. Opponent processing: DA_eff = DA_raw * (1 − σ(5HT)) --------
        da_eff = self._opponent_process(self.da, self.ht)

        # 4. Record history for plotting ---------------------------------
        self.history_da.append(self.da)
        self.history_na.append(self.na)
        self.history_ht.append(self.ht)
        self.history_da_eff.append(da_eff)

        return {"DA": self.da, "NA": self.na, "5HT": self.ht,
                "DA_eff": da_eff}

    def reset(self):
        """Reset concentrations to baseline (start of new episode)."""
        self.da = cfg.HORMONE_BASELINE
        self.na = cfg.HORMONE_BASELINE
        self.ht = cfg.HORMONE_BASELINE
        self._error_history.clear()
        self._prev_volatility = 0.0

    def get_vector(self) -> np.ndarray:
        """Return current hormonal vector [DA_eff, NA, 5HT]."""
        da_eff = self._opponent_process(self.da, self.ht)
        return np.array([da_eff, self.na, self.ht], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _decay(current: float, k: float, spike: float,
               enabled: bool) -> float:
        """C_{t+1} = C_t · exp(-k) + spike  (clamped to baseline if disabled)."""
        if not enabled:
            return cfg.HORMONE_BASELINE
        new_val = current * math.exp(-k) + spike
        return max(new_val, 0.0)  # Concentrations are non-negative

    def _compute_da_spike(self, td_error: float) -> float:
        """Dopamine spike proportional to TD error (Section 3A).

        Positive δ → DA spike above baseline;
        Negative δ → DA dip below baseline.
        """
        return td_error * cfg.DA_SPIKE_SCALE

    def _compute_na_spike(self, td_error: float) -> float:
        """Noradrenaline spike driven by 'Unexpected Uncertainty' (Section 3A).

        Tracks the moving variance of prediction errors. A sudden shift
        causes a sustained NA spike.
        """
        self._error_history.append(abs(td_error))
        if len(self._error_history) < 2:
            return 0.0

        observed_vol = float(np.std(list(self._error_history)))
        volatility_surprise = abs(observed_vol - self._prev_volatility)
        self._prev_volatility = observed_vol

        if volatility_surprise > cfg.VOLATILITY_THRESHOLD:
            return volatility_surprise * cfg.NA_SPIKE_SCALE
        return 0.0

    @staticmethod
    def _compute_ht_spike(reward: float, done: bool) -> float:
        """Serotonin spike from aversive events (Section 3A).

        Triggered by 'Death' (done=True with heavy penalty) or rewards
        below RISK_PENALTY_THRESHOLD.
        """
        if done and reward <= cfg.RISK_PENALTY_THRESHOLD:
            return abs(reward) / abs(cfg.DEATH_PENALTY) * cfg.HT_SPIKE_SCALE
        if reward < cfg.RISK_PENALTY_THRESHOLD:
            return abs(reward) / abs(cfg.DEATH_PENALTY) * cfg.HT_SPIKE_SCALE
        return 0.0

    @staticmethod
    def _opponent_process(da_raw: float, ht: float) -> float:
        """Opponent processing: DA_eff = DA_raw × (1 − σ(5HT)).

        σ is the logistic sigmoid centred so that at baseline
        concentration σ(baseline) ≈ 0.5 → DA_eff ≈ 0.5 * DA_raw when
        hormones are at rest. Shift is applied so baseline → 0 in sigmoid.
        """
        # Centre sigmoid at baseline:  σ(ht - baseline)
        sigmoid_ht = 1.0 / (1.0 + math.exp(-(ht - cfg.HORMONE_BASELINE)))
        return da_raw * (1.0 - sigmoid_ht)

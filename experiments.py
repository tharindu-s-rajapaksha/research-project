"""
experiments.py — Experiment Runners for Experiments 1, 2, and 3
=================================================================
Each experiment function returns a results dict containing all logged
data needed for evaluation and visualization.

Experiment 1: Volatile Multi-Armed Bandit  (NA Test)
Experiment 2: High-Stakes Foraging         (5-HT Test)
Experiment 3: Volatile Risky Foraging      (CAPSTONE — DA + NA + 5-HT together)
Legacy:       CartPole Physics Adaptation  (secondary / negative result)
"""

import random
from collections import deque

import numpy as np
import gymnasium as gym
import torch

import config as cfg
from meta_agent import HormonalMetaAgent
from worker import LocalRLWorker, StaticBaselineWorker
from environments import (VolatileBandit, HighStakesForaging,
                          VolatileRiskyForaging, ContextualRiskyForaging)


def _seed_all(seed: int):
    """Seed every RNG the run touches.

    The worker's ε-greedy selection and replay sampling use the *stdlib*
    ``random`` module, so seeding only numpy/torch would leave exploration
    non-reproducible — and, worse, break the paired significance tests, whose
    validity requires that Full and each ablation see the SAME stochastic
    stream on a given seed (so the only difference is the neuromodulator).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _make_agent(state_dim: int, action_dim: int, ablation_cfg: dict,
                replay_size: int = cfg.REPLAY_SIZE,
                volatility_threshold: float = cfg.VOLATILITY_THRESHOLD,
                plastic_alpha: float = None):
    """Factory: build Meta-Agent + Worker pair based on ablation config.

    All configurations use the SAME plastic ``LocalRLWorker`` so the only
    variable across configs is which hormone is enabled — the "Static
    Baseline" is simply this worker with every hormone clamped to baseline
    (α/ε/γ held at their base values).  The single exception is the
    ``"vanilla": True`` reference config, which swaps in the plain-MLP
    ε-greedy ``StaticBaselineWorker`` as an external sanity anchor.

    ``replay_size`` is passed per experiment: small for the volatile bandit
    (forget stale reward stats fast), large for foraging/control (retain the
    rare death transitions long enough to learn from them).
    """
    # `.get(..., False)` so fair-baseline configs (baselines.py) may omit the
    # DA/NA/5HT keys — a vanilla worker ignores modulation anyway, so the meta's
    # enable flags are irrelevant for it. Existing ablation configs pass explicit
    # keys, so their behaviour is unchanged.
    meta = HormonalMetaAgent(
        enable_da=ablation_cfg.get("DA", False),
        enable_na=ablation_cfg.get("NA", False),
        enable_5ht=ablation_cfg.get("5HT", False),
        volatility_threshold=volatility_threshold,
    )
    if ablation_cfg.get("vanilla", False):
        # Extra keys (all default to the original fixed-ε / Huber behaviour, so
        # the ablation's "Vanilla DQN" stays byte-identical): a fixed `epsilon`,
        # a linear ε-anneal (eps_start/eps_end/eps_decay_steps), and a
        # value-corrected target (loss="mse" or reward_scale) — see BASELINE_CONFIGS.
        worker = StaticBaselineWorker(
            state_dim, action_dim, replay_size=replay_size,
            epsilon=ablation_cfg.get("epsilon", cfg.EPSILON_BASE),
            eps_start=ablation_cfg.get("eps_start"),
            eps_end=ablation_cfg.get("eps_end"),
            eps_decay_steps=ablation_cfg.get("eps_decay_steps"),
            loss=ablation_cfg.get("loss", "huber"),
            reward_scale=ablation_cfg.get("reward_scale", 1.0),
        )
    else:
        # DA-strong probe may raise the plastic-coefficient init; None → default.
        kw = {} if plastic_alpha is None else {"plastic_alpha_init": plastic_alpha}
        worker = LocalRLWorker(state_dim, action_dim,
                               replay_size=replay_size, **kw)
    return meta, worker


# ======================================================================
# Experiment 1 — Volatile Multi-Armed Bandit (Section 5)
# ======================================================================
def run_experiment_1(ablation_cfg: dict = None, seed: int = cfg.SEED,
                     label: str = "Full Model") -> dict:
    """Run the volatile bandit experiment.

    Returns:
        dict with keys: rewards, actions, hormones (DA/NA/5HT),
        hyperparams (alpha/tau/gamma), adaptation_latency, label.
    """
    if ablation_cfg is None:
        ablation_cfg = cfg.ABLATION_CONFIGS["Full Model"]

    _seed_all(seed)

    env = VolatileBandit(seed=seed)
    meta, worker = _make_agent(env.observation_dim, env.action_dim,
                               ablation_cfg, replay_size=cfg.EXP1_REPLAY_SIZE)
    meta.hard_reset()

    state = env.reset()
    rewards, actions, optimal_arms = [], [], []

    for step_i in range(cfg.EXP1_TOTAL_STEPS):
        hormone_signal = meta.engine.plastic_gate()  # DA-gated plasticity
        action = worker.select_action(state, hormone_signal=hormone_signal)
        next_state, reward, done, _, info = env.step(action)

        worker.store_transition(state, action, reward, next_state,
                                float(done))
        td_error = worker.update(hormone_signal=hormone_signal)

        # Meta-Agent step → modulate worker
        modulation = meta.step(td_error, reward, done)
        worker.set_modulation(modulation["alpha"], modulation["epsilon"],
                              modulation["gamma"], modulation["punish_gain"])

        rewards.append(reward)
        actions.append(action)
        optimal_arms.append(info["optimal_arm"])  # true optimal arm this step
        state = next_state

    # ── Adaptation latency (per switch, against the CORRECT new arm) ─────
    # For each distribution switch, count the steps needed to re-lock onto
    # the arm that is optimal in the NEW phase (LOCK_N consecutive pulls).
    # The old code hard-coded "arm 4", but the optimal arm cycles through
    # optimal_sequence = [0, 4, 1, 3, 2], so after the last switch the
    # optimum is arm 2 — the old metric could never fire and pinned every
    # config to the worst-case cap.  We now report the mean latency across
    # switches plus the per-switch breakdown.
    LOCK_N = 5
    switch_steps = sorted(cfg.EXP1_SWITCH_STEPS)
    phase_bounds = switch_steps + [cfg.EXP1_TOTAL_STEPS]
    per_switch_latency = []
    for k, s_start in enumerate(switch_steps):
        if s_start >= len(actions):              # switch never reached
            continue
        s_end = min(phase_bounds[k + 1], len(actions))   # next switch/run end
        target_arm = optimal_arms[s_start]       # optimal arm in new phase
        latency = s_end - s_start                # worst case: never re-locks
        consecutive = 0
        for j in range(s_start, s_end):
            if actions[j] == target_arm:
                consecutive += 1
                if consecutive >= LOCK_N:
                    latency = (j - LOCK_N + 1) - s_start  # start of the run
                    break
            else:
                consecutive = 0
        per_switch_latency.append(latency)

    adaptation_latency = (float(np.mean(per_switch_latency))
                          if per_switch_latency else 0.0)

    return {
        "rewards": rewards,
        "actions": actions,
        "optimal_arms": optimal_arms,
        "hormones_da": list(meta.engine.history_da),
        "hormones_na": list(meta.engine.history_na),
        "hormones_ht": list(meta.engine.history_ht),
        "hormones_da_eff": list(meta.engine.history_da_eff),
        "alpha": list(meta.history_alpha),
        "epsilon": list(meta.history_epsilon),
        "gamma": list(meta.history_gamma),
        "adaptation_latency": adaptation_latency,
        "per_switch_latency": per_switch_latency,
        "label": label,
    }


# ======================================================================
# Experiment 2 — High-Stakes Foraging (Section 6)
# ======================================================================
def run_experiment_2(ablation_cfg: dict = None, seed: int = cfg.SEED,
                     label: str = "Full Model") -> dict:
    """Run the high-stakes foraging experiment.

    Returns:
        dict with keys: rewards, actions, cumulative_rewards, death_events,
        survival_steps, hormones, hyperparams, death_count, label.
    """
    if ablation_cfg is None:
        ablation_cfg = cfg.ABLATION_CONFIGS["Full Model"]

    _seed_all(seed)

    env = HighStakesForaging(seed=seed)
    meta, worker = _make_agent(env.observation_dim, env.action_dim,
                               ablation_cfg, replay_size=cfg.EXP2_REPLAY_SIZE)
    meta.hard_reset()

    state = env.reset()
    rewards, actions, cumulative, deaths = [], [], [], []
    steps_since_death = 0
    survival_steps = []

    for step_i in range(cfg.EXP2_TOTAL_STEPS):
        hormone_signal = meta.engine.plastic_gate()  # DA-gated plasticity
        action = worker.select_action(state, hormone_signal=hormone_signal)
        next_state, reward, done, _, info = env.step(action)

        worker.store_transition(state, action, reward, next_state,
                                float(info.get("death", False)))
        td_error = worker.update(hormone_signal=hormone_signal)

        modulation = meta.step(td_error, reward, info.get("death", False))
        worker.set_modulation(modulation["alpha"], modulation["epsilon"],
                              modulation["gamma"], modulation["punish_gain"])

        rewards.append(reward)
        actions.append(action)
        cumulative.append(info.get("cumulative", 0.0))
        steps_since_death += 1

        if info.get("death", False):
            deaths.append(step_i)
            survival_steps.append(steps_since_death)
            steps_since_death = 0
            worker.reset_episode()

        state = next_state

    # Count the trailing survival streak (from the last death — or from the
    # start, if no death ever occurred — to the end of the run). Without this the
    # final interval was dropped, biasing Survival_Rate downward for low-death
    # agents. If the very last step was a death, steps_since_death == 0 and there
    # is no trailing streak to add.
    if steps_since_death > 0:
        survival_steps.append(steps_since_death)

    return {
        "rewards": rewards,
        "actions": actions,
        "cumulative_rewards": cumulative,
        "death_events": deaths,
        "survival_steps": survival_steps,
        "hormones_da": list(meta.engine.history_da),
        "hormones_na": list(meta.engine.history_na),
        "hormones_ht": list(meta.engine.history_ht),
        "hormones_da_eff": list(meta.engine.history_da_eff),
        "alpha": list(meta.history_alpha),
        "epsilon": list(meta.history_epsilon),
        "gamma": list(meta.history_gamma),
        "death_count": len(deaths),
        "total_reward": sum(rewards),
        "label": label,
    }


# ======================================================================
# Experiment 3 — Volatile Risky Foraging (CAPSTONE: DA + NA + 5-HT together)
# ======================================================================
def run_experiment_3(ablation_cfg: dict = None, seed: int = cfg.SEED,
                     label: str = "Full Model",
                     plastic_alpha: float = None) -> dict:
    """Capstone experiment — the moving good arm (Exp 1) fused with the lethal
    high-EV arm (Exp 2), so all three neuromodulators are needed at once.

    Prediction: the Full model (all three) achieves the highest cumulative
    reward; removing NA or DA slows re-adaptation to the moving good arm, and
    removing 5-HT lets the agent get hooked on the lethal arm and die. Thus
    removing ANY single hormone is worse — the multi-hormone synergy claim.

    Returns:
        dict with rewards, actions, cumulative_rewards, death_events,
        survival_steps, adaptation_latency, death_count, total_reward, hormones.
    """
    if ablation_cfg is None:
        ablation_cfg = cfg.ABLATION_CONFIGS["Full Model"]

    _seed_all(seed)

    env = ContextualRiskyForaging(seed=seed)
    meta, worker = _make_agent(env.observation_dim, env.action_dim,
                               ablation_cfg, replay_size=cfg.VRF_REPLAY_SIZE,
                               volatility_threshold=cfg.VRF_VOLATILITY_THRESHOLD,
                               plastic_alpha=plastic_alpha)
    meta.hard_reset()

    state = env.reset()
    rewards, actions, optimal_arms, correct = [], [], [], []
    cumulative, deaths, survival_steps = [], [], []
    steps_since_death = 0

    for step_i in range(cfg.VRF_TOTAL_STEPS):
        hormone_signal = meta.engine.plastic_gate()  # DA-gated plasticity
        action = worker.select_action(state, hormone_signal=hormone_signal)
        next_state, reward, done, _, info = env.step(action)

        worker.store_transition(state, action, reward, next_state,
                                float(info.get("death", False)))
        td_error = worker.update(hormone_signal=hormone_signal)

        modulation = meta.step(td_error, reward, info.get("death", False))
        worker.set_modulation(modulation["alpha"], modulation["epsilon"],
                              modulation["gamma"], modulation["punish_gain"])

        rewards.append(reward)
        actions.append(action)
        optimal_arms.append(info["optimal_arm"])
        correct.append(1 if action == info["optimal_arm"] else 0)
        cumulative.append(info.get("cumulative", 0.0))
        steps_since_death += 1

        if info.get("death", False):
            deaths.append(step_i)
            survival_steps.append(steps_since_death)
            steps_since_death = 0
            worker.reset_episode()

        state = next_state

    # Count the trailing survival streak (see run_experiment_2 for rationale):
    # from the last death (or the start, if none) to the end of the run.
    if steps_since_death > 0:
        survival_steps.append(steps_since_death)

    # ── Re-adaptation latency (per reversal, ACCURACY-based) ─────────────
    # Contextual task: the correct action depends on the cue, so we cannot use
    # "consecutive pulls of one arm". Instead, latency = steps after a reversal
    # until the rolling accuracy (fraction of steps taking the cue's correct
    # action) first reaches VRF_CRIT_ACCURACY over a VRF_CRIT_WINDOW window.
    W = cfg.VRF_CRIT_WINDOW
    crit = cfg.VRF_CRIT_ACCURACY
    switch_steps = sorted(cfg.VRF_SWITCH_STEPS)
    phase_bounds = switch_steps + [cfg.VRF_TOTAL_STEPS]
    corr = np.asarray(correct, dtype=float)
    per_switch_latency = []
    for k, s_start in enumerate(switch_steps):
        if s_start >= len(corr):
            continue
        s_end = min(phase_bounds[k + 1], len(corr))
        latency = s_end - s_start          # worst case: never re-adapts
        for j in range(s_start, s_end - W + 1):
            if corr[j:j + W].mean() >= crit:
                latency = j - s_start
                break
        per_switch_latency.append(latency)

    adaptation_latency = (float(np.mean(per_switch_latency))
                          if per_switch_latency else 0.0)
    overall_accuracy = float(corr.mean()) if len(corr) else 0.0

    return {
        "rewards": rewards,
        "actions": actions,
        "optimal_arms": optimal_arms,
        "cumulative_rewards": cumulative,
        "death_events": deaths,
        "survival_steps": survival_steps,
        "hormones_da": list(meta.engine.history_da),
        "hormones_na": list(meta.engine.history_na),
        "hormones_ht": list(meta.engine.history_ht),
        "hormones_da_eff": list(meta.engine.history_da_eff),
        "alpha": list(meta.history_alpha),
        "epsilon": list(meta.history_epsilon),
        "gamma": list(meta.history_gamma),
        "adaptation_latency": adaptation_latency,
        "per_switch_latency": per_switch_latency,
        "accuracy": overall_accuracy,
        "death_count": len(deaths),
        "total_reward": sum(rewards),
        "risky_arm": env.risky_arm,
        "n_safe": env.n_safe,
        "n_cues": env.n_cues,
        "switch_steps": switch_steps,
        "label": label,
    }


# ======================================================================
# Legacy — CartPole Physics Adaptation (secondary / negative result)
# ======================================================================
def run_experiment_cartpole(ablation_cfg: dict = None, seed: int = cfg.SEED,
                            label: str = "Full Model") -> dict:
    """Run the CartPole physics adaptation experiment.

    Design — "train to competence, THEN perturb" (per seed):
      Phase 1: train under standard gravity until the agent is COMPETENT
               (rolling mean over EXP3_COMPETENCE_WINDOW ≥ EXP3_COMPETENCE_TARGET)
               or a training cap (EXP3_TRAIN_EPISODES) is hit.
      Phase 2: if competent, shock the dynamics AT THAT MOMENT — gravity →
               EXP3_NEW_GRAVITY, actuator force_mag × EXP3_FORCE_SCALE — and run
               EXP3_POST_EPISODES more episodes, measuring recovery.

    Why perturb at competence instead of a fixed late episode: a vanilla DQN on
    CartPole reliably solves the task and then, with continued training,
    catastrophically forgets (episode length collapses back to ~10 and stays
    there). Checking competence at a FIXED late episode lands after that
    collapse and mislabels a solved agent as "never competent", leaving
    recovery undefined for almost every seed. Perturbing at the competence peak
    measures exactly what the experiment asks — recovery from a dynamics shock,
    given the agent had actually learned the task — and is robust to the
    post-solution collapse. If the agent never reaches competence within the
    cap, recovery_time is NaN (flagged, not silently capped).

    Returns:
        dict with episode_lengths, hormones, hyperparams, recovery_time,
        pre_competence, is_competent, perturb_episode, label.
    """
    if ablation_cfg is None:
        ablation_cfg = cfg.ABLATION_CONFIGS["Full Model"]

    _seed_all(seed)

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    meta, worker = _make_agent(state_dim, action_dim, ablation_cfg,
                               replay_size=cfg.EXP3_REPLAY_SIZE)
    meta.hard_reset()

    episode_lengths = []
    all_rewards = []

    def _run_episode(ep_idx: int) -> int:
        """Run one CartPole episode end-to-end; return its length (steps)."""
        state, _ = env.reset(seed=seed + ep_idx)
        worker.reset_episode()
        ep_reward = 0.0
        step_count = 0
        for _t in range(500):  # CartPole-v1 caps at 500 steps
            hormone_signal = meta.engine.plastic_gate()  # DA-gated plasticity
            action = worker.select_action(state, hormone_signal=hormone_signal)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            worker.store_transition(state, action, reward, next_state,
                                    float(done))
            td_error = worker.update(hormone_signal=hormone_signal)

            modulation = meta.step(td_error, reward, done)
            worker.set_modulation(modulation["alpha"], modulation["epsilon"],
                                  modulation["gamma"], modulation["punish_gain"])

            ep_reward += reward
            step_count += 1
            state = next_state
            if done:
                break
        episode_lengths.append(step_count)
        all_rewards.append(ep_reward)
        return step_count

    # ── Phase 1: train until competent (or hit the cap) ──────────────────
    rolling = deque(maxlen=cfg.EXP3_COMPETENCE_WINDOW)
    is_competent = False
    for ep in range(cfg.EXP3_TRAIN_EPISODES):
        rolling.append(_run_episode(ep))
        if (len(rolling) == cfg.EXP3_COMPETENCE_WINDOW
                and float(np.mean(rolling)) >= cfg.EXP3_COMPETENCE_TARGET):
            is_competent = True
            break

    pre_competence = float(np.mean(rolling)) if rolling else 0.0
    perturb_episode = len(episode_lengths)  # per-seed: where the shock lands

    # ── Phase 2: perturb at the competence peak, then measure recovery ───
    if is_competent:
        env.unwrapped.gravity = cfg.EXP3_NEW_GRAVITY
        # CartPole has no friction parameter; scaling force_mag is our proxy
        # for the changed dynamics (weaker actuator).
        env.unwrapped.force_mag = env.unwrapped.force_mag * cfg.EXP3_FORCE_SCALE

        post_lengths = [_run_episode(perturb_episode + ep)
                        for ep in range(cfg.EXP3_POST_EPISODES)]

        # Recovery = first time 3 consecutive post-shock episodes clear the
        # recovery bar; worst case (never recovers) = the full post-window.
        recovery_time = float(cfg.EXP3_POST_EPISODES)
        consecutive = 0
        for i, length in enumerate(post_lengths):
            if length >= cfg.EXP3_RECOVERY_TARGET:
                consecutive += 1
                if consecutive >= 3:
                    recovery_time = float(i - 2)
                    break
            else:
                consecutive = 0
    else:
        recovery_time = float("nan")  # undefined — never competent within cap

    env.close()

    return {
        "episode_lengths": episode_lengths,
        "rewards": all_rewards,
        "hormones_da": list(meta.engine.history_da),
        "hormones_na": list(meta.engine.history_na),
        "hormones_ht": list(meta.engine.history_ht),
        "hormones_da_eff": list(meta.engine.history_da_eff),
        "alpha": list(meta.history_alpha),
        "epsilon": list(meta.history_epsilon),
        "gamma": list(meta.history_gamma),
        "recovery_time": recovery_time,
        "pre_competence": pre_competence,
        "is_competent": is_competent,
        "perturb_episode": perturb_episode,
        "label": label,
    }

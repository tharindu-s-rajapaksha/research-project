"""
experiments.py — Experiment Runners for Experiments 1, 2, and 3
=================================================================
Each experiment function returns a results dict containing all logged
data needed for evaluation and visualization.

Experiment 1: Volatile Multi-Armed Bandit  (NA Test)
Experiment 2: High-Stakes Foraging         (5-HT Test)
Experiment 3: CartPole Physics Adaptation   (Learning Rate Test)
"""

import random

import numpy as np
import gymnasium as gym
import torch

import config as cfg
from meta_agent import HormonalMetaAgent
from worker import LocalRLWorker, StaticBaselineWorker
from environments import VolatileBandit, HighStakesForaging


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
                replay_size: int = cfg.REPLAY_SIZE):
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
    meta = HormonalMetaAgent(
        enable_da=ablation_cfg["DA"],
        enable_na=ablation_cfg["NA"],
        enable_5ht=ablation_cfg["5HT"],
    )
    if ablation_cfg.get("vanilla", False):
        worker = StaticBaselineWorker(state_dim, action_dim,
                                      replay_size=replay_size)
    else:
        worker = LocalRLWorker(state_dim, action_dim,
                               replay_size=replay_size)
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

    # If no death occurred, survival = total steps
    if not deaths:
        survival_steps.append(cfg.EXP2_TOTAL_STEPS)

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
# Experiment 3 — CartPole Physics Adaptation (Section 7)
# ======================================================================
def run_experiment_3(ablation_cfg: dict = None, seed: int = cfg.SEED,
                     label: str = "Full Model") -> dict:
    """Run the CartPole physics adaptation experiment.

    Phase 1: Train under standard gravity (9.8) for EXP3_TRAIN_EPISODES.
    Phase 2: At EXP3_PERTURB_EPISODE, shock the dynamics — gravity → 29.4
             (3× default) and actuator force_mag × EXP3_FORCE_SCALE (0.5) —
             then run EXP3_POST_EPISODES more episodes.

    Recovery is only defined if the agent was COMPETENT before the shock
    (pre-perturb rolling mean ≥ EXP3_COMPETENCE_TARGET); otherwise there is
    nothing to "recover" and recovery_time is NaN.

    Returns:
        dict with episode_lengths, hormones, hyperparams, recovery_time,
        pre_competence, is_competent, label.
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

    total_episodes = cfg.EXP3_TRAIN_EPISODES + cfg.EXP3_POST_EPISODES
    episode_lengths = []
    all_rewards = []
    perturbed = False

    for ep in range(total_episodes):
        # Apply perturbation at the right episode
        if ep == cfg.EXP3_PERTURB_EPISODE and not perturbed:
            env.unwrapped.gravity = cfg.EXP3_NEW_GRAVITY
            # CartPole has no friction parameter; we scale force_mag as a
            # proxy for the changed dynamics (weaker actuator).
            env.unwrapped.force_mag = env.unwrapped.force_mag * cfg.EXP3_FORCE_SCALE
            perturbed = True

        state, _ = env.reset(seed=seed + ep)
        worker.reset_episode()
        ep_reward = 0.0
        step_count = 0

        for t in range(500):  # CartPole max steps
            hormone_signal = meta.engine.plastic_gate()  # DA-gated plasticity
            action = worker.select_action(state,
                                          hormone_signal=hormone_signal)
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

    env.close()

    # ── Pre-perturbation competence gate ────────────────────────────────
    # "Recovery" is only meaningful if the agent actually solved CartPole
    # before the shock.  Otherwise there is nothing to recover to, and the
    # recovery_time is NaN (flagged, not silently capped) so it is excluded
    # from the recovery statistics rather than polluting them.
    pre_lengths = episode_lengths[:cfg.EXP3_PERTURB_EPISODE]
    comp_window = pre_lengths[-cfg.EXP3_COMPETENCE_WINDOW:]
    pre_competence = float(np.mean(comp_window)) if comp_window else 0.0
    is_competent = pre_competence >= cfg.EXP3_COMPETENCE_TARGET

    post_lengths = episode_lengths[cfg.EXP3_PERTURB_EPISODE:]
    if is_competent:
        # Worst case (competent but never recovers) = full post-window.
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
        recovery_time = float("nan")  # undefined — never competent pre-shock

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
        "perturb_episode": cfg.EXP3_PERTURB_EPISODE,
        "label": label,
    }

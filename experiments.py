"""
experiments.py — Experiment Runners for Experiments 1, 2, and 3
=================================================================
Each experiment function returns a results dict containing all logged
data needed for evaluation and visualization.

Experiment 1: Volatile Multi-Armed Bandit  (NA Test)
Experiment 2: High-Stakes Foraging         (5-HT Test)
Experiment 3: CartPole Physics Adaptation   (Learning Rate Test)
"""

import numpy as np
import gymnasium as gym
import torch

import config as cfg
from meta_agent import HormonalMetaAgent
from worker import LocalRLWorker, StaticBaselineWorker
from environments import VolatileBandit, HighStakesForaging


def _make_agent(state_dim: int, action_dim: int, ablation_cfg: dict):
    """Factory: build Meta-Agent + Worker pair based on ablation config.

    Uses the config's explicit ``worker`` ("plastic"/"static") and ``plastic``
    flags. Falls back to inferring a static baseline from all-hormones-off for
    backward compatibility with older config dicts.
    """
    worker_type = ablation_cfg.get("worker")
    if worker_type is None:
        worker_type = ("static" if not (ablation_cfg["DA"] or ablation_cfg["NA"]
                                         or ablation_cfg["5HT"]) else "plastic")

    if worker_type == "static":
        worker = StaticBaselineWorker(state_dim, action_dim)
        meta = HormonalMetaAgent(enable_da=False, enable_na=False,
                                  enable_5ht=False)
    else:
        worker = LocalRLWorker(state_dim, action_dim,
                               plastic=ablation_cfg.get("plastic", True))
        meta = HormonalMetaAgent(
            enable_da=ablation_cfg["DA"],
            enable_na=ablation_cfg["NA"],
            enable_5ht=ablation_cfg["5HT"],
        )
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

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = VolatileBandit(seed=seed)
    meta, worker = _make_agent(env.observation_dim, env.action_dim,
                               ablation_cfg)
    meta.hard_reset()

    state = env.reset()
    rewards, actions = [], []

    for step_i in range(cfg.EXP1_TOTAL_STEPS):
        hormone_signal = meta.engine.get_vector()[0]  # DA_eff
        action = worker.select_action(state, hormone_signal=hormone_signal)
        next_state, reward, done, _, info = env.step(action)

        worker.store_transition(state, action, reward, next_state,
                                float(done))
        td_error = worker.update(hormone_signal=hormone_signal)

        # Meta-Agent step → modulate worker
        modulation = meta.step(td_error, reward, done)
        worker.set_modulation(modulation["alpha"], modulation["tau"],
                              modulation["gamma"])

        rewards.append(reward)
        actions.append(action)
        state = next_state

    # ── Adaptation latency ───────────────────────────────────────────
    # For each reward switch, measure how many steps it takes the agent to
    # lock onto the NEW optimal arm (5 consecutive pulls). The optimal arm
    # per phase comes from the environment's own schedule, so the metric
    # always tracks the *current* best arm (not a hardcoded one).
    switch_steps = sorted(cfg.EXP1_SWITCH_STEPS)
    per_switch_latency = []
    for j, s in enumerate(switch_steps):
        next_s = switch_steps[j + 1] if j + 1 < len(switch_steps) else cfg.EXP1_TOTAL_STEPS
        target_arm = env.optimal_sequence[j + 1]
        window = actions[s:next_s]
        latency = len(window)  # worst case: never adapted within this phase
        consecutive = 0
        for i, a in enumerate(window):
            if a == target_arm:
                consecutive += 1
                if consecutive >= 5:
                    latency = i - 4  # start of the 5-pull run
                    break
            else:
                consecutive = 0
        per_switch_latency.append(latency)

    adaptation_latency = float(np.mean(per_switch_latency)) if per_switch_latency else 0.0

    return {
        "rewards": rewards,
        "actions": actions,
        "hormones_da": list(meta.engine.history_da),
        "hormones_na": list(meta.engine.history_na),
        "hormones_ht": list(meta.engine.history_ht),
        "hormones_da_eff": list(meta.engine.history_da_eff),
        "alpha": list(meta.history_alpha),
        "tau": list(meta.history_tau),
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

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = HighStakesForaging(seed=seed)
    meta, worker = _make_agent(env.observation_dim, env.action_dim,
                               ablation_cfg)
    meta.hard_reset()

    state = env.reset()
    rewards, actions, cumulative, deaths = [], [], [], []
    steps_since_death = 0
    survival_steps = []

    for step_i in range(cfg.EXP2_TOTAL_STEPS):
        hormone_signal = meta.engine.get_vector()[0]
        action = worker.select_action(state, hormone_signal=hormone_signal)
        next_state, reward, done, _, info = env.step(action)

        worker.store_transition(state, action, reward, next_state,
                                float(info.get("death", False)))
        td_error = worker.update(hormone_signal=hormone_signal)

        modulation = meta.step(td_error, reward, info.get("death", False))
        worker.set_modulation(modulation["alpha"], modulation["tau"],
                              modulation["gamma"])

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
        "tau": list(meta.history_tau),
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
    Phase 2: Change gravity to 30.0, friction to 0.01, run for
             EXP3_POST_EPISODES more episodes.

    Returns:
        dict with episode_lengths, hormones, hyperparams, recovery_time,
        label.
    """
    if ablation_cfg is None:
        ablation_cfg = cfg.ABLATION_CONFIGS["Full Model"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    meta, worker = _make_agent(state_dim, action_dim, ablation_cfg)
    meta.hard_reset()

    total_episodes = cfg.EXP3_TRAIN_EPISODES + cfg.EXP3_POST_EPISODES
    episode_lengths = []
    all_rewards = []
    perturbed = False

    for ep in range(total_episodes):
        # Apply perturbation at the right episode
        if ep == cfg.EXP3_PERTURB_EPISODE and not perturbed:
            env.unwrapped.gravity = cfg.EXP3_NEW_GRAVITY
            # CartPole doesn't have a friction parameter directly;
            # we modify force_mag as a proxy for changed dynamics.
            env.unwrapped.force_mag = env.unwrapped.force_mag * cfg.EXP3_NEW_FRICTION
            perturbed = True

        state, _ = env.reset(seed=seed + ep)
        worker.reset_episode()
        ep_reward = 0.0
        step_count = 0

        for t in range(500):  # CartPole max steps
            hormone_signal = meta.engine.get_vector()[0]
            action = worker.select_action(state,
                                          hormone_signal=hormone_signal)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            worker.store_transition(state, action, reward, next_state,
                                    float(done))
            td_error = worker.update(hormone_signal=hormone_signal)

            modulation = meta.step(td_error, reward, done)
            worker.set_modulation(modulation["alpha"], modulation["tau"],
                                  modulation["gamma"])

            ep_reward += reward
            step_count += 1
            state = next_state
            if done:
                break

        episode_lengths.append(step_count)
        all_rewards.append(ep_reward)

    env.close()

    # Recovery time: first episode after perturbation that sustains
    # >= RECOVERY_TARGET steps for 3 consecutive episodes
    recovery_time = cfg.EXP3_POST_EPISODES  # worst case
    post_lengths = episode_lengths[cfg.EXP3_PERTURB_EPISODE:]
    consecutive = 0
    for i, length in enumerate(post_lengths):
        if length >= cfg.EXP3_RECOVERY_TARGET:
            consecutive += 1
            if consecutive >= 3:
                recovery_time = i - 2
                break
        else:
            consecutive = 0

    return {
        "episode_lengths": episode_lengths,
        "rewards": all_rewards,
        "hormones_da": list(meta.engine.history_da),
        "hormones_na": list(meta.engine.history_na),
        "hormones_ht": list(meta.engine.history_ht),
        "hormones_da_eff": list(meta.engine.history_da_eff),
        "alpha": list(meta.history_alpha),
        "tau": list(meta.history_tau),
        "gamma": list(meta.history_gamma),
        "recovery_time": recovery_time,
        "perturb_episode": cfg.EXP3_PERTURB_EPISODE,
        "label": label,
    }

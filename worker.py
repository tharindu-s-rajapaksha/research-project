"""
worker.py — Local RL Worker (The Actuator)
===========================================
A DQN-based reinforcement learning agent whose hyperparameters (learning
rate α, exploration rate ε, discount factor γ, and 5-HT loss-aversion gain)
are dynamically modulated by the Hormonal Meta-Agent.

Implements:
    • Experience replay buffer
    • Target network with periodic sync
    • ε-greedy action selection with NA-modulated exploration rate
    • Integration with PlasticNetwork (differentiable plasticity)
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import config as cfg
from plasticity import PlasticNetwork

# Named tuple for replay buffer entries
Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """Fixed-size circular experience-replay buffer."""

    def __init__(self, capacity: int = cfg.REPLAY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class LocalRLWorker:
    """DQN Worker with neuromodulated hyperparameters.

    The Worker receives modulation signals from the Meta-Agent every step
    and adjusts its:
        • Learning rate   α  (via optimizer param_group lr)
        • Exploration ε  (ε-greedy rate, driven by NA)
        • Discount       γ  (TD target computation)
        • Loss aversion  g  (5-HT amplification of negative rewards)
    """

    def __init__(self, state_dim: int, action_dim: int, device: torch.device = cfg.DEVICE,
                 replay_size: int = cfg.REPLAY_SIZE,
                 plastic_alpha_init: float = cfg.PLASTIC_ALPHA_INIT):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Online & Target networks (with differentiable plasticity). The plastic
        # coefficient init is tunable for the DA-strong probe; default unchanged.
        self.policy_net = PlasticNetwork(state_dim, cfg.HIDDEN_DIM, action_dim,
                                         plastic_alpha_init=plastic_alpha_init).to(device)
        self.target_net = PlasticNetwork(state_dim, cfg.HIDDEN_DIM, action_dim,
                                         plastic_alpha_init=plastic_alpha_init).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer — lr will be overridden each step
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.ALPHA_BASE)

        # Replay — capacity is experiment-specific (see EXP*_REPLAY_SIZE)
        self.memory = ReplayBuffer(capacity=replay_size)

        # Dynamic hyperparameters (set by Meta-Agent)
        self.alpha   = cfg.ALPHA_BASE
        self.epsilon = cfg.EPSILON_BASE
        self.gamma   = cfg.GAMMA_BASE
        self.punish_gain = 1.0          # 5-HT loss-aversion factor (≥ 1)

        # Per-action harm estimate (EMA of |negative reward|) for the 5-HT
        # behavioural-inhibition penalty at action selection.
        self.harm_estimate = np.zeros(action_dim, dtype=np.float32)

        # Step counter for target network sync
        self._step_count = 0

    # ──────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray, hormone_signal: float = 1.0) -> int:
        """ε-greedy action selection with NA-modulated exploration rate ε.

        Args:
            state:          Current observation (numpy).
            hormone_signal: DA-gated plasticity signal (``plastic_gate``; 0 at
                            rest) for the plastic forward pass.

        Returns:
            Chosen action index.
        """
        # The greedy forward pass still runs (and advances the Hebbian trace)
        # so plasticity reflects lived experience even on exploratory steps.
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(s, hormone_signal).squeeze(0).cpu().numpy()

        # 5-HT behavioural inhibition: when serotonin is elevated
        # (punish_gain > 1), subtract a penalty from actions with a learned
        # harm history, actively withholding risky actions.  At rest
        # (punish_gain = 1) the penalty is 0, so the baseline is unaffected.
        if self.punish_gain > 1.0:
            penalty = ((self.punish_gain - 1.0) * cfg.RISK_INHIBITION_WEIGHT
                       * self.harm_estimate)
            q_values = q_values - penalty

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        return int(np.argmax(q_values))

    # ──────────────────────────────────────────────────────────────────
    # Learning
    # ──────────────────────────────────────────────────────────────────
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        # Update the per-action harm estimate (EMA of loss magnitude) used by
        # the 5-HT behavioural-inhibition penalty in select_action.
        d = cfg.HARM_EMA_DECAY
        loss_mag = max(0.0, -float(reward))
        self.harm_estimate[action] = (d * self.harm_estimate[action]
                                      + (1.0 - d) * loss_mag)

    def update(self, hormone_signal: float = 1.0) -> float:
        """Perform one gradient step on the policy network.

        Args:
            hormone_signal: DA-gated plasticity signal (``plastic_gate``; 0 at
                            rest) passed to the plastic forward pass.

        Returns:
            TD error (float) for the Meta-Agent.
        """
        if len(self.memory) < cfg.BATCH_SIZE:
            return 0.0

        # --- Apply modulated learning rate ---
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.alpha

        batch = self.memory.sample(cfg.BATCH_SIZE)
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(states, hormone_signal).gather(1, actions)

        # Target: r + γ max_a' Q_target(s', a')
        # The target net is a frozen snapshot — do NOT advance its Hebbian
        # trace, or the bootstrap target would drift every update.
        with torch.no_grad():
            next_q = self.target_net(next_states, hormone_signal,
                                     update_trace=False).max(1)[0]
            target = rewards + self.gamma * next_q * (1.0 - dones)

        td_error = (target.unsqueeze(1) - q_values).mean().item()

        # 5-HT loss aversion: up-weight the LEARNING SIGNAL from punishing
        # (negative-reward) transitions by punish_gain, so death experiences
        # dominate the gradient and the learned value of harmful actions
        # falls.  We weight the per-sample loss rather than scaling the
        # reward, because the Huber (smooth_l1) loss saturates its gradient
        # for large errors — amplifying the reward magnitude would be clipped
        # away, whereas re-weighting the loss is not.  punish_gain = 1 at rest.
        per_sample = F.smooth_l1_loss(q_values, target.unsqueeze(1),
                                      reduction="none").squeeze(1)
        if self.punish_gain > 1.0:
            weights = torch.where(rewards < 0.0,
                                  torch.full_like(rewards, self.punish_gain),
                                  torch.ones_like(rewards))
            loss = (per_sample * weights).sum() / weights.sum()
        else:
            loss = per_sample.mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # --- Periodic target network sync ---
        self._step_count += 1
        if self._step_count % cfg.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return td_error

    def set_modulation(self, alpha: float, epsilon: float, gamma: float,
                       punish_gain: float = 1.0):
        """Apply modulated hyperparameters from the Meta-Agent."""
        self.alpha       = alpha
        self.epsilon     = epsilon
        self.gamma       = gamma
        self.punish_gain = punish_gain

    def reset_episode(self):
        """Reset plastic traces for a new episode lifetime."""
        self.policy_net.reset_traces()


class StaticBaselineWorker:
    """A standard ε-greedy DQN with fixed hyperparameters (no modulation).

    Used both as the ablation control condition (``Vanilla DQN``) and as the
    pool of FAIR reference baselines for the "beats standard RL" claim
    (``baselines.py``). The extra keyword arguments below all default to values
    that reproduce the original fixed-ε, Huber-loss behaviour EXACTLY (so the
    Vanilla-DQN config stays byte-identical to the Static Baseline):

        • eps_start / eps_end / eps_decay_steps — a standard linear ε-annealing
          schedule (e.g. 1.0 → 0.05 over N action steps). When ``eps_start`` is
          None (default) ε is the fixed ``epsilon``, as before.
        • loss ∈ {"huber" (default), "mse"} and ``reward_scale`` — the
          value-corrected baseline: Huber clips the gradient of the rare −500
          catastrophe, so a plain DQN under-weights it. MSE (no clipping) or
          scaling the reward into Huber's balanced range tests whether 5-HT is
          genuinely necessary or merely fixes that Huber artefact.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 device: torch.device = cfg.DEVICE,
                 epsilon: float = cfg.EPSILON_BASE,
                 replay_size: int = cfg.REPLAY_SIZE,
                 eps_start: float = None, eps_end: float = None,
                 eps_decay_steps: int = None,
                 loss: str = "huber", reward_scale: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.epsilon = epsilon
        self._replay_size = replay_size

        # Optional linear ε-annealing schedule (None → fixed ε, original behaviour)
        self._eps_start = eps_start
        self._eps_end = eps_end if eps_end is not None else epsilon
        self._eps_decay_steps = eps_decay_steps
        self._act_count = 0

        # Value-target options (defaults reproduce the original Huber loss exactly)
        self._loss = loss
        self._reward_scale = reward_scale

        # Standard MLP (no plasticity)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, action_dim),
        ).to(device)

        self.target_net = nn.Sequential(
            nn.Linear(state_dim, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, action_dim),
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=cfg.ALPHA_BASE)
        self.memory = ReplayBuffer(capacity=replay_size)
        self._step_count = 0

    def _current_epsilon(self) -> float:
        """Effective ε this step. Fixed by default; a linear anneal from
        ``eps_start`` to ``eps_end`` over ``eps_decay_steps`` when scheduled."""
        if self._eps_start is None:
            return self.epsilon
        frac = min(1.0, self._act_count / max(1, self._eps_decay_steps))
        self._act_count += 1
        return self._eps_start + (self._eps_end - self._eps_start) * frac

    def select_action(self, state: np.ndarray, **_kwargs) -> int:
        if random.random() < self._current_epsilon():
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(s).argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self, **_kwargs) -> float:
        if len(self.memory) < cfg.BATCH_SIZE:
            return 0.0

        batch = self.memory.sample(cfg.BATCH_SIZE)
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        if self._reward_scale != 1.0:               # value-corrected baseline
            rewards = rewards * self._reward_scale
        next_states = torch.FloatTensor(
            np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + cfg.GAMMA_BASE * next_q * (1.0 - dones)

        td_error = (target.unsqueeze(1) - q_values).mean().item()
        if self._loss == "mse":                     # no gradient clipping of the
            loss = F.mse_loss(q_values, target.unsqueeze(1))   # rare −500 target
        else:
            loss = F.smooth_l1_loss(q_values, target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % cfg.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return td_error

    def set_modulation(self, alpha, epsilon, gamma, punish_gain=1.0):
        """No-op — vanilla DQN ignores modulation."""
        pass

    def reset_episode(self):
        """No-op — no plasticity traces to reset."""
        pass

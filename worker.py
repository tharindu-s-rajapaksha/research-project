"""
worker.py — Local RL Worker (The Actuator)
===========================================
A DQN-based reinforcement learning agent whose hyperparameters (learning
rate α, exploration temperature τ, discount factor γ) are dynamically
modulated by the Hormonal Meta-Agent.

Implements:
    • Experience replay buffer
    • Target network with periodic sync
    • Softmax action selection with modulable temperature
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
        • Learning rate  α  (via optimizer param_group lr)
        • Temperature   τ  (softmax exploration)
        • Discount      γ  (TD target computation)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 device: torch.device = cfg.DEVICE):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Online & Target networks (with differentiable plasticity)
        self.policy_net = PlasticNetwork(
            state_dim, cfg.HIDDEN_DIM, action_dim).to(device)
        self.target_net = PlasticNetwork(
            state_dim, cfg.HIDDEN_DIM, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer — lr will be overridden each step
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=cfg.ALPHA_BASE)

        # Replay
        self.memory = ReplayBuffer()

        # Dynamic hyperparameters (set by Meta-Agent)
        self.alpha = cfg.ALPHA_BASE
        self.tau   = cfg.TAU_BASE
        self.gamma = cfg.GAMMA_BASE

        # Step counter for target network sync
        self._step_count = 0

    # ──────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray,
                      hormone_signal: float = 1.0) -> int:
        """Softmax action selection with modulable temperature τ.

        Args:
            state:          Current observation (numpy).
            hormone_signal: DA_eff for the plastic forward pass.

        Returns:
            Chosen action index.
        """
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(s, hormone_signal)
            # Clamp Q-values to prevent numerical overflow in softmax
            q_values = q_values.clamp(-100.0, 100.0)
            tau = max(self.tau, 0.01)
            logits = q_values / tau
            # Stabilize: subtract max for numerical safety
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            probs = F.softmax(logits, dim=-1)
            # Guard against NaN (fallback to uniform)
            if torch.isnan(probs).any() or (probs <= 0).all():
                return random.randrange(self.action_dim)
            action = torch.multinomial(probs, 1).item()
        return action

    # ──────────────────────────────────────────────────────────────────
    # Learning
    # ──────────────────────────────────────────────────────────────────
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self, hormone_signal: float = 1.0) -> float:
        """Perform one gradient step on the policy network.

        Args:
            hormone_signal: DA_eff passed to the plastic forward pass.

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
        next_states = torch.FloatTensor(
            np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(states, hormone_signal).gather(1, actions)

        # Target: r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states, hormone_signal).max(1)[0]
            target = rewards + self.gamma * next_q * (1.0 - dones)

        td_error = (target.unsqueeze(1) - q_values).mean().item()
        loss = F.smooth_l1_loss(q_values, target.unsqueeze(1))

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

    def set_modulation(self, alpha: float, tau: float, gamma: float):
        """Apply modulated hyperparameters from the Meta-Agent."""
        self.alpha = alpha
        self.tau   = tau
        self.gamma = gamma

    def reset_episode(self):
        """Reset plastic traces for a new episode lifetime."""
        self.policy_net.reset_traces()


class StaticBaselineWorker:
    """A standard ε-greedy DQN with fixed hyperparameters (no modulation).

    Used as the control condition in ablation studies.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 device: torch.device = cfg.DEVICE,
                 epsilon: float = 0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.epsilon = epsilon

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
        self.memory = ReplayBuffer()
        self._step_count = 0

    def select_action(self, state: np.ndarray, **_kwargs) -> int:
        if random.random() < self.epsilon:
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
        next_states = torch.FloatTensor(
            np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + cfg.GAMMA_BASE * next_q * (1.0 - dones)

        td_error = (target.unsqueeze(1) - q_values).mean().item()
        loss = F.smooth_l1_loss(q_values, target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % cfg.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return td_error

    def set_modulation(self, alpha, tau, gamma):
        """No-op — static baseline ignores modulation."""
        pass

    def reset_episode(self):
        """No-op — no plasticity traces to reset."""
        pass

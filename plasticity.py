"""
plasticity.py — Differentiable Plasticity & Three-Factor Learning
==================================================================
Custom PyTorch layer `NeuromodulatedLinear` that maintains per-synapse
Hebbian eligibility traces and modulates active weights via a hormonal
signal during the forward pass.

Key equations (Section 4):
    E_t = (1 - η_decay) · E_{t-1} + η_trace · (Pre_i × Post_j)
    W_active = W_baseline + (Hormone_signal × E_t)
    ΔW = (Pre × Post × M) + η_static × δ_TD
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class NeuromodulatedLinear(nn.Module):
    """A fully-connected layer with differentiable Hebbian plasticity.

    During the forward pass, the effective weight matrix is:
        W_active = W_baseline + hormone_signal × E_t

    where E_t is the per-synapse eligibility (Hebbian) trace updated
    according to the three-factor learning rule.
    """

    def __init__(self, in_features: int, out_features: int,
                 eta_decay: float = cfg.ETA_DECAY,
                 eta_trace: float = cfg.ETA_TRACE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eta_decay = eta_decay
        self.eta_trace = eta_trace

        # Baseline (slow) weights — updated by backpropagation.
        # Use the EXACT nn.Linear default init so that, when plasticity is
        # gated off (hormone_signal = 0), this layer is distributionally
        # identical to a standard Linear layer — otherwise the "same
        # architecture" static baseline would train differently by accident.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

        # Hebbian eligibility trace — *not* a parameter (no gradient)
        self.register_buffer("hebb_trace", torch.zeros(out_features, in_features))

        # Learnable plasticity coefficient (α in Backpropamine)
        self.alpha = nn.Parameter(torch.full((out_features, in_features), 0.01))

    def reset_trace(self):
        """Zero the Hebbian trace at the start of a new episode/lifetime."""
        self.hebb_trace.zero_()

    def forward(self, x: torch.Tensor,
                hormone_signal: float = 1.0,
                update_trace: bool = True) -> torch.Tensor:
        """Forward pass with plastic weight modulation.

        Args:
            x:               Input tensor of shape (batch, in_features).
            hormone_signal:  Scalar from the Meta-Agent's hormonal vector
                             (typically DA_eff). Controls the magnitude of
                             the Hebbian contribution.
            update_trace:    If False, the eligibility trace is NOT advanced.
                             Used for the (frozen) target network, whose trace
                             must stay a fixed snapshot rather than drift with
                             every target computation.

        Returns:
            Output tensor of shape (batch, out_features).
        """
        # W_active = W_baseline + hormone_signal × α × E_t  (Section 4A)
        w_active = self.weight + hormone_signal * self.alpha * self.hebb_trace
        output = F.linear(x, w_active, self.bias)

        # ── Update eligibility trace (Hebbian rule) ──────────────────
        # E_t = (1 - η_decay) · E_{t-1} + η_trace · (Pre_i × Post_j)
        if update_trace:
            with torch.no_grad():
                # Pre = mean over batch of x;  Post = mean over batch of output
                pre  = x.mean(dim=0)          # (in_features,)
                post = output.mean(dim=0)     # (out_features,)
                outer = torch.outer(post, pre)  # (out, in)
                self.hebb_trace = ((1.0 - self.eta_decay) * self.hebb_trace
                                   + self.eta_trace * outer)

        return output


class PlasticNetwork(nn.Module):
    """A small MLP where selected layers are `NeuromodulatedLinear`.

    Architecture:
        Input → NeuromodulatedLinear(hidden) → ReLU
              → NeuromodulatedLinear(hidden) → ReLU
              → Linear(output)   [head — standard, no plasticity]
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.plastic1 = NeuromodulatedLinear(input_dim, hidden_dim)
        self.plastic2 = NeuromodulatedLinear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor,
                hormone_signal: float = 1.0,
                update_trace: bool = True) -> torch.Tensor:
        """
        Args:
            x:              State tensor  (batch, input_dim).
            hormone_signal: Scalar modulation from Meta-Agent.
            update_trace:   If False, Hebbian traces are not advanced
                            (used for the frozen target network).

        Returns:
            Q-values tensor (batch, output_dim).
        """
        x = F.relu(self.plastic1(x, hormone_signal, update_trace))
        x = F.relu(self.plastic2(x, hormone_signal, update_trace))
        return self.head(x)

    def reset_traces(self):
        """Reset all Hebbian traces (start of episode/lifetime)."""
        self.plastic1.reset_trace()
        self.plastic2.reset_trace()

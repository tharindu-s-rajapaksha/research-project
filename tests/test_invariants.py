"""
test_invariants.py — Backing the "reduces to baseline at rest" invariant.

The fair single-variable ablation depends on every hormonal modulation reducing
to its base value when the hormone sits at its resting concentration, so that the
"Static Baseline" is exactly the Full Model with hormones frozen. This test pins
that invariant: at rest, α=α_base, ε=ε_base, γ=γ_base, punishment gain=1, and the
DA-gated plastic factor=0.

Also checks the Holm helper on a known case.

Run:  python test_invariants.py      (asserts; exits non-zero on failure)
   or pytest test_invariants.py
"""

import config as cfg
from meta_agent import HormonalMetaAgent
from evaluation import _holm

_TOL = 1e-9


def test_rest_invariant():
    meta = HormonalMetaAgent(enable_da=True, enable_na=True, enable_5ht=True)
    B = cfg.HORMONE_BASELINE

    # DA_eff at rest = DA_raw * (1 - sigma(0)) = 1.0 * 0.5 = 0.5
    alpha = meta._modulate_lr(B * 0.5)
    epsilon = meta._modulate_epsilon(B)
    gamma = meta._modulate_discount(B)
    gain = meta._punishment_gain(B)
    gate = meta.engine.plastic_gate()

    assert abs(alpha - cfg.ALPHA_BASE) < _TOL, f"alpha {alpha} != {cfg.ALPHA_BASE}"
    assert abs(epsilon - cfg.EPSILON_BASE) < _TOL, f"epsilon {epsilon} != {cfg.EPSILON_BASE}"
    assert abs(gamma - cfg.GAMMA_BASE) < _TOL, f"gamma {gamma} != {cfg.GAMMA_BASE}"
    assert abs(gain - 1.0) < _TOL, f"punish_gain {gain} != 1.0"
    assert abs(gate - 0.0) < _TOL, f"plastic_gate {gate} != 0.0"


def test_holm_monotone_and_capped():
    # Two tiny p's stay tiny after scaling; the largest is scaled by 1.
    adj = _holm([0.01, 0.04, 0.5])
    # Holm: sorted 0.01,0.04,0.5 -> 3*0.01=0.03, 2*0.04=0.08, 1*0.5=0.5
    assert abs(adj[0] - 0.03) < 1e-12, adj
    assert abs(adj[1] - 0.08) < 1e-12, adj
    assert abs(adj[2] - 0.5) < 1e-12, adj
    # Monotone non-decreasing in sorted order and capped at 1.
    assert _holm([0.9, 0.9]) == [1.0, 1.0]
    assert _holm([]) == []


if __name__ == "__main__":
    test_rest_invariant()
    test_holm_monotone_and_capped()
    print("OK — rest invariant holds (alpha/epsilon/gamma/gain/plastic_gate) "
          "and Holm helper is correct.")

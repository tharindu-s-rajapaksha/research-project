"""
tools/bandit_gamma0_probe.py — SENSITIVITY CHECK (Threats-to-Validity, M4).

The bandit (Exp 1) is treated as a full MDP: the worker bootstraps with γ=0.99
on a task that is really a *contextual bandit* (state = one-hot of the last
action; reward depends only on the current pull). γ=0.99 inflates and "sticks"
the Q-values across switches. It is applied identically to every config, so the
ablation stays fair — but a reviewer may ask whether the headline NA result
survives the correct γ=0 (immediate-reward) treatment.

This probe monkeypatches γ_base = γ_max = 0 (so every config, static and
modulated, uses γ=0) and re-runs Exp 1 for Full / Ablated-NA / Ablated-DA /
Static, reporting adaptation latency with paired Full-vs-config tests. Expected:
NA still drives adaptation (Full ≪ Ablated-NA), confirming the conclusion is not
an artefact of the γ choice. Runs SERIALLY (monkeypatch would not propagate to
spawned workers).

Usage:
    python tools/bandit_gamma0_probe.py --seeds 42 43 44 45 46
"""

import os
import sys
import argparse

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJ)

import config as cfg              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Bandit gamma=0 sensitivity probe")
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=[42, 43, 44, 45, 46])
    a = ap.parse_args()

    # ── Force γ=0 everywhere (immediate-reward / true contextual bandit) ──
    cfg.GAMMA_BASE = 0.0
    cfg.GAMMA_MAX = 0.0
    os.environ.setdefault("NEUROMOD_FORCE_CPU", "1")

    from experiments import run_experiment_1
    from evaluation import _paired_p, _mean_ci

    configs = ["Full Model", "Ablated NA", "Ablated DA", "Static Baseline"]
    print("=" * 70)
    print("  BANDIT gamma=0 SENSITIVITY - does NA still drive adaptation?")
    print(f"  seeds={a.seeds}  (gamma_base=gamma_max=0)")
    print("=" * 70)

    lat = {c: [] for c in configs}
    for s in a.seeds:
        for c in configs:
            r = run_experiment_1(ablation_cfg=cfg.ABLATION_CONFIGS[c], seed=s,
                                 label=c)
            lat[c].append(float(r["adaptation_latency"]))

    fm, _, _ = _mean_ci(lat["Full Model"])
    print("-" * 70)
    # ASCII only — Windows cp1252 console can't encode arrows.
    print(f"{'config':<18} {'latency(lo=better)':>18}   "
          f"(paired vs Full: p_t / p_wilcox)")
    for c in configs:
        m, _, _ = _mean_ci(lat[c])
        if c == "Full Model":
            print(f"{c:<18} {m:>12.1f}")
        else:
            _, p_t, p_w = _paired_p(lat["Full Model"], lat[c])
            better = "Full faster" if fm < m else "Full slower"
            print(f"{c:<18} {m:>12.1f}   p_t={p_t:.4f} p_w={p_w:.4f}  ({better})")
    print("-" * 70)
    print("  Expected: Full << Ablated-NA (NA still drives adaptation), i.e. the")
    print("  headline NA result is NOT an artefact of the gamma=0.99 treatment.")
    print("=" * 70)


if __name__ == "__main__":
    main()

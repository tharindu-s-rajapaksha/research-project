"""
tools/da_strong_probe.py — EXPLORATORY / FUTURE WORK (not a main claim).

The audit found the DA differentiable-plasticity pillar inert-to-harmful in the
committed results: `PLASTIC_ALPHA_INIT = 0.002` makes the fast weights a ~few-%
perturbation, deliberately tiny for stability. This probe tests the reframe's
open question — "can DA be made individually necessary if the fast weights are
made stronger AND the task is hard enough to need associative memory?" — by:

  • raising the plastic coefficient init (default 0.03, ~15× the main config), and
  • running the *hard-regime* contextual capstone (4 cues + a large replay buffer),
    the regime RESEARCH_NOTES §6.3 identifies as overwhelming the base learner,

then comparing Full vs Ablated-DA (all else equal) across seeds on reward,
accuracy, re-adaptation latency and deaths, with paired t + Wilcoxon tests.

HONEST FRAMING: report whichever way it comes out. If DA becomes individually
necessary here it is a *conditional* result (needs a strong gate + a hard
associative task), NOT a headline; if it does not, DA stays a documented
neutral/negative and this is future work. Runs SERIALLY (it monkeypatches the
hard-regime constants, which would not propagate to spawned workers).

Usage:
    python tools/da_strong_probe.py --seeds 42 43 44 --alpha 0.03 --cues 4 --replay 5000
"""

import os
import sys
import argparse

import numpy as np

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJ)

import config as cfg              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="DA-strong exploratory probe")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--alpha", type=float, default=0.03,
                    help="plastic-coefficient init (main config = 0.002)")
    ap.add_argument("--cues", type=int, default=4,
                    help="hard-regime #cues (main capstone = 3)")
    ap.add_argument("--replay", type=int, default=5000,
                    help="hard-regime replay size (main capstone = 600)")
    a = ap.parse_args()

    # ── Install the HARD regime (monkeypatch cfg BEFORE building any env) ──
    cfg.VRF_N_CUES = a.cues
    cfg.VRF_N_SAFE_ARMS = a.cues
    cfg.VRF_REPLAY_SIZE = a.replay
    os.environ.setdefault("NEUROMOD_FORCE_CPU", "1")

    from experiments import run_experiment_3       # import AFTER the monkeypatch
    from evaluation import _paired_p, _mean_ci

    FULL = cfg.ABLATION_CONFIGS["Full Model"]
    ABL_DA = cfg.ABLATION_CONFIGS["Ablated DA"]

    print("=" * 74)
    print("  DA-STRONG PROBE (exploratory) — hard-regime contextual capstone")
    print(f"  plastic_alpha={a.alpha}  cues={a.cues}  replay={a.replay}  "
          f"seeds={a.seeds}")
    print("=" * 74)

    metrics = ("total_reward", "accuracy", "adaptation_latency", "death_count")
    full = {m: [] for m in metrics}
    abl = {m: [] for m in metrics}
    for s in a.seeds:
        rf = run_experiment_3(ablation_cfg=FULL, seed=s, label="Full",
                              plastic_alpha=a.alpha)
        ra = run_experiment_3(ablation_cfg=ABL_DA, seed=s, label="AblDA",
                              plastic_alpha=a.alpha)
        for m in metrics:
            full[m].append(float(rf[m]))
            abl[m].append(float(ra[m]))
        print(f"  seed {s}: Full reward={rf['total_reward']:.0f} "
              f"acc={rf['accuracy']:.2f} | AblDA reward={ra['total_reward']:.0f} "
              f"acc={ra['accuracy']:.2f}")

    # ── Paired Full-vs-AblatedDA tests (is DA individually necessary here?) ──
    lower_better = {"adaptation_latency", "death_count"}
    print("-" * 74)
    print(f"{'metric':<20} {'Full':>12} {'AblatedDA':>12} {'p_t':>9} "
          f"{'p_wilcox':>9}  DA-helps?")
    for m in metrics:
        fm, _, _ = _mean_ci(full[m])
        am, _, _ = _mean_ci(abl[m])
        _, p_t, p_w = _paired_p(full[m], abl[m])
        helps = (fm < am) if m in lower_better else (fm > am)
        tag = "yes" if (helps and p_w < 0.05) else ("dir" if helps else "no")
        print(f"{m:<20} {fm:>12.2f} {am:>12.2f} {p_t:>9.4f} {p_w:>9.4f}  {tag}")
    print("-" * 74)
    print("  'yes' = Full better AND Wilcoxon p<0.05 (DA individually necessary "
          "here);\n  'dir' = right direction, not significant;  'no' = DA does "
          "not help.\n  NOTE: exploratory — a positive result is CONDITIONAL "
          "(strong gate + hard task),\n  not a headline claim; a null result "
          "keeps DA a documented negative.")
    print("=" * 74)


if __name__ == "__main__":
    main()

# Research Notes — Multi-Neuromodulated Modular RL
*Audit, redesign, methodology, and results — a working record for the dissertation.*

> Purpose: a single reference you can lift directly into the thesis (Methodology,
> Results, Discussion, Threats to Validity, Future Work). Numbers are from the
> corrected, multi-seed (n=10) runs. Where a result is still pending, it says so.

---

## 0. Executive summary

An audit of the original implementation found that its headline results were
**artifacts of four defects** (a broken discount factor, a metric that measured the
wrong action, a confounded baseline, and single-seed pseudoreplicated statistics).
After fixing these (Phase 1), the corrected results **did not support** the
hypothesis — the original neuromodulator mechanisms were largely inert or harmful.
The mechanisms were then **redesigned** (Phase 2) so each actually influences
behaviour in the intended direction, and re-tested across 10 seeds with paired
statistics.

Final standing (10 seeds, all experiments complete):
- **Experiment 1 — the Full model significantly beats standard RL.** It adapts to
  reward switches ~19% faster than Vanilla DQN (latency 207 vs 254, p=0.0014) and
  ~21% faster than the matched Static baseline (p=0.0016), with higher reward
  (p≈0.002). On the bandit, **DA/plasticity is the main contributor** (removing it
  hurts, p=0.04); NA contributes directionally but is not cleanly isolated at n=10.
- **Experiment 2 — 5-HT harm aversion: strongly confirmed.** Removing 5-HT raises
  deaths **≈5.2×** (89 → 468) and flips cumulative reward from **+13,839 to −20,619**
  (paired t on deaths t≈−47, p<10⁻⁶). Necessary AND sufficient for survival. Strongest
  result. (Framing correction: the risky arm is *negative-EV* (−5 < +5), so 5-HT
  reaches the **true reward optimum** a plain DQN misses — it does not sacrifice
  reward for safety; see §6.1.)
- **Experiment 3 — DA/plasticity FAILS on continuous control (a clean negative).**
  DA-gated plasticity *impairs* CartPole: only **2/10** DA-on seeds reach competence
  vs **8/10** with DA off, and Full's reward is significantly *lower* than Static
  (p=0.035) and Vanilla (p=0.006). Recovery is inconclusive (too few competent Full seeds).

**One-line takeaway:** the two mechanisms that act on the *policy* (NA→ε-exploration,
5-HT→behavioural inhibition) beat standard RL; the mechanism that acts on the *network
weights* (DA→Hebbian plasticity) helps discrete fast re-locking (bandit) but hurts
stable continuous control (CartPole). A nuanced, defensible mixed result.

---

## 1. Corrected experimental methodology (for the Methodology chapter)

**Architecture under test.** A dual-module system: a **Hormonal Meta-Agent**
(the "Chemical Engine") tracks three digital neuromodulators (DA, NA, 5-HT) with
homeostatic spike-and-decay dynamics and opponent processing, and emits a
modulation vector each timestep. A **Local RL Worker** (DQN with a differentiable
Hebbian-plastic body) consumes that vector, adjusting learning rate α, exploration
ε, discount γ, a 5-HT loss-aversion gain, and a DA plasticity gate.

**Fair single-variable ablation.** Every configuration runs on the **identical**
`LocalRLWorker`; the only thing that changes is which hormone is enabled. This is
the key methodological fix — differences are now attributable to the neuromodulator,
not to a confounded change of architecture.

| Config | DA | NA | 5-HT | Role |
|---|---|---|---|---|
| Full Model | ✓ | ✓ | ✓ | all mechanisms |
| Ablated DA | ✗ | ✓ | ✓ | isolates dopamine/plasticity |
| Ablated NA | ✓ | ✗ | ✓ | isolates adaptation |
| Ablated 5-HT | ✓ | ✓ | ✗ | isolates harm aversion |
| Static Baseline | ✗ | ✗ | ✗ | **same architecture, hormones frozen at rest** — the "static agent" the hypothesis claims to beat |
| Vanilla DQN | — | — | — | plain-MLP ε-greedy DQN, external sanity anchor (not a clean ablation) |

**"Reduces to baseline at rest" invariant.** By construction every modulation maps
to its base value when its hormone is at the resting concentration, so *Static
Baseline is exactly the Full Model with hormones held constant.* This is what makes
the ablation clean, and is verified by a unit check (α=α_base, ε=ε_base, γ=γ_base,
gain=1, plastic-gate=0 at rest).

**Statistics.** Each (experiment × config) is run on **10 shared seeds**
(`SEEDS = 42…51`). For every headline metric we report **mean ± 95% CI**
(t-based, `evaluation.summarize_multiseed`). Significance uses a **paired t-test**
(Full vs each config, seeds matched; `evaluation.compute_multiseed_pvalues`),
dropping seeds where a metric is undefined (e.g. an ungated CartPole recovery).
Runs are parallelised across CPU processes (`--workers N`); results are reassembled
in sorted-seed order so pairing stays aligned.

**Metrics.**
- *Exp 1 (bandit):* **Adaptation latency** = mean over switches of the steps to
  re-lock (5 consecutive pulls) onto the arm that is optimal *in the new phase*;
  plus cumulative reward.
- *Exp 2 (foraging):* **Death count**, **Survival rate** (mean steps between deaths),
  cumulative reward, over 5,000 steps.
- *Exp 3 (CartPole):* **Recovery time** = episodes after the physics shock to sustain
  ≥300 steps for 3 consecutive episodes — **defined only if the agent was competent
  pre-shock** (rolling mean ≥350 over the last 20 pre-shock episodes); NaN otherwise.

---

## 2. Phase 1 — Audit findings (what was wrong originally)

Grouped by how badly each threatened a claim. File references are to the current
tree (line numbers may have since shifted).

| # | Severity | Where | Defect | Why it invalidated a result |
|---|---|---|---|---|
| C1 | 🔴 Critical | `main.py`, `evaluation.py` | **Single seed; pseudoreplicated p-values.** Stats were computed by chopping *one* seed's reward stream into 50-step windows and t-testing across them. | Autocorrelated windows are not independent replicates; every p-value was meaningless. The hypothesis requires significance *across seeds*. |
| C2 | 🔴 Critical | `experiments.py` (Exp 1 latency) | **Adaptation latency counted the wrong arm.** It looked only after the *last* switch and counted pulls of arm 4, but the optimal-arm sequence is `[0,4,1,3,2]`, so after the last switch the optimum is arm 2. | The condition never fired → latency pinned to the worst-case cap for every config → the entire NA/adaptation result was noise (Ablated-NA even looked *better*). |
| C3 | 🔴 Critical | `meta_agent._modulate_discount` | **γ collapsed at rest.** `γ = γ_base·σ(5HT−baseline)` gives **γ ≈ 0.495 at rest** vs the baseline's fixed 0.99 (visible as `Gamma = 0.4950` on the interim slide). | Halved the effective horizon of the "dynamic" agent on every step → CartPole (return = horizon) was crippled; the Full model performed at *random* level while plain DQN solved it. Both a bug and an unfair confound. |
| C4 | 🔴 Critical | `worker.py` | **Confounded baseline.** The "Static Baseline" differed from the Full model in ≥4 ways at once (plain MLP vs plastic, ε-greedy vs softmax, no modulation, γ=0.99 vs 0.495). | No Full-vs-Static difference could be attributed to neuromodulation. |
| C5 | 🔴 Critical | `experiments.py` (Exp 3) | **Undefined recovery.** Recovery was measured even when the agent never reached competence pre-shock. | With the crippled agent never learning CartPole, "recovery" measured nothing and was silently capped. |
| H1 | 🟠 High | `config.py` | **No Ablated-DA config** → dopamine's causal contribution was never isolated. |
| H2 | 🟠 High | spec/docstrings | **τ formula self-contradictory:** text said `τ = τ_base/NA` (which *reduces* exploration on volatility, defeating NA), code did the opposite. |
| H3 | 🟠 High | `neuromodulators` | **Opponent processing fights the α-boost** after harm; DA_eff permanently halved at rest (undocumented). |
| H4 | 🟠 High | `worker.py`/`plasticity.py` | **Frozen target net still mutated its Hebbian trace** every forward, so bootstrap targets drifted. |
| H5 | 🟠 High | `evaluation.py` | **Significance tested only on summed reward**, never on the actually-claimed metrics (latency/deaths/recovery). |

---

## 3. Phase 1 — Fixes applied (all validity fixes, independent of outcome)

- **C3:** re-centred discount to `γ = γ_base + (γ_max−γ_base)·excess(5HT)` → rest = γ_base,
  5-HT only *lengthens* horizon (never shortens it). Every modulation now shares the
  `excess()` helper so it reduces to base at rest.
- **C4:** all configs use the same `LocalRLWorker`; "Static Baseline" = that worker
  with hormones frozen. Plain DQN retained as a clearly-labelled "Vanilla DQN" anchor.
- **C2:** latency measured **per switch against the correct new-phase arm**, reported
  as the mean plus the per-switch breakdown.
- **C5:** recovery **gated on pre-shock competence** (NaN and flagged otherwise);
  perturbation constants aligned to one story (gravity 29.4 = 3×, force ×0.5).
- **C1/H5:** multi-seed harness + **paired per-seed tests + 95% CIs on every metric**;
  window-chunking test deleted.
- **H1:** added `Ablated DA`.
- **H2:** τ formula corrected in docs (and later superseded by ε — §5).
- **H4:** `update_trace=False` for the target-net forward.
- Cleanup: dead `_prev_volatility`, "LunarLander" comment, stale constants.

**What Phase 1 revealed:** with the bugs removed, the *original* mechanisms did not
beat standard RL, and the previously-reported "5-HT survival win" **disappeared** —
it had been an artifact of C3 (serotonin was only rescuing the agent from the broken
resting γ). This motivated Phase 2.

---

## 4. Why the original mechanisms could not work (Discussion material)

- **Exploration via Boltzmann/softmax temperature** is scale-sensitive: when action
  Q-values are near-equal (CartPole), any moderate temperature is near-uniform, so
  the agent explored far more than ε-greedy and never converged.
- **5-HT never entered the action values.** It only touched γ and suppressed DA_eff
  (which affects plasticity/α). Nothing made the *policy* risk-sensitive, so the
  original model could not avoid the risky action once γ was correct. (NB: the
  original spec claimed risky EV = 0.9×50 = **+45** > safe +5 — this omits the −500
  death, which the DQN *does* receive as a per-step reward. The true per-step EV of
  risky is 0.9(50)+0.1(−500) = **−5 < +5**, i.e. risky is *negative-EV*; the correct
  framing is in §6.1.)
- **DA-plasticity applied at full strength always.** The Hebbian fast-weights were
  active even at rest, injecting noise into stable-control learning.

---

## 5. Phase 2 — Mechanism redesign (the current model; for Methodology + Discussion)

Hormone **sensing** (spike/decay, opponent processing) is unchanged from the spec;
the redesign is in how each hormone **acts on behaviour**. Shared helper:
`excess(c) = clip(2·(σ(c−B) − 0.5), 0, 1)` — 0 at/below the resting baseline B=1,
rising to 1 as the concentration saturates.

**Noradrenaline → exploration rate ε (was softmax τ).**
`ε_t = ε_base + (ε_max − ε_base)·excess(NA)`, with `ε_base=0.1, ε_max=0.5`.
ε-greedy is scale-invariant, so it works where Q-gaps are tiny. The **NA detector is
redesigned to be a directional, one-sided mean-shift detector on the external reward
signal** (fires only when recent reward drops significantly below the established
baseline; `z = max(0, mean_baseline − mean_recent)/std`, threshold 2.0). It is
**deliberately not** keyed on TD-error, because TD-error is large throughout ordinary
learning and would flood control tasks with spurious exploration. (Yu & Dayan 2005 —
unexpected uncertainty.)

**Serotonin → harm aversion, via two policy-level pathways.**
Both scale with `excess(5HT)` and vanish at rest.
1. *Punishment-sensitive learning:* the per-sample Huber loss of **negative-reward**
   transitions is up-weighted by a gain `g_t = 1 + (G−1)·excess(5HT)`, `G=4`. (We
   re-weight the loss rather than scaling the reward, because Huber saturates its
   gradient for large errors, so reward-scaling would be clipped away.) Harmful
   actions therefore lose value faster.
2. *Behavioural inhibition:* a per-action harm estimate `harm[a]` (EMA, decay 0.99,
   of `max(0, −reward)`) is maintained; at action selection, when 5-HT is elevated,
   `Q[a] ← Q[a] − (g_t−1)·w·harm[a]`. This actively **withholds** actions with a
   harmful history — breaking the trap where the agent stays hooked on a high-EV
   lethal action and never samples the safe one. (Daw 2002; Cools 2011; Crockett 2009
   — serotonergic behavioural inhibition / punishment sensitivity.)

**Dopamine → gated differentiable plasticity (was always-on).**
The Hebbian third factor is now `plastic_gate = clip(2·|DA_eff − DA_eff_rest|, 0, 1)`,
**0 at rest** (network behaves as a standard DQN) and rising as DA_eff deviates in
either direction (reward surprise or performance collapse). The plastic coefficient
init was lowered (`PLASTIC_ALPHA_INIT=0.002`) and `NeuromodulatedLinear` now uses the
**exact `nn.Linear` init**, so the gate-off plastic net is distributionally identical
to a plain layer (this alone recovered CartPole competence for the frozen baseline).
The learning-rate boost ceiling was tightened (`α_max = 2×α_base`, was 5×) because
large DA-driven α spikes destabilised value learning.

---

## 6. Results so far

### 6.1 Experiment 2 — High-Stakes Foraging (5-HT), n=10 ✅ headline result

Cumulative-reward and survival over 5,000 steps (mean ± 95% CI):

| Config | Deaths ↓ | Survival ↑ | Total Reward ↑ |
|---|---|---|---|
| **Full Model** | **89.2 ± 2.1** | **55.8 ± 1.4** | **+13,839 ± 3,316** |
| Ablated DA | 81.4 ± 2.5 | 60.9 ± 1.9 | +14,547 ± 3,318 |
| Ablated NA | 89.4 ± 3.0 | 55.7 ± 1.9 | +13,760 ± 3,437 |
| **Ablated 5-HT** | **467.8 ± 17.3** | **10.7 ± 0.4** | **−20,619 ± 8,658** |
| Static Baseline | 466.9 ± 18.1 | 10.7 ± 0.4 | −20,633 ± 8,843 |
| Vanilla DQN | 466.7 ± 17.5 | 10.7 ± 0.4 | −20,640 ± 8,639 |

Paired tests (Full vs …):
- **vs Ablated 5-HT:** deaths t=−47.4 (p<10⁻⁶), survival t=65.0 (p<10⁻⁶), reward t=11.9 (p<10⁻⁶). **Full is vastly better.**
- vs Static and vs Vanilla: same picture (deaths t≈−45/−47, survival t≈66).
- vs Ablated NA: **no difference** (p>0.77) — NA is correctly irrelevant to survival.
- vs Ablated DA: Ablated-DA is *slightly* better (deaths 81 vs 89, p<10⁻⁶) — DA-plasticity mildly **hurts** survival (extra exploration noise).

**Interpretation (thesis-ready):** Serotonin is **necessary and sufficient** for
survival here — but the framing matters. The risky action is **negative expected
value**: per-step EV = 0.9(50) + 0.1(−500) = **−5**, *below* the safe +5, so the
reward-maximising policy is already to avoid it. 5-HT therefore does **not** trade
reward for safety; it lets the agent reach the **true reward optimum** that a plain
DQN fails to find. The plain DQN fails for value-estimation reasons: (i) the
Huber/`smooth_l1` loss clips the gradient of the rare −500, so the catastrophe is
under-weighted; (ii) the small replay buffer (500) under-samples the 10% death;
(iii) ε-greedy keeps resampling the risky arm. 5-HT's two pathways (punishment
up-weighting + behavioural inhibition) counter exactly these failures. Concretely:
the optimal always-safe policy scores ≈**+25,000**; **Full = +13,839** (dies ≈89× —
it *reduces*, not eliminates, the failure); every config **without** 5-HT
(Ablated-5HT, Static, Vanilla) collapses to ≈**−20,600** and ≈467 deaths. Every
config *with* 5-HT succeeds, cleanly isolating the serotonergic
behavioural-inhibition mechanism as the causal driver — as hypothesised, though for
the corrected (value-estimation-failure) reason rather than "caution overriding
higher-reward greed". **The original "mathematical trap" arithmetic (risky EV +45 >
safe +5) was wrong; state the −5 EV in the dissertation.**

### 6.2 Experiment 1 — Volatile Bandit (NA / adaptation), n=10 ✅ beats standard RL

Adaptation latency (steps to re-lock, lower better) and cumulative reward (mean ± 95% CI):

| Config | Latency ↓ | Total Reward ↑ |
|---|---|---|
| **Full Model** | **206.9 ± 28.7** | **30,341 ± 575** |
| Ablated DA | 245.9 ± 16.6 | 29,340 ± 486 |
| Ablated NA | 227.4 ± 17.4 | 29,787 ± 526 |
| Ablated 5-HT | 200.0 ± 21.6 | 30,154 ± 634 |
| Static Baseline | 262.3 ± 10.8 | 29,081 ± 238 |
| Vanilla DQN | 254.3 ± 13.7 | 29,390 ± 353 |

Paired tests (Full vs …):
- **vs Vanilla DQN:** latency t=−4.57 (p=0.0014), reward t=4.46 (p=0.0016) — **Full significantly better** (~19% faster adaptation).
- **vs Static Baseline:** latency t=−4.47 (p=0.0016), reward t=4.06 (p=0.003) — **Full significantly better** (~21% faster).
- vs Ablated DA: latency t=−2.41 (p=0.039), reward t=3.01 (p=0.015) — **DA/plasticity helps** adaptation here.
- vs Ablated NA: Full better directionally (207 vs 227) but **not significant** (p=0.23).
- vs Ablated 5-HT: no difference (5-HT correctly irrelevant to the bandit).

**Interpretation:** the full neuromodulated agent adapts to distribution switches
significantly faster and earns more than a standard DQN — the core "beats standard RL"
claim, supported with paired statistics. Notably, on this discrete fast-switching task
the **dopaminergic fast-weights** contribute more than noradrenergic exploration.

### 6.3 Experiment 3 — CartPole (DA / plasticity), n=10 ❌ negative (clean)

The diagnostic is the **pre-shock competence rate** — of 10 seeds, how many learned
CartPole (rolling ≥350) before the perturbation (this is `N_seeds` on Recovery_Time):

| Config | DA on? | Competent seeds | Total Reward |
|---|---|---|---|
| Full Model | yes | **2 / 10** | 142,361 ± 10,078 |
| Ablated NA | yes | 3 / 10 | 140,658 ± 17,068 |
| Ablated 5-HT | yes | 1 / 10 | 141,433 ± 11,311 |
| **Ablated DA** | no | **8 / 10** | 169,574 ± 16,646 |
| Static Baseline | no | **8 / 10** | 161,498 ± 11,203 |
| Vanilla DQN | — | 4 / 10 | 162,058 ± 10,286 |

Paired reward tests (Full vs …): worse than **Static** (t=−2.48, p=0.035), **Vanilla**
(t=−3.62, p=0.006), and **Ablated DA** (t=−3.04, p=0.014); indistinguishable from the
other DA-on configs.

**Interpretation:** DA-gated plasticity **impairs** stable continuous control. Configs
with dopamine active reach competence in only 1–3/10 seeds vs 8/10 with DA off, and the
Full model earns significantly less reward than the static baseline. **Recovery time is
inconclusive** — the Full model reaches competence too rarely (2 seeds) to measure a
recovery advantage; among the few competent seeds it is not distinguishable from Static.
The intended dopaminergic "fast re-adaptation" benefit is therefore **not demonstrated**
on continuous control; the fast-weights that help discrete re-locking (Exp 1) hurt here.

---

## 7. Full parameter reference (`config.py`)

| Symbol | Value | Meaning |
|---|---|---|
| B (baseline) | 1.0 | resting hormone concentration |
| k_DA / k_NA / k_5HT | 0.1 / 0.08 / 0.03 | decay rates (5-HT slowest → longest "mood") |
| DA/NA/5HT spike scale | 1 / 2 / 3 | spike gains |
| VOLATILITY_WINDOW / THRESHOLD | 200 / 2.0 | NA reward-drop detector window / z-threshold |
| RISK_PENALTY_THRESHOLD | −50 | reward below this = aversive (5-HT spike) |
| α_base / α_max scale | 1e-3 / 2× | learning rate & DA boost ceiling |
| ε_base / ε_max | 0.1 / 0.5 | ε-greedy rate at rest / NA-saturated |
| γ_base / γ_max | 0.99 / 0.999 | discount at rest / 5-HT-saturated |
| HT_PUNISHMENT_GAIN (G) | 4.0 | max loss up-weight on losses |
| RISK_INHIBITION_WEIGHT / HARM_EMA_DECAY | 1.0 / 0.99 | 5-HT behavioural-inhibition penalty |
| PLASTIC_ALPHA_INIT / η_decay / η_trace | 0.002 / 0.05 / 0.01 | plasticity strength & trace dynamics |
| HIDDEN_DIM / REPLAY / BATCH / TARGET_SYNC | 128 / 500 / 64 / 100 | DQN hyperparameters |
| SEEDS | 42…51 (10) | statistical replicates |
| Exp1 | 5 arms, 4,000 steps, switches [500,1100,1800,3000], μ_hi/lo=10/2 | |
| Exp2 | 5,000 steps, safe +5, risky +50 / 10% death −500 | |
| Exp3 | 300+300 episodes, gravity 29.4 (3×), force ×0.5, competence 350 / recovery 300 | |

---

## 8. Assumptions & design decisions (state and defend these)

1. **Hormones are hand-designed control laws, not learned.** The mapping from
   signals to α/ε/γ/gain/gate is fixed (bio-inspired), not meta-learned. This is a
   deliberate scope choice; a learned meta-controller is future work.
2. **"Static baseline" = same architecture, frozen hormones.** The central
   comparison. Vanilla DQN is a secondary sanity anchor only.
3. **One-step hormonal lag is intentional.** A step's learning uses the hormonal
   context from before that step's outcome is known — you cannot modulate on a
   TD-error before it exists. Biologically plausible; not a bug.
4. **NA keys on reward, DA on TD-error.** Split by design: reward-drop = environmental
   worsening (NA→explore); TD-surprise = model error (DA→plasticity/α). On CartPole
   (constant +1 reward) NA is intentionally quiet and DA carries adaptation.
5. **Opponent processing halves DA_eff at rest** (σ(0)=0.5). Retained from the spec;
   internally consistent (all rest-references use 0.5). Flagged as a design choice to
   discuss, not a bug.
6. **Death in Exp 2 is terminal-like** (bootstrap cut via done-flag) but the episode
   continues with cumulative score reset — models "loss of progress", not end of run.
7. **Competence/recovery thresholds** (350/300 of 500) are judgement calls applied
   identically to all configs; 70%/60% of max is "learned the task", not near-perfect.

---

## 9. Threats to validity & limitations (Threats-to-Validity section)

- **n=10 seeds.** Ample for the large effects (5-HT survival; Exp-1 Full-vs-baseline).
  **The NA contribution is under-powered** — removing NA raises latency 207→227 but
  p=0.23, so NA is not *individually* isolated at n=10 even though the Full model wins.
  A larger n (or an NA-specific stress task) would be needed to claim NA causally.
  Report CIs, not just p.
- **Exp-3 recovery is under-powered by construction.** Because DA-plasticity suppresses
  competence, only 2/10 Full seeds qualify for a recovery measurement, so the recovery
  comparison is inconclusive rather than a clean "no faster recovery". State it as
  "could not be assessed", and lead the Exp-3 story with the competence-rate and reward
  results, which are clear.
- **Hand-tuned thresholds** (competence, ε_max, punishment gain, harm weight) were set
  by pilot inspection, not swept — a sensitivity analysis would strengthen the claims.
- **DA-plasticity is at best neutral, sometimes harmful** on the tasks tested; its
  intended benefit (rapid intra-lifetime re-adaptation) is not yet demonstrated to
  beat the frozen baseline. Be honest about this.
- **Toy environments.** Bandit / two-choice foraging / CartPole are deliberately
  minimal to isolate each hormone; generalisation to richer domains is untested.
- **Vanilla DQN is not a clean ablation** (differs in body + selection); use it only
  as an external anchor, not for causal claims.

---

## 10. Suggested narrative & future work (Discussion / Conclusion)

**Honest headline the results support:** *"The full multi-neuromodulated agent
significantly outperforms a standard DQN on two of three tasks. On a volatile bandit
it adapts ~19% faster to reward switches (p=0.001); on high-stakes foraging a
serotonergic behavioural-inhibition mechanism produces robust, statistically
overwhelming harm aversion (≈5× fewer catastrophic failures, p<10⁻⁶), cleanly isolated
from the other neuromodulators. However, the dopaminergic gated-plasticity mechanism
is task-dependent: it aids rapid re-locking on the discrete bandit but IMPAIRS stable
continuous control (CartPole), where it is significantly worse than the matched static
baseline and fails to demonstrate the intended faster recovery."* This is a credible,
defensible mixed result — the split between policy-level modulation (works) and
weight-level plasticity (task-dependent) is the intellectual core of the discussion.

**Two "beats standard RL" wins with proper statistics** (Exp 1 adaptation, Exp 2
survival, both vs Vanilla DQN and vs the matched Static baseline) plus **one clean
negative** (Exp 3 plasticity) is a stronger, more honest contribution than a uniform
"it works" — and it directly answers the research question mechanism-by-mechanism.

**Framing the method contribution:** the value is as much the **evaluation protocol**
(fair same-architecture ablation, reduces-to-baseline invariant, per-seed paired
statistics, competence-gated recovery) as the mechanisms — it is what let you tell a
real effect (5-HT) from three bug-induced mirages.

**Future work:** (i) a learned/meta-optimised meta-controller instead of hand-designed
laws; (ii) a DA-plasticity gate that distinguishes *environmental change* from
*ordinary learning error* so plasticity helps recovery without hurting acquisition;
(iii) sensitivity sweeps over the hand-set thresholds; (iv) richer, higher-dimensional
non-stationary environments; (v) more seeds for the marginal NA effects.

---

## 11. Reproducibility (Appendix)

```
# Full study (all experiments, 10 seeds, 12 CPU workers)
python main.py --workers 12

# Per experiment (writes summary_multiseed_expN.csv, pvalues_expN.csv)
python main.py --exp 1 --workers 12   # Bandit (NA)      — fast
python main.py --exp 2 --workers 12   # Foraging (5-HT)  — fast
python main.py --exp 3 --workers 12   # CartPole (DA)    — slow (~bulk of runtime)
```
Workers run on CPU by default (these tiny nets are ~2× faster per-run on CPU than
GPU; measured 25.6s vs 47.3s for one bandit run). Outputs land in `research_results/`:
per-config dashboards, comparative bar charts (mean ± 95% CI), regret curves, and the
summary/p-value CSVs. Key modules: `neuromodulators.py` (sensing), `meta_agent.py`
(modulation laws), `plasticity.py` (`NeuromodulatedLinear`), `worker.py` (DQN + ε-greedy
+ 5-HT pathways), `experiments.py` (protocols/metrics), `evaluation.py` (stats/plots),
`ablation.py` (multi-seed runner).
```
```

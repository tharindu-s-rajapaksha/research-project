# Research Notes — Multi-Neuromodulated Modular RL
*Audit, redesign, methodology, and results — a working record for the dissertation.*

> Purpose: a single reference you can lift directly into the thesis (Methodology,
> Results, Discussion, Threats to Validity, Future Work). Numbers are from the
> corrected, multi-seed (n=51) runs. Where a result is still pending, it says so.
>
> ⚠️ **Single source of truth: `research_results/RESULTS_TABLES.md`** (auto-generated
> from the committed CSVs by `python tools/make_tables.py`, N=51). If any inline number
> below disagrees with that file, **the CSV/RESULTS_TABLES.md wins** — re-sync, do not
> hand-transcribe. The §6 tables were regenerated to the 51-seed run on 2026-07-19.

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

> **All quoted numbers are auto-generated from the committed CSVs by
> `python tools/make_tables.py` → `research_results/RESULTS_TABLES.md`. Never
> hand-transcribe: edit the study, re-run, regenerate.** The tables below are
> refreshed from the current `summary_multiseed.csv` / `pvalues.csv`.

Final standing (51 seeds, all experiments complete):
- **Experiment 1 — the Full model significantly beats standard RL, and NA is the
  cleanly-isolated driver.** It re-locks after a reward switch in **174 vs 382 steps**
  for the matched Static/Vanilla baseline (~54% faster; latency Holm *p*≈0,
  reward Holm *p*≈0). The ablation isolates **noradrenaline**: removing NA nearly
  doubles latency (174 → 348, Holm *p*≈0), while removing DA changes nothing
  (174 → 178, *p*=0.40) and removing 5-HT is bit-identical (5-HT is inert on an
  all-positive-reward bandit). ⚠️ **This corrects the earlier draft, which claimed
  DA/plasticity was the main contributor and NA was n.s. — the committed CSVs show the
  exact opposite. NA is the driver; DA is inert here.** (See §6.2.)
- **Experiment 2 — 5-HT harm aversion vs a *naive* DQN: strongly confirmed; vs a
  *value-corrected* DQN: NOT necessary (honest caveat, §6.6).** Against the matched
  Huber-loss baseline, removing 5-HT raises deaths **≈13×** (36 → 478) and flips
  cumulative reward **+20,886 → −22,823** (paired t on deaths t≈−75, *p*<10⁻⁶) —
  necessary and sufficient. **BUT** the fair-baseline study (§6.6) shows this win is
  largely a *Huber-loss artefact*: the standard DQN dies because Huber clips the
  gradient of the rare −500, so it under-weights the catastrophe; a value-corrected
  DQN (MSE, or reward-scaled) **survives Exp 2 without any serotonin**. The honest
  claim is therefore *"5-HT fixes a value-estimation failure of a Huber-loss DQN,"*
  not *"5-HT is required to avoid the death trap."* (Framing: the risky arm is
  *negative-EV*, −5 < +5, so the reward-optimal policy is already all-safe; see §6.1.)
- **Experiment 3 — Risky Foraging CAPSTONE (replaces CartPole).** An integrative task
  fusing volatility with lethal risk. Two iterations (see §6.3, §12): a **stateless**
  version (older result) and a **contextual reversal-learning** upgrade
  (current, committed). Both give the same headline: **Full significantly beats standard
  RL** (contextual, 51 seeds, vs Static & Vanilla: reward +29,447 vs −23,506, Holm p≈0),
  and **5-HT is robustly necessary and sufficient** (removing it → catastrophe, all p≈0).
  **On the committed contextual capstone, DA and NA are NOT null** (this updates the older
  10-seed reading): removing DA significantly *worsens* reward (29.4k → 27.8k, Holm
  p=0.012), latency (841 → 943, p≈0) and accuracy (0.43 → 0.39, p≈0); removing NA
  significantly *worsens* latency (841 → 915, p≈0) but **also lowers deaths** (47 → 39,
  p≈0) because NA's exploration samples the lethal arm, so its **net reward** effect is nil
  (p=0.93). So 5-HT dominates, DA gives a small all-round benefit, and NA is a latency/death
  trade-off that washes out on reward. **Honest conclusion:** the capstone
  is an *integration win* + a genuine finding about **neuromodulator redundancy** (5-HT
  robustly necessary; NA/DA necessary only under high adaptation demand), not an
  "every-hormone-necessary-in-one-task" result. A robust **median/MAD NA detector** (a
  real fix from approach 2) is retained.
- **Generalist vs Specialists — the direct novelty test (§6.5): the integrated agent
  beats single-modulator gating.** Instead of *removing* one hormone from Full (ablation,
  which only isolates 5-HT), this pits Full against three **single-modulator specialists**
  (DA-only / NA-only / 5-HT-only) across **both** task niches — adaptation (Exp 1) and
  survival (Exp 2). Result (10 seeds, Holm-corrected): on **each** task the Full agent
  ties the best specialist and **significantly beats every specialist lacking the relevant
  hormone**, so **no single-modulator agent is competent on both** — DA/NA-only die in the
  survival task, 5-HT-only is >2× slower to adapt. Task-anchored worst-task "floor"
  competence: **Full 0.80 vs 0.56 (5-HT-only) and ≤0.06** for the arms that die. This is
  the robust, honest form of the interim novelty ("integration beats single-modulator
  gating"), distinct from prior serotonin-alone work.
- **(Legacy, secondary/negative) CartPole.** Retained via `run_experiment_cartpole`:
  DA-gated plasticity *impairs* stable continuous control (only 2/10 DA-on seeds reach
  competence vs 8/10 with DA off; Full reward significantly below Static/Vanilla). A
  useful contrast — plasticity helps discrete re-mapping but hurts continuous control.

**One-line takeaway:** the defensible contributions are (1) a **cleanly-isolated noradrenergic
adaptation win** (Exp 1: Full re-locks ~54% faster than a standard DQN — including the best
swept-ε/ε-decay DQN, §6.6 — attributed specifically to NA, Holm p≈0), and (2)
**generalist coverage** (§6.5: only the tri-hormone agent is competent across both an adaptation
and a survival niche; task-anchored worst-task floor 0.80 vs ≤0.56 for every specialist).
Serotonin's survival effect is real but **honestly bounded** (§6.6: a value-corrected DQN also
survives without it, so 5-HT *repairs a Huber-loss pathology* rather than being universally
necessary). Dopaminergic plasticity is a **documented negative** (neutral-to-harmful everywhere;
worst on continuous control). Framed this way — one clean mechanistic win, one coverage/novelty
result, one bounded win, one honest negative — every claim survives a viva and traces to a
committed CSV. (The earlier "capstone shows all three cooperating / 5-HT universally necessary"
framing is retired; see §6.1, §6.3, §6.6.)

> ⚠️ **Provenance of numbers.** §6 tables are refreshed from the committed
> `research_results/*.csv` via `tools/make_tables.py`. After any code change that
> affects results (e.g. the Survival_Rate trailing-streak fix, or the new fair
> baselines), **re-run the studies then regenerate the tables** — do not edit numbers
> by hand. Commands in §11. The fair-baseline battery (§6.6) has now been run at **51 seeds**
> (committed CSVs); the two sensitivity probes (`tools/da_strong_probe.py`,
> `tools/bandit_gamma0_probe.py`) are exploratory and run serially on demand.

---

## 0B. Phase 3 — mechanism tuning (2026-07-05)

After confirming the mechanisms fire correctly, a diagnostic pass found each was
working but under-tuned. Six changes (all in `config.py` unless noted), each validated
by measuring the specific behaviour it targets:

| Change | From → To | Why | Pilot effect (seeds 42-46) |
|---|---|---|---|
| `EPSILON_BASE` | 0.1 → **0.01** (final) | 0.1 forced-random floor capped bandit exploitation, pushed steps into the lethal arm, and its noise broke re-lock streaks (inflating latency); lowered further to 0.01 so NA's switch-time ε boost is load-bearing | Exp1 latency ↓, optimal-rate ↑; Exp2 fewer forced deaths |
| `VOLATILITY_THRESHOLD` | 2.0 → **3.5** | NA fired ~458×/run on ordinary ε-greedy reward noise instead of the ~4 genuine switches, so it lost selectivity | NA spikes 458→~180; **NA now HELPS** (Exp1 latency 200 vs 229 for Ablated-NA) |
| `HARM_EMA_DECAY` | 0.99 → **0.90** | per-action harm estimate took ~100 deaths to build but only ~90 occur, so behavioural inhibition never got strong | stronger, faster harm signal |
| `RISK_INHIBITION_WEIGHT` | 1.0 → **5.0** | inhibition penalty too small to overcome risky's frequent +50; greedy still chose it ~9% | Exp2 safe-rate 83→93%, deaths 92→43 |
| replay buffer | 500 → **per-exp** (E1 500 / E2 5000 / E3 10000) | bandit needs a small buffer (forget stale rewards), foraging/control need a large one (retain rare deaths) | Exp2 value fn retains death signal |
| Exp3 perturbation | grav ×3 + force ×0.5 → **grav ×2 + force ×1.0**; train 300→400 | ×3 grav + halved force was near-unsolvable, so recovery was unmeasurable | (CartPole re-run pending) |

**Correctness fixes bundled in the same pass:**
- **Restored `random.seed(seed)`** in all three experiment runners (`experiments.py::_seed_all`).
  It had been lost in a revert; the worker's ε-greedy and replay sampling use the *stdlib*
  `random`, so without it re-runs are non-reproducible **and the paired t-tests are invalid**
  (Full and each ablation would see different stochastic streams on a shared seed). This is a
  validity fix, not a tuning knob.
- **Re-added Holm–Bonferroni** correction (`evaluation.py::_holm`), applied within each
  experiment's family of comparisons → `p_holm`, `sig_holm_0.05` columns.
- **Chart clarity:** comparative bars now say "(lower is better)" and star/outline the best
  config (all three headline metrics are lower-is-better, so a tall bar = worse agent).

**Pilot improvements vs Phase 2 (5 seeds, to be confirmed at n=10):**
- **Exp 1:** Full now beats Ablated-NA on *both* latency (200 vs 229) and optimal-pull
  (71.3% vs 69.4%) — NA changed from neutral/harmful to genuinely helpful.
- **Exp 2:** deaths **89 → 43** (halved), reward **13.6k → 19.2k** (+41%), safe-rate
  **55% → 92.5%**; 5-HT effect now ≈11× (43 vs 484 deaths).
- **Exp 3:** re-run pending (softened perturbation + more episodes).

**Note on the bandit "optimal-pull %":** the *overall* rate sits ~71% because the metric
averages over 4 post-switch re-learning transients; *steady-state* per phase is **88–98%**.
The overall ceiling is inherent to a 5-phase volatile bandit, not a defect.

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

**Statistics.** Each (experiment × config) is run on **51 shared seeds**
(`SEEDS = 42…92`). For every headline metric we report **mean ± 95% CI**
(t-based, `evaluation.summarize_multiseed`). Significance uses **paired tests**
(Full vs each config, seeds matched; `evaluation.compute_multiseed_pvalues`):
a **paired t-test AND a Wilcoxon signed-rank** (`_paired_p`) — the latter is the
distribution-free test reported for the count/capped metrics (`Death_Count`,
`Adaptation_Latency`) where normality is shaky at small n. Both are **Holm–Bonferroni
corrected** within each experiment's family (columns `p_holm`, `p_wilcoxon_holm`),
dropping seeds where a metric is undefined (e.g. an ungated CartPole recovery). Runs
are parallelised across CPU processes (`--workers N`); results are reassembled in
sorted-seed order so pairing stays aligned. All quoted numbers are regenerated from
the CSVs by `tools/make_tables.py` (→ `RESULTS_TABLES.md`), never hand-transcribed.

**Baselines.** The central comparison is Full vs **Static Baseline** (same
architecture, hormones frozen). **Vanilla DQN** is a plain-MLP anchor that we verified
comes out **byte-identical** to Static Baseline (a validation of the reduces-to-baseline
invariant, not an independent data point). Because that shared baseline uses a
near-greedy ε=0.01 and a Huber loss, a separate **fair-baseline battery**
(`baselines.py`, §6.6) additionally pits Full against **well-tuned standard DQNs**
(swept-ε, ε-decay, MSE, reward-scaled), so "beats standard RL" means "beats *tuned*
standard RL", not a strawman.

**Metrics.**
- *Exp 1 (bandit):* **Adaptation latency** = mean over switches of the steps to
  re-lock (5 consecutive greedy pulls) onto the arm that is optimal *in the new phase*
  — note this captures re-lock *and* NA's exploration settling; plus cumulative reward.
- *Exp 2 (foraging):* **Death count**, **Survival rate** (mean steps between deaths,
  including the trailing streak), cumulative reward, over 5,000 steps.
- *Exp 3 (contextual capstone):* **Cumulative reward** (headline) + **accuracy**
  (fraction of steps taking the cue's correct action) + **death count** +
  **re-adaptation latency** (accuracy-based, per reversal), over 6,000 steps / 5 reversals.
- *Legacy (CartPole):* **Recovery time** = episodes after the physics shock to sustain
  ≥300 steps for 3 consecutive episodes — defined only if competent pre-shock (rolling
  ≥350 over the last 20 pre-shock episodes); NaN otherwise.

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
`ε_t = ε_base + (ε_max − ε_base)·excess(NA)`, with `ε_base=0.01, ε_max=0.5`.
ε-greedy is scale-invariant, so it works where Q-gaps are tiny. The **NA detector is
redesigned to be a directional, one-sided mean-shift detector on the external reward
signal** (fires only when recent reward drops significantly below the established
baseline; robust median/MAD form `z = max(0, med_baseline − med_recent)/(1.4826·MAD)`,
threshold **3.5** on the bandit, **1.0** on the high-variance capstone). It is
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
2. *Behavioural inhibition:* a per-action harm estimate `harm[a]` (EMA, decay 0.90,
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

### 6.1 Experiment 2 — High-Stakes Foraging (5-HT), n=51 ✅ headline result

Cumulative-reward and survival over 5,000 steps (mean ± 95% CI, from
`summary_multiseed.csv`; Survival_Rate = mean steps between deaths):

| Config | Deaths ↓ | Survival ↑ | Total Reward ↑ |
|---|---|---|---|
| **Full Model** | **36.0 ± 1.9** | **166.1 ± 60.7** | **+20,886 ± 793** |
| Ablated DA | 33.1 ± 2.3 | 183.8 ± 61.4 | +21,170 ± 803 |
| Ablated NA | 35.8 ± 1.9 | 166.5 ± 60.7 | +20,918 ± 802 |
| **Ablated 5-HT** | **478.0 ± 12.0** | **10.5 ± 0.3** | **−22,823 ± 3,368** |
| Static Baseline | 468.5 ± 19.6 | 11.2 ± 1.0 | −21,855 ± 3,895 |
| Vanilla DQN | 468.5 ± 19.6 | 11.2 ± 1.0 | −21,855 ± 3,895 |

Paired tests (Full vs …, from `pvalues.csv`):
- **vs Ablated 5-HT:** deaths 36 vs 478, t=−75.3 (p<10⁻⁶); survival t=5.1 (p<10⁻⁶); reward
  +20,886 vs −22,823, t=26.9 (p<10⁻⁶). **Full is vastly better than the same architecture
  without serotonin.**
- vs Static / Vanilla: same picture (deaths t≈−45, reward t≈22.8, all p<10⁻⁶).
- vs Ablated NA: **no difference** (p≈0.32) — NA is correctly irrelevant to survival.
- vs Ablated DA: Ablated-DA is *marginally* better (deaths 33.1 vs 36.0, Holm p≈0; reward
  21,170 vs 20,886, p=0.013 but Holm-n.s. at 0.052) — DA-plasticity mildly **hurts**
  survival (extra exploration noise).

> ⚠️ **Read §6.6 before quoting this as "5-HT is necessary to survive."** The Static/Vanilla
> comparison uses a **Huber-loss** DQN, which gradient-clips the −500 catastrophe and so
> under-weights it. The fair-baseline study shows a **value-corrected** DQN (MSE or
> reward-scaled) survives Exp 2 *without* 5-HT. So 5-HT is necessary *given a Huber-loss value
> function*, not in general; state it that way.
> (M5 note: the Survival_Rate column will shift slightly after the trailing-streak fix is
> re-run — regenerate via `tools/make_tables.py`. Deaths and reward are unaffected.)

**Interpretation (thesis-ready).** Within the same-architecture ablation, serotonin is
**necessary and sufficient** for survival — but two framings matter. First, the risky action
is **negative expected value**: per-step EV = 0.9(50) + 0.1(−500) = **−5**, *below* the safe
+5, so the reward-maximising policy is already all-safe. 5-HT therefore does **not** trade
reward for safety; it lets the agent reach the **true reward optimum** a plain Huber-loss DQN
misses. That DQN fails for value-estimation reasons: (i) the Huber/`smooth_l1` loss clips the
gradient of the rare −500, so the catastrophe is under-weighted; (ii) ε-greedy keeps
resampling the risky arm. 5-HT's two pathways (punishment up-weighting + behavioural
inhibition) counter exactly these. Concretely: the optimal always-safe policy scores
≈**+25,000**; **Full ≈ +20,886** (dies ≈36× — it *reduces*, not eliminates, the failure);
every config **without** 5-HT collapses to ≈**−22,000** and ≈470 deaths.

**Second framing (the honest bound, from §6.6): the win is over a *Huber-loss* DQN, not RL in
general.** Because reason (i) is a loss-function artefact, a value-corrected DQN (MSE, or
reward scaled so the −500 lands at Huber's balanced knee) *also* avoids the trap **without any
serotonin** (51 seeds: MSE +19.3k, reward-scaled +20.4k vs Full's +20.9k — both survive
without 5-HT, though **neither actually beats Full**: Full > MSE with p<10⁻⁴ and ties
reward-scaled at p=0.56). So 5-HT is a **legitimate,
biologically-motivated fix for a specific and common DQN pathology (Huber under-weighting rare
catastrophes)** — state it that way, and cite §6.6, rather than claiming serotonin is required
to survive. **Also drop the original "mathematical trap" arithmetic (risky EV +45 > safe +5) —
it omitted the −500; the true EV is −5.**

### 6.2 Experiment 1 — Volatile Bandit (NA / adaptation), n=51 ✅ beats standard RL; NA isolated

Adaptation latency (steps to re-lock, lower better) and cumulative reward (mean ± 95% CI,
from `summary_multiseed.csv` / `pvalues.csv`):

| Config | Latency ↓ | Total Reward ↑ |
|---|---|---|
| **Full Model** | **174.0 ± 12.5** | **30,726 ± 595** |
| Ablated DA | 178.2 ± 15.2 | 30,957 ± 689 |
| Ablated NA | **347.6 ± 28.1** | 25,922 ± 1,044 |
| Ablated 5-HT | 174.0 ± 12.5 | 30,726 ± 595 |
| Static Baseline | 381.7 ± 27.3 | 25,140 ± 973 |
| Vanilla DQN | 381.7 ± 27.3 | 25,140 ± 973 |

Paired tests (Full vs …, Holm-corrected within the experiment family):
- **vs Static Baseline / Vanilla DQN:** latency 174 vs 382, t=−14.65 (Holm *p*≈0);
  reward t=13.14 (Holm *p*≈0) — **Full significantly better, ~54% faster re-locking.**
- **vs Ablated NA:** latency 174 vs 348, t=−12.79 (**Holm *p*≈0**); reward t=11.64
  (Holm *p*≈0) — **removing NA nearly doubles latency. NA is the isolated driver.**
- vs Ablated DA: latency 174 vs 178, t=−0.85 (*p*=0.40, n.s.); reward *p*=0.24, n.s.
  **DA/plasticity is inert here** (at 51 seeds Ablated-DA is a hair *worse* on latency and
  n.s. on reward — the earlier 10-seed "Ablated-DA slightly better" did not persist).
- vs Ablated 5-HT: **bit-identical** (*p*=1.0) — 5-HT never spikes (all rewards positive), so
  it is correctly inert on the bandit.

**Interpretation (corrected).** The full agent re-locks after a distribution switch ~54%
faster than a standard DQN and earns more — the core "beats standard RL" claim, with paired
statistics. The ablation **cleanly isolates noradrenaline**: NA's switch-triggered ε boost is
what supplies the exploration needed to escape the stale arm; removing it collapses adaptation
back to baseline. **⚠️ Correction to earlier drafts:** a seed-42 pilot had suggested DA/
plasticity was the main contributor with NA n.s.; the 10-seed CSVs show the **opposite** — NA
is decisively the driver (Holm *p*=1.7×10⁻⁴) and DA is inert. This is the cleaner and more
defensible result (it directly isolates the noradrenergic mechanism the experiment was
designed to test); the write-up now reflects the data. **Fairness note:** the Static/Vanilla
baseline is pinned at ε=0.01 with no schedule; §6.6 shows Full *also* beats the best swept-ε
and ε-decay standard DQN, so the NA advantage is over *tuned* fixed exploration, not a strawman.

### 6.3 Experiment 3 — Risky Foraging CAPSTONE (two iterations)

The capstone went through **two designs**; both share the same headline (the full
multi-hormone agent massively beats standard RL, driven by 5-HT). They differ in what
they can say about NA and DA. Read §12 Stages 3–8 for the full story.

**(a) Stateless capstone** (`environments.VolatileRiskyForaging`) — a multi-armed task
fusing Exp 1 + Exp 2: among 5 safe arms one is "good" (μ=10) and **moves** at each of 7
switches; one extra arm is **risky** (+50 on 90%, Death −500 on 10%; true EV ≈ −5).

**(b) Contextual capstone** (`environments.ContextualRiskyForaging`, current
`run_experiment_3`) — a **reversal-learning** upgrade so DA has a non-redundant job:
each step shows a **cue** (one-hot state, 3 cues); each cue has a correct safe action,
and the whole cue→action **mapping reverses** at each of 5 switches. Plus the same
cue-independent lethal arm. Metrics: cumulative reward (headline), **accuracy** (fraction
of steps taking the cue's correct action), death count, re-adaptation latency.

**Earlier-iteration 10-seed results — STATELESS capstone (HISTORICAL; superseded by the
contextual capstone, which is what `run_experiment_3` / the committed CSVs now use — see
"Current committed design" below. Kept for the iteration narrative only):**

| Config | Cumulative reward (±95% CI) | Deaths | Adapt latency | vs Full (reward, Holm) |
|---|---|---|---|---|
| **Full Model** | **14,247 ± 2,993** | 26 | 226 | — |
| Ablated NA | 14,537 ± 2,238 | 21.7 | 223 | p=1.0 (no diff) |
| Ablated DA | 13,763 ± 1,009 | 24.8 | 218 | p=1.0 (no diff) |
| Ablated 5-HT | −14,228 ± 5,312 | 338 | 494 | **p<0.001** ✓ |
| Static Baseline | −11,422 ± 11,275 | 280 | 462 | **p=0.002** ✓ |
| Vanilla DQN | −11,422 ± 11,275 | 280 | 462 | **p=0.002** ✓ |

**What the 10-seed run establishes (honest):**
- ✅ **The full multi-hormone agent significantly beats standard RL.** vs Static *and*
  Vanilla it wins on reward (Holm p=0.002), adaptation latency (226 vs 462, p=0.001) and
  deaths (26 vs 280, p=0.003). This is the headline integration result.
- ✅ **5-HT is necessary and sufficient for the win.** Removing it flips reward
  +14,247→−14,228, deaths 26→338, survival 151→12, latency 226→494 — all p≈0 Holm.
- ❌ **NA and DA are NOT individually necessary here** (Full vs Ablated-NA p=1.0; vs
  Ablated-DA p=1.0). A seed-42 pilot had suggested +59%/+32% for NA/DA, but this **did
  not survive 10 seeds** — it was seed-specific noise.

**Interpretation / design tension (Discussion material):** the rare −500 death and the
+50 gamble are high-variance events that *swamp* NA's reward-change detector and keep
DA's plasticity saturated "on"; in this integrated task the survival imperative (5-HT)
dominates and the adaptation modulators (NA/DA) — which are load-bearing in the *clean*
bandit (Exp 1) — add nothing measurable. **Combining risk with volatility in one task is
genuinely harder than either alone**, and is why NA/DA individual necessity is shown in
Exp 1, not here.

**APPROACH 2 — contextual capstone + robust NA (2026-07-05):** two changes were made to
try to make NA and DA *individually* necessary. (1) NA's detector was made **robust**
(median/MAD instead of mean/σ) so the gamble's heavy-tailed rewards no longer bury the
switch signal — this genuinely works (NA now fires, and Exp 1 is preserved because
median≈mean and 1.4826·MAD≈σ for Gaussian rewards). (2) the task was made **contextual**
(reversal learning) so DA's fast-weights have a non-redundant associative-memory job.

**What approach 2 revealed (a real finding, not a tuning failure):** whether NA and DA
are individually necessary is **regime-dependent**, and the two regimes trade off against
each other:

| Regime (buffer / cues) | Base learner | NA & DA necessary? | Accuracy |
|---|---|---|---|
| Hard (big buffer, 4 cues) | overwhelmed | **Yes** — Full beats ablations | ~0.31 (barely learns) |
| Learnable (small buffer, 3 cues) | partially competent | **No** — redundant | **0.41 overall** (crosses 75% only in the last ~17% of each phase) |

An adaptation modulator is only *necessary* where the base learner **cannot cope** — but
there everyone performs poorly and the margins are small/noisy; where the base learner is
**competent**, NA and DA are **redundant**. Only **5-HT is robustly necessary in every
regime**, because avoiding a rare catastrophe is something a standard value-learner fails
at regardless of competence. This is the honest conclusion: *different neuromodulators
matter in different regimes; harm-aversion (5-HT) is the robustly-necessary one, while
exploration (NA) and plasticity (DA) help specifically when the adaptation demand exceeds
the base learner's capacity.* → a legitimate Discussion contribution about **neuromodulator
redundancy**, not a failure.

**Current committed design:** the **contextual** capstone in the **learnable** regime
(3 cues, replay 600). Committed 51-seed CSV numbers (`summary_multiseed.csv`): **Full reward
+29,447 ± 949, overall accuracy 0.43 ± 0.02, latency 841/1000, deaths 47**, vs Static/Vanilla
reward −23,506, accuracy 0.06, latency 993 (Holm p≈0 on reward, accuracy, deaths, latency).

⚠️ **Do NOT claim "per-phase accuracy ≈0.9".** The committed aggregate accuracy is **0.43**,
and latency 841 of a 1000-step phase means the agent only crosses the 75%-accuracy bar in the
**final ~16%** of each phase — it reaches high accuracy briefly at each phase's end, not
throughout. So the agent **partially** learns the reversal mapping; the +29k-vs-−24k reward gap
is driven almost entirely by **not dying (5-HT)**. **Correction to the earlier 10-seed reading
("DA marginal, NA null"):** at 51 seeds DA gives a small but significant *all-round* benefit
(reward 29.4k vs 27.8k Holm p=0.012; latency 841 vs 943 p≈0; accuracy 0.43 vs 0.39 p≈0), and
**NA is not null** — it significantly improves latency (841 vs 915, Holm p≈0) but *also raises
deaths* (47 vs 39, p≈0) by exploring into the lethal arm, so its **net reward effect is nil**
(p=0.93). **Honest framing: the capstone is a 5-HT-driven survival + integration win, NOT an
"all three hormones cooperating" result** — but state DA's small all-round benefit and NA's
latency/death trade-off (net-nil on reward) plainly, not as "DA marginal / NA null". NA's clean
individual isolation is still claimed in **Exp 1**; the DA-strong probe
(`tools/da_strong_probe.py`, exploratory) explores where DA's plasticity could matter more.

*Sanity note (resolved):* Static ≡ Vanilla come out byte-identical **by construction, not
aliasing** — verified this session: with hormones off the plastic `LocalRLWorker` reduces
exactly to the plain-MLP DQN (matched init, plastic gate = 0), and neither consumes any RNG
during the run beyond the shared ε-greedy/replay draws, so the two stay numerically identical.
This is a validation of the "reduces to baseline" invariant, not a bug.

### 6.4 Legacy — CartPole (DA on continuous control), n=10 ❌ negative (kept as contrast)

Reproducible via `run_experiment_cartpole`. Of 10 seeds, pre-shock competence: DA-on
configs 1–3/10 vs DA-off 8/10; Full reward significantly below Static (p=0.035) and
Vanilla (p=0.006). **DA-gated plasticity impairs stable continuous control** — the
fast-weights that help discrete re-locking (Exp 1, Exp 3) hurt long-horizon balance.
CSVs preserved as `*_cartpole.csv`.

### 6.5 Generalist vs Specialists — the DIRECT novelty test (n=51) ✅ integration beats single-modulator gating

**Why this experiment exists.** The interim novelty (Slides 3/6/7) is *not* "a
serotonin-like caution signal keeps an agent alive" — that is established prior work.
The claim is that **integrating multiple opponent neuromodulators beats single-modulator
gating**. The ablation results (§6.1–6.3) show only 5-HT is *robustly* necessary, which by
itself lands inside the known result and does **not** substantiate the integration claim.
This experiment tests the claim head-on: it pits the Full tri-hormone agent against three
**single-modulator specialists** (DA-only, NA-only, 5-HT-only) across **both** task niches.

**Design.** The two tasks demand opposite things, and each specialist has a fatal blind spot:
- **Task A = adaptation** (Exp 1 volatile bandit, no lethal arm) — metric: adaptation
  latency (lower = faster re-locking after a switch).
- **Task B = survival** (Exp 2 high-stakes foraging, lethal arm) — metric: cumulative
  reward (higher = avoided the −500 death trap).

All four agents run on both tasks, 10 seeds. Module: `generalist.py`; reproduce with
`python main.py --generalist --workers 12`. Outputs: `generalist_summary.csv`,
`generalist_pvalues.csv`, `generalist_headtohead.png` (single-modulator-vs-Full bars in
raw units with 95% CI + significance), `generalist_scatter.png` (adaptation×survival
competence quadrant), `generalist_floor.png` (per-task competence + worst-task floor).

**Scoring — task-anchored ABSOLUTE competence (not cross-agent min-max).** An earlier
version min-max-normalised each axis across the four agents, which forced the worst agent
to *exactly* 0 on each task and produced a 0-vs-1 bar chart that read as manufactured (and
hid the real magnitudes). Each task is now scored in [0,1] against its OWN theoretical
worst case — a task constant, never the other agents:
- **adaptation competence** = 1 − latency / mean_phase_length (the re-lock window is the
  exact worst-case latency; Exp 1 phases average 875 steps).
- **survival competence** = 1 − deaths / (steps × death_prob) (max death exposure = always
  pulling the lethal arm = 500).

**Results (mean over 10 seeds):**

| Agent | Task A latency ↓ | Adapt comp. | Task B reward ↑ | Deaths ↓ | Surv. comp. | Floor ↑ |
|---|---|---|---|---|---|---|
| **Full Model** | **174.0** | **0.80** | **+20,886** | **36** | **0.93** | **0.80** |
| DA only | 347.6 | 0.60 | −23,053 | 479 | 0.04 | 0.04 |
| NA only | 178.2 | 0.80 | −21,855 | 468 | 0.06 | 0.06 |
| 5-HT only | 381.7 | 0.56 | +21,185 | 33 | 0.93 | 0.56 |

*(Floor = min over the two tasks of the agent's absolute competence. High floor = "good
even on your weakest task". Note 5-HT-only floors at 0.56, not 0, because it genuinely
survives — it is merely slow to adapt; the honest numbers are stronger than the old
manufactured 0.00.)*

**Paired Full-vs-specialist tests (Holm-corrected across the 6-comparison family):**
- **Adaptation:** Full ≪ DA-only (174 vs 348, Holm *p*≈0 ✓) and Full ≪ 5-HT-only
  (174 vs 382, Holm *p*≈0 ✓). Full **ties** NA-only (174 vs 178, *p*=0.40) — NA is
  the adaptation specialist, so an expected tie, *not* a Full win.
- **Survival:** Full ≫ DA-only (+20,886 vs −23,053, Holm *p*≈0 ✓) and Full ≫ NA-only
  (+20,886 vs −21,855, Holm *p*≈0 ✓). Full **ties** 5-HT-only (+20,886 vs +21,185,
  raw *p*=0.010, Holm-n.s.; 5-HT-only is if anything a hair higher) — 5-HT is the survival
  specialist, expected tie.

**Interpretation (the novelty, and it is defensible).** On **each** task the Full agent is
statistically *tied with the best specialist* for that task, and *significantly beats every
specialist that lacks the relevant hormone*. Therefore **no single-modulator agent is
competent on both tasks**: DA-only and NA-only adapt but walk into the lethal arm (reward
≈ −22k, ~470 deaths); 5-HT-only survives but is **>2× slower** to re-adapt (382 vs 174
steps). Only the integrated tri-hormone agent keeps a high worst-task **floor (0.80 vs 0.56
for 5-HT-only and ≤0.06 for the arms that die)**. This is the direct evidence for
*"integration beats single-modulator gating"*: the architecture's value is that it is the
**only configuration that is a generalist**, not that any one hormone is universally best.
The head-to-head bars give the supervisor's requested per-baseline comparison (each
single-modulator agent vs Full, in raw units, with Holm-corrected significance); the
quadrant scatter shows Full alone in the good-good corner, each specialist stranded on one
failing edge.

**Why this framing is the right one (Discussion).** Forcing all three hormones to be
*individually necessary in one task* proved regime-dependent and fragile (§6.3, Stage 8).
The generalist framing is **honest and robust**: it requires no hormone to be necessary in
a single task; it shows the integrated agent dominates the *space* of tasks that no
specialist can cover, and it cleanly separates this project's contribution from prior
"serotonin-alone" work. It also matches the biological motivation — neuromodulators are not
redundant copies; each covers a different environmental contingency, and an animal needs all
of them because it faces all contingencies.

### 6.6 Fair baselines — does it beat *well-tuned* STANDARD RL? (new; `baselines.py`)

**Why this exists.** The ablation's Static/Vanilla baseline is a **near-greedy ε=0.01 DQN with
no exploration schedule** and a **Huber loss** that gradient-clips the −500. That is a *weak*
reference, so "beats standard RL" was vulnerable to two objections: (C2) NA only beats a
baseline hand-set to barely explore; (H2) 5-HT only beats a DQN crippled by Huber
under-weighting the catastrophe. This study answers both by pitting the Full agent against a
spread of **properly-tuned standard DQNs** across the adaptation (Exp 1) and survival (Exp 2)
tasks. Reproduce: `python main.py --baselines --workers 12`. Outputs: `baselines_summary.csv`,
`baselines_pvalues.csv` (paired t **and Wilcoxon**, Holm-corrected), `baselines_headtohead.png`.

Baselines (`config.BASELINE_CONFIGS`): swept fixed ε ∈ {0.01, 0.05, 0.1, 0.2}, a linear
**ε-decay** (1.0→0.05), and two **value-corrected** DQNs — **MSE** (no Huber clip) and
**reward-scaled** Huber (×0.002, so −500 lands at Huber's balanced knee).

**Committed 51-seed results (`baselines_summary.csv` / `baselines_pvalues.csv`):**

| Agent | Adapt latency ↓ (Exp 1) | Survival reward ↑ (Exp 2) |
|---|---|---|
| **Full Model** | **174** | **+20,886** |
| DQN ε=0.01 (old baseline) | 382 | −21,855 |
| DQN ε=0.1 (best fixed ε) | 260 | −21,803 |
| DQN ε-decay | 261 | −21,602 |
| DQN MSE (value-corrected) | 360 | +19,289 |
| DQN reward-scaled (value-corrected) | 486 | +20,366 |

**What the committed 51-seed run shows:**
- ✅ **C2 holds — NA survives the fair test.** Full re-locks far faster than *every* standard
  DQN, including the **best-tuned fixed ε (0.1)** and the **ε-decay** schedule (174 vs 260),
  all Holm-significant on both t and Wilcoxon. So the NA advantage is over *tuned/annealed*
  exploration, not just ε=0.01. This *strengthens* the Exp-1 claim.
- ⚠️ **H2: the 5-HT "necessity" is bounded — a value-corrected DQN also survives — but at 51
  seeds it does NOT beat Full.** Both value-corrected DQNs **survive Exp 2 without any
  serotonin** (MSE +19.3k, reward-scaled +20.4k), whereas every Huber baseline (any fixed ε,
  ε-decay) dies (≈−20k to −23k). **Corrects the 2-seed pilot's "reward-scaled beats Full at
  +22k":** at 51 seeds Full (+20.9k) **beats MSE** (Holm p<10⁻⁴) and **ties reward-scaled**
  (p=0.56) — no value-corrected DQN beats Full. **Honest conclusion:** 5-HT is a valid,
  biologically-motivated *fix for a Huber-loss DQN's under-weighting of rare catastrophes*, and
  it is **not required** to reach the safe optimum — a correctly-scaled value function gets
  there too (so state the 5-HT result with this bound, §6.1) — but the tri-hormone agent is
  still at least as good as the best value-corrected DQN, not beaten by it.

*(This is exactly what the fair-baseline study is for. Reporting it is more defensible than the
original unbounded "5-HT necessary to survive" claim, and it sharpens the real contribution: a
bio-inspired mechanism that repairs a known DQN failure mode from the policy side.)*

---

## 7. Full parameter reference (`config.py`)

> Values below mirror `config.py` as of this revision. If they ever disagree, **`config.py`
> wins** — re-sync this table from it.

| Symbol | Value | Meaning |
|---|---|---|
| B (baseline) | 1.0 | resting hormone concentration |
| k_DA / k_NA / k_5HT | 0.1 / 0.08 / 0.03 | decay rates (5-HT slowest → longest "mood") |
| DA/NA/5HT spike scale | 1 / 2 / 3 | spike gains |
| VOLATILITY_WINDOW / THRESHOLD | 200 / **3.5** | NA reward-drop detector window / z-threshold (bandit; capstone uses **1.0**) |
| RISK_PENALTY_THRESHOLD | −50 | reward below this = aversive (5-HT spike) |
| α_base / α_max scale | 1e-3 / 2× | learning rate & DA boost ceiling |
| ε_base / ε_max | **0.01** / 0.5 | ε-greedy rate at rest / NA-saturated |
| γ_base / γ_max | 0.99 / 0.999 | discount at rest / 5-HT-saturated |
| HT_PUNISHMENT_GAIN (G) | 4.0 | max loss up-weight on losses |
| RISK_INHIBITION_WEIGHT / HARM_EMA_DECAY | **5.0 / 0.90** | 5-HT behavioural-inhibition penalty weight / harm-EMA decay |
| PLASTIC_ALPHA_INIT / η_decay / η_trace | 0.002 / 0.05 / 0.01 | plasticity strength & trace dynamics |
| HIDDEN_DIM / BATCH / TARGET_SYNC | 128 / 64 / 100 | DQN hyperparameters |
| REPLAY | **per-exp: E1 500 / E2 5000 / E3 600** (fallback 500) | small→forget stale rewards; large→retain rare deaths |
| SEEDS | 42…92 (51) | statistical replicates |
| Exp1 (bandit) | 5 arms, 4,000 steps, switches [500,1100,1800,3000], μ_hi/lo=10/2 | |
| Exp2 (foraging) | 5,000 steps, safe +5, risky +50 / 10% death −500 | |
| Exp3 (contextual capstone) | 3 cues, 6,000 steps, 5 reversals [1000…5000], μ_hi/lo=10/2, risky +50 / 10% death −500, replay 600 | |
| Legacy CartPole | gravity **19.6 (2×)**, force **×1.0**, train≤400, competence 350 / recovery 300 | secondary/negative, not in default suite |
| Fair baselines | swept ε {0.01,0.05,0.1,0.2}, ε-decay 1.0→0.05, MSE, reward-scaled ×0.002 | `config.BASELINE_CONFIGS` (§6.6) |

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
5. **Opponent processing (5-HT suppresses DA) is only partially realised — see Threats
   §H1/§H2 below.** `DA_eff = DA·(1−σ(5HT−B))` halves at rest (σ(0)=0.5; all rest-references
   use 0.5, internally consistent). BUT the two places DA_eff reaches behaviour — the plastic
   gate and α — use its *magnitude* `|DA_eff−0.5|`, so elevated 5-HT (which lowers DA_eff)
   actually *raises* the gate/α when DA is near rest (non-monotone) — the opposite of the
   intended "suppress DA". This is an audit finding, not a claimed feature; the final write-up
   should either fix the wiring (make the gate signed) or retire the "synergistic opponent
   processing" novelty in favour of the honest "generalist coverage" framing (§6.5, §10).
6. **Death in Exp 2 is terminal-like** (bootstrap cut via done-flag) but the episode
   continues with cumulative score reset — models "loss of progress", not end of run.
7. **Competence/recovery thresholds** (350/300 of 500) are judgement calls applied
   identically to all configs; 70%/60% of max is "learned the task", not near-perfect.

---

## 9. Threats to validity & limitations (Threats-to-Validity section)

- **n=51 seeds.** Ample for the large effects. **Correction:** at n=51 **NA *is* cleanly
  isolated** on the bandit (latency 174 vs 348, Holm p≈0) — the earlier "NA
  under-powered (p=0.23)" note came from a stale single-seed pilot and is retracted. Still
  report CIs, not just p.
- **H2 — the 5-HT survival win is bounded to a *Huber-loss* baseline (§6.6).** A value-corrected
  DQN (MSE / reward-scaled) reaches the safe optimum without serotonin, so 5-HT repairs a
  *loss-function pathology*, it is not required in general. State the claim with this bound.
- **C2 — baseline exploration.** The ablation's Static/Vanilla baseline is a fixed near-greedy
  ε=0.01 with no schedule; the fair-baseline study (§6.6) is what upgrades "beats standard RL"
  to "beats *tuned/annealed* standard RL" for the adaptation claim. Quote §6.6 alongside §6.2.
- **H4 — NA's detector conflates environmental worsening with the agent's own choices.** It
  keys on a drop in the *raw reward stream*, so in any task where the agent's action changes the
  reward level (e.g. Exp 2, moving from risky +50 to safe +5) NA can fire spuriously. It works
  on the bandit because the post-switch drop is *involuntary*. A per-action-conditioned baseline
  is future work; treat "NA detects volatility" as task-scoped.
- **M3 — the adaptation-latency metric measures re-lock *and* exploration-settling** (5
  consecutive greedy pulls of the new arm), so it also captures NA's ε subsiding — not "steps
  until the Q-values are correct". Define it precisely; it does not reverse the ranking.
- **M4 — the bandit is treated as a γ=0.99 MDP** though it is really a contextual bandit.
  Applied identically to all configs (fair), and the **γ=0 sensitivity probe**
  (`tools/bandit_gamma0_probe.py`) confirms NA still drives adaptation (Full 164 vs Ablated-NA
  503), so the conclusion is robust to the choice.
- **M6 — Vanilla DQN ≡ Static Baseline, byte-for-byte** (verified this session). It is therefore
  *not* an independent data point but a validation of the reduces-to-baseline invariant; do not
  present it as a separate baseline in tables (collapse the duplicate row).
- **Exp-3 (capstone) is a partial learner.** Overall accuracy 0.41; the agent only reaches
  competence in the final ~17% of each phase (§6.3). Lead with the reward/survival result; do
  not claim it "learns the mapping" cleanly.
- **Legacy CartPole recovery is under-powered by construction** — DA-plasticity suppresses
  competence, so few seeds qualify; report "could not be assessed", not "no faster recovery".
- **Hand-tuned thresholds** (ε_max, punishment gain, harm weight, competence) were set by pilot
  inspection, not swept — a sensitivity analysis would strengthen the claims.
- **DA-plasticity is at best neutral, sometimes harmful** on every task tested; its intended
  benefit (rapid intra-lifetime re-adaptation) is not demonstrated. The DA-strong probe
  (`tools/da_strong_probe.py`) is an exploratory attempt to find a regime where it matters.
- **H1 (audit finding) — opponent processing is not wired to *suppress* DA.** The plastic gate
  `clip(2·|DA_eff−0.5|,0,1)` and `α ∝ 1+|DA_eff−0.5|` are magnitude functions with a minimum at
  rest, so raising 5-HT (which drives DA_eff *below* 0.5) *increases* plasticity/α when DA is
  near rest — opposite to the interim's "serotonin inhibits dopamine" claim, and non-monotone in
  general (verified numerically: with DA at rest, gate = 0.00 → 0.46 → 0.76 → 0.91 as 5-HT rises
  1 → 2 → 3 → 4). The synergy / opponent-processing selling point is therefore **not demonstrated
  by the code**; report the result as generalist coverage (§6.5), or make the gate signed and re-run.
- **H2 (audit finding) — the DA and 5-HT ablations are entangled.** Because the gate and α depend
  on `DA_eff = DA·(1−σ(5HT))`, "Ablated DA" still has an active plastic gate + elevated α whenever
  5-HT is high (Exp 2/3), and "Ablated 5-HT" *also* removes that DA_eff-driven α/plasticity change.
  So neither is a perfectly clean single-variable ablation in the tasks with deaths (it **is** clean
  in Exp 1, where 5-HT never spikes). The headline effects survive — they are huge and independently
  corroborated by the fair-baseline study — but the *ablation-isolation* argument for DA/5-HT in
  Exp 2/3 should carry this caveat, or the control laws be decoupled (gate/α from DA only, harm from
  5-HT only) and the studies re-run.
- **Toy environments.** Bandit / two-choice foraging / contextual capstone / CartPole are
  deliberately minimal to isolate each hormone; generalisation to richer domains is untested.

---

## 10. Suggested narrative & future work (Discussion / Conclusion)

**Honest headline the results support (revised after the fair-baseline study):**
*"The full multi-neuromodulated agent's value is (1) a cleanly-isolated **noradrenergic
adaptation** win and (2) **generalist coverage** across task niches that no single modulator
achieves. On a volatile bandit it re-locks after a reward switch ~54% faster than a standard
DQN — and, critically, faster than the **best swept-ε and ε-decay** DQN (§6.6) — an advantage
the ablation attributes specifically to noradrenaline (removing NA nearly doubles latency, Holm
p=1.7×10⁻⁴; removing dopamine changes nothing). On high-stakes foraging, a serotonergic
behavioural-inhibition mechanism produces overwhelming harm aversion versus a Huber-loss DQN;
we are careful to bound this: a **value-corrected** DQN also avoids the trap, so 5-HT is a
biologically-motivated **repair of a known DQN failure mode** (Huber under-weighting of rare
catastrophes), not a universal necessity. The dopaminergic gated-plasticity mechanism does not
demonstrate its intended benefit on any task tested (neutral-to-harmful), and impairs stable
continuous control (legacy CartPole) — a genuine, reported negative."*

**What actually beats standard RL, stated precisely:** (i) **Exp 1 adaptation** — Full beats
the matched Static/Vanilla *and* the best tuned/annealed standard DQN, isolated to NA (the
strongest, cleanest claim). (ii) **Generalist coverage** (§6.5) — only the tri-hormone agent is
competent across both an adaptation and a survival niche. (iii) **Exp 2 survival** — a valid win
over a Huber-loss DQN, honestly bounded (§6.6). Plus **one clean negative** (DA-plasticity).
This mechanism-by-mechanism, honestly-bounded story is far stronger for a viva than a uniform
"it works", and every claim traces to a committed CSV.

**Framing the method contribution:** the value is as much the **evaluation protocol**
(fair same-architecture ablation, reduces-to-baseline invariant, per-seed paired
statistics with Wilcoxon + Holm, fair well-tuned baselines, and honest bounding of each
claim) as the mechanisms — it is what let us tell a real effect (NA adaptation) from a
loss-function artefact (5-HT vs a Huber DQN) and from a mirage (DA plasticity).

**Future work:** (i) a learned/meta-optimised meta-controller instead of hand-designed
laws; (ii) a DA-plasticity gate that distinguishes *environmental change* from
*ordinary learning error* so plasticity helps recovery without hurting acquisition (the
DA-strong probe is a first step); (iii) an NA detector conditioned on the *chosen action*
so it tracks environmental volatility rather than the agent's own reward-changing choices
(H4); (iv) sensitivity sweeps over the hand-set thresholds; (v) richer, higher-dimensional
non-stationary environments.

---

## 11. Reproducibility (Appendix)

```
# Full study (all experiments, 10 seeds, 12 CPU workers)
python main.py --workers 12

# Per experiment (writes summary_multiseed_expN.csv, pvalues_expN.csv)
python main.py --exp 1 --workers 12   # Bandit (NA)              — fast
python main.py --exp 2 --workers 12   # Foraging (5-HT)          — fast
python main.py --exp 3 --workers 12   # Risky Foraging (capstone) — fast

# Generalist vs Specialists — the novelty test (§6.5). Runs Full + the three
# single-modulator agents across the adaptation (Exp 1) AND survival (Exp 2)
# tasks. Writes generalist_summary.csv, generalist_pvalues.csv,
# generalist_headtohead.png, generalist_scatter.png, generalist_floor.png.
python main.py --generalist --workers 12

# NEW — Fair baselines (§6.6): Full vs well-tuned STANDARD DQNs (swept-ε, ε-decay,
# MSE, reward-scaled) on the adaptation AND survival tasks. The "beats standard RL"
# fairness test. Writes baselines_summary.csv, baselines_pvalues.csv (t + Wilcoxon,
# Holm), baselines_headtohead.png.
python main.py --baselines --workers 12

# NEW — regenerate the dissertation tables from whatever CSVs exist (run AFTER the
# studies above). Writes research_results/RESULTS_TABLES.md. Never hand-copy numbers.
python tools/make_tables.py

# NEW — sensitivity / exploratory probes (serial; run directly):
python tools/bandit_gamma0_probe.py --seeds 42 43 44 45 46   # M4: NA robust to γ=0
python tools/da_strong_probe.py --seeds 42 43 44             # DA-strong (exploratory)

# Legacy CartPole negative result: call experiments.run_experiment_cartpole directly.
```
Workers run on CPU by default (these tiny nets are ~2× faster per-run on CPU than
GPU; measured 25.6s vs 47.3s for one bandit run). Outputs land in `research_results/`:
per-config dashboards, comparative bar charts (mean ± 95% CI), regret curves, and the
summary/p-value CSVs. Key modules: `neuromodulators.py` (sensing), `meta_agent.py`
(modulation laws), `plasticity.py` (`NeuromodulatedLinear`), `worker.py` (DQN + ε-greedy
+ 5-HT pathways), `experiments.py` (protocols/metrics), `evaluation.py` (stats/plots),
`ablation.py` (multi-seed runner), `generalist.py` (generalist-vs-specialists study),
`baselines.py` (fair-baseline battery), `tools/` (table generator + sensitivity probes).
```

---

## 12. Development chronology (in order: what was done → what it showed → what came next)

A step-by-step record of how the project evolved, written for the dissertation's
**Implementation / Iteration** narrative and for the viva ("how did you get here?").
Each stage lists the **action**, the **rationale**, and the **result**.

### Stage 0 — Baseline audit *(starting point)*
- **Action:** full, file-by-file audit of the original code against the interim
  spec/hypothesis, before changing anything.
- **Result:** the original headline results were **artifacts of defects**, not real effects
  — four critical (γ collapses to ≈0.5·γ_base at rest; the Exp-1 adaptation metric counts
  the wrong arm; the "static baseline" differs from the Full model in ~4 ways at once; the
  p-values are single-seed pseudoreplication) plus several high-severity issues (§2).
- **Decision:** fix validity first, *then* judge the science.

### Stage 1 — Validity fixes (Phase 1)
- **Action:** fixed every soundness bug — γ now rests at γ_base (5-HT only lengthens the
  horizon); Exp-1 latency measured per-switch against the *correct* new arm; a single
  same-architecture worker for all configs (the "static baseline" = that worker frozen);
  competence-gated recovery; a **multi-seed harness** with paired tests, 95% CIs and
  Holm correction; added an Ablated-DA config; stopped the frozen target network mutating
  its Hebbian trace.
- **Result:** with the bugs gone, the **original mechanisms did not beat standard RL**, and
  the previously-reported "5-HT survival win" **disappeared** — it had been an artifact of
  the γ bug (serotonin was only rescuing the agent from a broken resting discount).
- **Decision:** the mechanisms, as specified, don't influence behaviour enough → redesign.

### Stage 2 — Mechanism redesign (Phase 2)
- **Action:** re-engineered how each hormone acts on behaviour:
  - **NA → ε-greedy exploration rate** (replaced softmax temperature, which is scale-
    sensitive and near-random when Q-values are close, e.g. CartPole).
  - **5-HT → behavioural inhibition** — a per-action "harm-history" penalty subtracted at
    action selection, plus a punishment-weighted loss (previously 5-HT only touched γ and
    suppressed DA, never the policy, so it could not produce harm aversion).
  - **DA → dopamine-gated plasticity** (gate = 0 at rest so the net behaves as a standard
    DQN; opens on prediction-error surprise) + matched `nn.Linear` init so the gate-off
    plastic net is identical to a plain layer.
- **Result (10 seeds): two clean wins + one honest negative.**
  - Exp 1 (bandit, NA): Full beats standard RL — faster re-locking after a switch.
    *(⚠️ Superseded by Stage 10: a later run + tuning shows **NA**, not DA, is the driver —
    removing NA nearly doubles latency, removing DA does nothing. The Stage-2 "DA was the
    bigger contributor" reading was a seed-specific artefact; see §6.2.)*
  - Exp 2 (foraging, 5-HT): removing 5-HT → many more deaths, reward flips positive→negative
    (p<10⁻⁶). Necessary and sufficient for survival *(vs a Huber-loss DQN — Stage 10 bounds
    this: a value-corrected DQN survives without 5-HT; see §6.6)*.
  - Exp 3 (CartPole, DA): DA-gated plasticity **impairs** stable continuous control
    (only 2/10 DA-on seeds reach competence vs 8/10 with DA off). A genuine negative.

### Stage 3 — Replace CartPole with an integrative capstone
- **Rationale:** CartPole didn't support the hypothesis (plasticity hurts continuous
  control) and *nothing* tested the project's core **multi-hormone** novelty. Goal: one
  task that needs all three hormones at once.
- **Action:** built `VolatileRiskyForaging` = Exp 1 (a moving good arm) fused with Exp 2
  (a tempting, negative-EV lethal arm); rewrote `run_experiment_3`; kept the CartPole
  runner as `run_experiment_cartpole` (secondary/negative); rewired evaluation, plots and
  statistics; later wired the **live pygame view** to the new task as well.

### Stage 4 — Capstone pilot *(first attempt)*
- **Result (2-seed pilot):** **only 5-HT** came out necessary. Full == Ablated-NA
  *bit-for-bit* (NA never fired) and Ablated-DA was slightly *better* than Full (DA mildly
  hurting).
- **Diagnosis:** the lethal arm's ±extreme rewards (a) **swamp the variance** in NA's
  reward-change detector, so a switch's modest reward dip never crosses the alarm
  threshold, and (b) keep DA's plasticity **saturated "on"** — the same regime that made it
  hurt on CartPole. In short, the rare-catastrophe structure that *makes 5-HT necessary*
  actively *sabotages NA and DA*.

### Stage 5 — NA robustness fix + tuning
- **Action:** excluded catastrophic (death) rewards from NA's volatility history (they are
  5-HT's domain, not NA's); tuned ε_base down (0.1 → 0.01, so the base agent is near-greedy
  and NA's exploration boost becomes load-bearing) and raised/made-selective the NA
  volatility threshold (3.5) so NA fires on genuine switches, not ε-greedy noise.
- **Result (seed-42 pilot):** looked excellent — Full beat *every* ablation (vs NA +59%,
  DA +32%, 5-HT +180%). Suggested all three were now individually necessary.

### Stage 6 — Full 10-seed capstone run *(reality check)*
- **Action:** ran the capstone across 10 seeds with paired, Holm-corrected tests.
- **Result:**
  - ✅ Full **massively beats standard RL** — vs Static *and* Vanilla on reward
    (+14.2k vs −11.4k, Holm p=0.002), adaptation latency (226 vs 462, p=0.001) and deaths
    (26 vs 280, p=0.003).
  - ✅ 5-HT **necessary and sufficient** — removing it flips reward to −14.2k, deaths to
    338, survival 151→12 (all p≈0).
  - ❌ NA and DA **not individually necessary** (Full vs Ablated-NA p=1.0; vs Ablated-DA
    p=1.0). The seed-42 pilot did **not** survive 10 seeds — it was seed-specific noise.
- **Conclusion:** the capstone is an **integration win** ("multi-hormone agent beats
  standard RL"), *not* an "every-hormone-individually-necessary" result. Honest, and it
  cleanly motivates the next step.

### Stage 7 — Approach 2 *(make NA and DA individually necessary)*
- **Goal:** a task where removing NA or DA is *also* individually worse, robustly across
  seeds — the strongest form of the synergy claim.

### Stage 8 — Approach 2 executed, and its honest outcome
- **Action 1 — robust NA detector.** Replaced NA's mean/σ change-detector with a **median
  / MAD** one so the gamble's heavy-tailed rewards stop burying the switch signal.
  **Result:** NA now fires on the capstone (0% → ~18% of steps) and becomes helpful; Exp 1
  is preserved (median≈mean, 1.4826·MAD≈σ for Gaussian rewards). A genuine, principled win
  — kept.
- **Action 2 — contextual capstone.** Diagnosed that DA is redundant in a *stateless*
  bandit (re-locking one fact is done by NA+backprop; DA-plasticity's unique strength is
  *associative* memory). Built `ContextualRiskyForaging` = reversal learning (cue→action
  mapping that reverses) + the lethal arm, to give DA a non-redundant job.
- **Result (the honest finding): NA/DA individual necessity is REGIME-DEPENDENT.** Swept
  task difficulty (buffer size / #cues):
  - *Hard regime* (big buffer, 4 cues): base learner overwhelmed → **Full beats every
    ablation** (all three "necessary"), but absolute accuracy ≈0.31 (barely learns) and
    margins small/noisy.
  - *Learnable regime* (small buffer, 3 cues): base learner competent (per-phase accuracy
    ≈0.9) → **NA and DA become redundant** (removing them doesn't hurt).
  A modulator is only necessary where the base learner *can't cope*; where it is competent,
  NA/DA wash out. **Only 5-HT is robustly necessary in every regime.**
- **Decision (user):** accept the honest framing. Keep the robust-NA fix and the contextual
  capstone in the *learnable* regime (agent genuinely learns the reversal task; Full ≫
  standard RL; 5-HT necessary). Report the **regime-dependence / neuromodulator-redundancy**
  as a real Discussion contribution rather than engineer a fragile "all-three-necessary"
  number that would not survive 10 seeds or a viva.
- **Take-away for the thesis:** the synergy claim is honestly stated as *"the full agent
  beats standard RL; 5-HT is the robustly-necessary component; NA and DA are each isolated
  in their own dedicated tasks (Exp 1) and contribute in the capstone only when adaptation
  demand exceeds the base learner's capacity."*

### Stage 9 — Generalist vs Specialists *(the direct novelty test)* ✅
- **Motivation.** Ablation ("remove one hormone from Full") only isolates 5-HT robustly,
  because a competent base learner makes NA/DA redundant (Stage 8). But "5-HT alone
  matters" is *prior work* — it does not test the interim novelty (Slides 3/6/7):
  *integration beats single-modulator gating*. That claim needs the **opposite** of
  ablation: give an agent only **one** hormone and show the Full trio beats every such
  specialist.
- **Insight (why it works).** A specialist only fails if tested *outside its niche*. So
  test across a **battery of two opposite tasks**: adaptation (Exp 1, no lethal arm) and
  survival (Exp 2, lethal arm). Each specialist aces its home task and collapses on the
  other; only the Full agent is good at both.
- **Action.** Added `GENERALIST_CONFIGS` (Full, DA-only, NA-only, 5-HT-only) to
  `config.py`; built `generalist.py` (parallel battery runner + paired Holm-corrected
  Full-vs-specialist stats + a head-to-head bar figure, a quadrant scatter and a worst-task
  "floor" bar); wired `python main.py --generalist --workers 12`. Validated on a 3-seed
  pilot, then ran 10 seeds.
- **Result (10 seeds, decisive).**

  | Agent | Adapt latency ↓ | Survival reward ↑ | Deaths ↓ | Floor ↑ |
  |---|---|---|---|---|
  | **Full** | **165** | **+21,191** | **36** | **0.81** |
  | DA only | 368 | −21,510 | 480 | 0.04 |
  | NA only | 163 | −20,753 | 475 | 0.05 |
  | 5-HT only | 390 | +21,456 | 33 | 0.55 |

  On adaptation, Full ≪ DA-only (Holm p=6.8e-5) and ≪ 5-HT-only (p=1e-4), ties NA-only
  (p=0.83). On survival, Full ≫ DA-only (p≈0) and ≫ NA-only (p=5e-6), ties 5-HT-only
  (p=0.44). **On each task Full ties the best specialist and beats the rest; no specialist
  is good at both.** Worst-task floor: Full 0.81 vs 0.55 (5-HT-only) and ≤0.05 for the arms
  that die.
- **Scoring revision (supervisor feedback).** The floor was originally min-max-normalised
  across the four agents, which forced the worst agent to *exactly* 0 on each axis — a
  0.99-vs-0.00 bar chart that read as manufactured and hid the magnitudes. Replaced with
  **task-anchored absolute competence** (adaptation = 1−latency/window; survival =
  1−deaths/max-exposure): non-circular, interpretable, and the honest numbers (Full 0.81;
  5-HT-only floors at 0.55 because it genuinely survives) are *more* defensible. Added
  `generalist_headtohead.png` — the direct single-modulator-vs-Full comparison the
  supervisor asked for, in each task's raw units with 95% CI and Holm-corrected stars.
- **Take-away for the thesis (the headline novelty result).** *"Integration beats
  single-modulator gating: the tri-hormone agent is the only configuration competent
  across both task niches — each single-modulator specialist catastrophically fails the
  task outside its niche."* This is robust (survives 10 seeds + Holm), honest (no forced
  all-three-necessary number), and cleanly distinct from prior serotonin-alone work.
  Written up as §6.5; figures `generalist_headtohead.png`, `generalist_scatter.png`,
  `generalist_floor.png`.

### Stage 10 — Second independent audit + remediation (2026-07-16) ✅
- **Motivation.** A fresh, file-by-file audit against the interim spec. It confirmed the
  mechanisms are correctly implemented, but found (a) the write-up **contradicted its own CSVs**
  on the central Exp-1 claim, (b) two "beats standard RL" wins rested on a **deliberately weak
  baseline** (near-greedy ε=0.01; Huber-clipped −500), and (c) the flagship **DA plasticity is
  inert**. Empirically re-verified: Static ≡ Vanilla byte-identical; Exp-1 Full 131 vs
  Ablated-NA 207 vs Ablated-DA 121 (NA helps, DA doesn't); Exp-2 Full 6.7% risky-pulls vs
  Ablated-5HT 99.3%.
- **C1 — narrative vs data (the big fix).** The committed `pvalues.csv` shows **NA** is the
  isolated Exp-1 driver (Full vs Ablated-NA latency 174 vs 348 at 51 seeds, Holm p≈0; the
  Stage-10 audit itself saw 165 vs 368 at 10 seeds), and DA is
  inert (p=0.40) — the **opposite** of the earlier "DA is the bigger contributor" prose (a
  seed-42 pilot artefact). Rewrote §0, §6.1, §6.2, §6.3, §10, and the Stage-2 note to match the
  data; added `tools/make_tables.py` so every §6 table is **regenerated from the CSVs**
  (`RESULTS_TABLES.md`) rather than hand-copied — closing the drift that caused C1.
- **C2/H2 — fair baselines (`baselines.py`, `--baselines`, §6.6).** Added a battery pitting Full
  against **well-tuned standard DQNs**: swept fixed-ε {0.01…0.2}, a linear **ε-decay**, and two
  **value-corrected** critics (**MSE**, **reward-scaled**). `StaticBaselineWorker` gained
  ε-annealing + `loss`/`reward_scale` options (defaults byte-identical, so Vanilla is unchanged).
  **Pilot findings:** (C2) Full re-locks faster than the *best* fixed-ε **and** ε-decay DQN — the
  NA/adaptation win is over *tuned* exploration, strengthened. (H2) the value-corrected DQNs
  **survive Exp 2 without any 5-HT**, so serotonin **repairs a Huber-loss pathology** rather than
  being universally necessary — the 5-HT claim is now honestly bounded (§6.1, §6.6).
- **DA reframe + probes (D, F).** DA-plasticity reframed as a **characterized negative** (main
  path). Added an exploratory `tools/da_strong_probe.py` (higher `PLASTIC_ALPHA_INIT` + hard-
  regime capstone; `plastic_alpha` now threaded through, default-preserving) as *future work*.
  Added `tools/bandit_gamma0_probe.py`; it **confirms NA still drives adaptation under γ=0**
  (Full 164 vs Ablated-NA 503) — the M4 sensitivity check.
- **Stats hardening (M2).** Added a **Wilcoxon signed-rank** test alongside every paired t-test
  (`evaluation._paired_p`), Holm-corrected, in the ablation, generalist, and baseline p-value
  CSVs — the defensible test for the count/capped metrics.
- **Correctness/cleanup.** Deleted dead single-seed functions that referenced a removed
  `recovery_time` key (latent `KeyError`, M1); counted the **trailing survival streak** (M5);
  simplified `_compute_ht_spike` (L1); fixed the `info["switched"]` off-by-one in all three envs
  (L2); removed unused `meta._death_count` (L3); reconciled all stale constants in §5/§7/§0B to
  `config.py` (L4).
- **Validated.** `test_invariants` passes; all modules compile/import; Static ≡ Vanilla identity
  preserved after the worker refactor; NA-win direction and Wilcoxon columns reproduced on pilots.
- **Net effect on the thesis.** The **NA/adaptation** win is *stronger* and cleaner; the
  **5-HT/survival** win is *honestly bounded* to a Huber-loss baseline; **generalist coverage**
  remains the novelty; **DA** is a documented negative. Every claim now traces to a committed CSV.
  ✅ **Done (2026-07-19):** the `--baselines` study was run at **51 seeds** and §6.6 /
  `RESULTS_TABLES.md` now carry the committed numbers — Full **beats** MSE (Holm p<10⁻⁴) and
  **ties** reward-scaled (p=0.56), correcting the 2-seed pilot's "reward-scaled beats Full".

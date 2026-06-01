# Diagnostic Metrics

Epistemic Crucible produces a **vector** of diagnostic metrics, not a single aggregate score.
Each metric targets a distinct aspect of epistemic capability. Collapsing them would hide the
trade-offs that make the benchmark informative.

---

## 1. Task Success Rate (TSR)

**Definition:** Fraction of episodes where the agent achieved the declared goal.

**Formula:**
```
TSR = (number of episodes with goal_achieved=True) / (total episodes)
```

**Required data:** `outcome` records, field `goal_achieved`.

**Interpretation:**
- High TSR: agent reaches the goal reliably.
- Low TSR: agent fails to complete tasks, examine failure modes via `failure_diversity`.
- TSR alone cannot distinguish luck from understanding; always pair with `transfer_success`.

**Gaming risk:** An agent that memorises the exact seed-to-action mapping for every training seed
can achieve TSR=1.0 on training episodes without any causal knowledge. Shortcut-sensitive families
(affordance, tool_substitution) expose this on test splits.

---

## 2. Transfer Success (TS)

**Definition:** Per-(family, agent) TSR on training seeds vs test seeds, plus their delta.

**Formula:**
```
train_TSR = TSR(split=train)
test_TSR  = TSR(split=test)
delta     = test_TSR - train_TSR
```

**Required data:** `outcome` records, fields `goal_achieved`, `split`, `family`, `agent`.

**Interpretation:**
- `delta > 0`: agent generalises beyond training worlds (rare without causal grounding).
- `delta ≈ 0`: performance is consistent, either reliably good or reliably bad.
- `delta < 0`: agent exploits train-specific features; performance degrades on test.
- Negative delta on affordance/tool_substitution is expected for colour-cue baselines.

**Gaming risk:** A lucky positive delta can arise from small test sets with easy seeds.
Always report standard deviation across seeds alongside the mean delta.

---

## 3. Shortcut Sensitivity (SS)

**Definition:** TSR drop from train to test for shortcut-exposed task families (affordance, tool_substitution).

**Formula:**
```
SS = train_TSR - test_TSR    [for affordance and tool_substitution only]
```

**Required data:** `outcome` records filtered to `family ∈ {affordance, tool_substitution}`.

**Interpretation:**
- `SS ≈ 0`: agent is not exploiting the surface shortcut, either it learned the hidden property
  or it never succeeded on either split.
- `SS > 0`: agent relies on a surface cue (colour in affordance, shape in tool_substitution)
  that is decorrelated in test worlds.
- `SS < 0`: agent performs better on test, inspect why (possibly the test worlds are easier).

**Failure interpretation:** High SS combined with low test TSR is the primary signal of
shortcut exploitation.

**Gaming risk:** A memorisation agent achieves SS≈0 because it uses different lookup keys
per split (the task_id encodes the split label). SS must be interpreted alongside the absolute
test TSR value.

---

## 4. Intervention Validity (IV)

**Definition:** Fraction of APPLY/COMBINE/INSPECT steps that produced at least one non-empty effect string.

**Formula:**
```
IV = |{steps : action.kind ∈ {apply, combine, inspect} ∧ effects ≠ []}| / |intervention_steps|
```

**Required data:** `step` records, fields `action.kind`, `effects`.

**Interpretation:**
- `IV ≈ 1.0`: agent's interventions are well-targeted; it consistently triggers causal rules.
- `IV ≈ 0.0`: agent is applying tools randomly with no effect.
- Moderate IV with low TSR suggests the agent intervenes effectively but on wrong objects.

**Failure interpretation:** IV=0 combined with goal failure indicates the agent never discovered
the correct interaction. IV>0 with goal failure indicates the agent triggers some rules but
fails to chain them into a solution.

**Gaming risk:** INSPECT on a DETECTOR object always produces a marker effect. Excluding INSPECT
gives a stricter metric `apply_combine_validity`. Report both variants when detectors are present.

---

## 5. Intervention Efficiency (IE)

**Definition:** For successful episodes only, the ratio of oracle solution length to agent step count.

**Formula:**
```
IE = oracle_action_count / agent_steps    [capped at 1.0, successful episodes only]
```

**Required data:** `outcome` records with `goal_achieved=True`; TaskSpec for `oracle_action_count`.

**Interpretation:**
- `IE = 1.0`: agent solved the task in exactly the oracle number of steps.
- `IE < 1.0`: agent took more steps than necessary (explored before converging).
- Requires `get_spec` callable; returns `value=None` without it.

**Failure interpretation:** Not applicable to failed episodes; always condition on `goal_achieved=True`.

**Gaming risk:** Agents that solve tasks through short lucky action sequences score higher than
systematic solvers. Efficiency should be read alongside the success rate, high IE with low TSR
means the agent occasionally gets lucky instead of reliably solving tasks.

---

## 6. Counterfactual Accuracy (CA)

**Definition:** Accuracy of agent behaviour or predictions under counterfactual conditions.

Two modes:

**Behavioral mode:** For COUNTERFACTUAL family tasks, checks whether the agent applied the source
to the object that actually has the relevant latent property (`classify_correct_obj_id`).

```
CA_behavioral = |{episodes : agent applied source to correct_obj}| / |counterfactual_episodes|
```

**Prediction mode:** Counts PREDICT action steps and checks whether `predicted_effect` matches
the actual effect observed after the intervention. Returns `count=0` if no PREDICT actions exist.

**Required data:**
- Behavioral: `step` + `outcome` records, plus TaskSpec for `classify_correct_obj_id`.
- Prediction: `step` records with `action.kind=predict`.

**Interpretation:**
- High behavioral CA: agent identifies which object has the relevant property even under
  counterfactual property assignment (train vs test COUNTERFACTUAL family).
- Low prediction CA: agent's causal model is inaccurate.

**Failure interpretation:** Behavioral CA=0.5 on balanced splits is consistent with random choice
between two equally-visible blocks, the baseline floor for this metric.

**Gaming risk:** An agent that always applies source to block_0 achieves CA=1.0 on training
(block_0 is soluble on train) but CA=0.0 on test (property is swapped). Behavioral CA must be
reported separately for train and test splits.

---

## 7. Concept Reuse Proxy (CR)

**Definition:** Whether gathering causal evidence transfers to novel appearances, measured on the
TEST split (same hidden rule, changed visible features).

```
CR = test_TSR(episodes with prior evidence) - test_TSR(episodes without prior evidence)
```

"Prior evidence" = the agent performed at least one **informative** INSPECT/APPLY intervention
during the episode (one producing a real effect, not `no_effect`/`illegal_action`).

**Required data:** `step` records (to detect informative interventions) + `outcome` records
(goal_achieved), filtered to `split=test`.

**Interpretation:**
- Positive CR: agents that experiment before acting succeed more often on unfamiliar surface forms , 
  evidence of concept reuse instead of surface-pattern matching.
- CR ≈ 0: intervening provides no transferable benefit; the agent is not reusing latent concepts.

**Failure interpretation:** Low reuse suggests the agent learned task-specific routines rather than
intervention-grounded abstractions.

**Gaming risk:** Inflated if hidden-rule structure leaks through object types/names, or by an agent
that always intervenes trivially. Read alongside `intervention_validity`.

---

## 8. Failure Diversity (FD)

**Definition:** Count of distinct failure modes across all failed episodes.

**Failure modes:**

| Mode | Condition |
|------|-----------|
| `timeout` | Episode steps ≥ default max steps |
| `no_interaction` | Zero APPLY/COMBINE/INSPECT actions in episode |
| `high_illegal` | Illegal action rate > 50% |
| `no_effects` | No non-empty effect strings across the episode |
| `energy_depleted` | Agent energy ≤ 0 at episode end |

**Formula:**
```
FD.mode_count = |{failed episodes : mode condition holds}|    [per mode]
FD.distinct_modes = |{modes : FD.mode_count > 0}|
```

**Required data:** `outcome` records with `goal_achieved=False`, plus `step` records for per-step detail.

**Interpretation:**
- `distinct_modes = 1`: agent fails in a stereotyped way (e.g., always times out).
- `distinct_modes > 3`: agent fails for diverse reasons, may indicate exploration without convergence.
- `no_interaction` failures indicate the agent never attempts causal interventions.
- `high_illegal` failures indicate the agent's action selection is poorly grounded.

**Gaming risk:** Failure diversity does not penalise agents that fail identically in predictable
ways. High diversity is not inherently desirable, a diverse failure mode distribution with 100%
failure rate is worse than a uniform single failure mode with 10% failure rate.

---

## 9. Curriculum Progression (CP)

**Definition:** Trend in TSR across seed-ordered windows within each agent+family run.

**Formula:**
```
windows   = [TSR(seeds[0:w]), TSR(seeds[w:2w]), …]    [window size w, default 5]
slope     = linear_regression_slope(window_index, window_TSR)
```

**Required data:** `outcome` records; seeds used as episode-temporal ordering proxy.

**Interpretation:**
- `slope > 0`: agent improves over time (relevant for learning agents: tabular_rl, world_model).
- `slope ≈ 0`: performance is flat, either the agent is not learning or already at ceiling.
- `slope < 0`: performance degrades, possible catastrophic forgetting.

**Failure interpretation:** Meaningful only for agents with persistent state across episodes
(Q-table, transition model). Random and heuristic agents should show flat slopes.

**Gaming risk:** Slope is statistically meaningless with fewer than 3 windows (≥15 seeds).
Variance in small seed counts can produce spurious positive or negative slopes.
Require at least 3 windows before interpreting the trend direction.

# Benchmark Card: Epistemic Crucible Micro

## Overview

**Epistemic Crucible Micro** is a generative diagnostic benchmark for evaluating causal and
counterfactual reasoning in agents operating in symbolic grid worlds. Unlike reward-only
benchmarks, it evaluates the *process* of discovery, whether agents correctly identify hidden
causal properties, make valid interventions, and update beliefs across distribution shifts.

---

## Task Description

### World structure
- 6×6 grid, 3–4 objects, one agent.
- Objects have **visible properties** (type, colour, shape, texture, size, state, marker) and
  **hidden causal properties** (conductivity, magnetism, solubility, fragility, charge, affinity).
- Agents observe only visible properties; hidden properties are never in the observation.

### Action space
MOVE (4 directions), PICKUP, DROP, INSPECT, APPLY (tool → target), COMBINE (object × object),
SWAP, ISOLATE, PREDICT, WAIT.

### Task families

| Family | Goal | Key hidden property | Train/Test split |
|--------|------|---------------------|-----------------|
| A: Affordance | OPEN gate | conductivity | RED correlated on train; decorrelated on test |
| B: Causal Gate | OPEN gate (multi-step) | magnetism | Layout varies; structure stable |
| C: Counterfactual | CLASSIFY soluble block | solubility | Property assignment swapped on test |
| D: Tool Substitution | OPEN gate | conductivity | ROD shape on train; FLAT shape on test |
| E: Contradiction | OPEN gate (affinity) | affinity | No affinity restriction on train; required on test |

---

## Evaluation Protocol

### Seeds and splits
- Seeds are integers assigned deterministically to TRAIN (80%) or TEST (20%) using an offset RNG
  to avoid correlation with world generation.
- Default evaluation (`configs/baselines.yaml`): 20 seeds per family × 6 symbolic agents.
- Publication sweep (`configs/publication.yaml`): 100 seeds × 7 agents (incl. the learned
  `neural_ppo`) × 5 families. Committed reference artifacts: [`results/reference/`](../results/reference/).

### Episode structure
1. Environment reset with seed + TaskSpec.
2. Agent acts for up to `max_steps` (30–40 depending on family).
3. Goal checked at episode end via `check_goal(goal, world)`.
4. Full trace logged to JSONL (one record per step + one outcome record per episode).

### Reward signal
By default the environment is reward-free (`reward=0.0`); metrics are computed from goal
achievement and trace analysis, not accumulated reward, this keeps the diagnostic traces and
symbolic baselines untouched. An **opt-in** reward layer (`crucible/rewards.py`, enabled with
`terminate_on_goal=True`) supplies a sparse goal reward plus optional shaping for training the
neural baseline; it does not affect the default evaluation path.

### Neural baseline (`neural_ppo`)
A learned policy over the variable candidate-action set (object embeddings pooled to a state;
each candidate scored from its action kind/direction/referenced objects). It reads only public
observations. The released checkpoint is trained by **behavior cloning** the heuristic solver on
the affordance TRAIN split (`python experiments/train_neural.py --config configs/neural.yaml`),
then evaluated frozen through the standard runner so it flows through the same metric vector. PPO
fine-tuning (sparse goal reward + potential-based navigation shaping) is implemented and GPU-aware
but disabled by default, on this small task it is unstable and can collapse the warm-start.

**GPU note (honest):** the agent is GPU-capable and was developed/verified on an RTX 5090 (sm_120,
CUDA 12.8). However, this symbolic benchmark is **CPU-bound**, episode rollouts are Python env
steps and the policy is tiny, so training is *faster* and bit-reproducible on CPU, while CUDA adds
run-to-run variance from nondeterministic atomics. The reference checkpoint is therefore trained on
CPU. A GPU is neither required nor beneficial for this task at its current scale.

### LLM panel (`experiments/run_llm_eval.py`)
A cross-provider evaluation of open-weights models at the 7-8B and 32B scales
(local, GPU; 32B in 4-bit) and frontier models (Claude via Anthropic, GPT/o3 via
OpenAI). The protocol is designed to be a **genuine** test of intervention-grounded
reasoning, not prompt-following:

- **Neutral prompt**, describes only the action mechanics and the goal. It does
  *not* state that hidden properties exist, that effects are reported, or any
  solution strategy; the agent must discover the intervention itself.
- **Object-ID anonymization**, IDs encode the split (`aff_0_train_gate`), so the
  goal, observation, candidate list, and effect feedback are rewritten to neutral
  labels (`object_1..N`). A regression test guards against split/ID leakage.
- **Index action selection** over a legal-only candidate list (decouples decision
  quality from output formatting), greedy decoding, cached responses.
- **Decision-focused layout** (`compact_layout`) clusters objects near the agent
  so the task isolates the causal decision from long-horizon navigation. A full-task
  control (objects spread across the grid) shows the strongest models transfer at
  close to their decision-focused rates, so the layout removes a navigation confound;
  the weaker 7-8B models sit at the low end.
- **Paired train/test** evaluation, an in-context **dose-response shortcut probe**
  (cue/mechanistic/anti-cue demonstrations × count), and a **hint ladder**
  (intervention/property/rule hints) that localizes attribution vs execution failure.

Reference results and the capability gradient are in
[`results/reference/llm/`](../results/reference/llm/).

---

## Metric Vector

Metrics are **not** collapsed into a single score. Report the full vector:

| Metric | What it measures |
|--------|-----------------|
| Task Success Rate (TSR) | Goal achievement fraction |
| Transfer Success (TS) | Train vs test TSR delta per family/agent |
| Shortcut Sensitivity (SS) | TSR drop under surface-feature decorrelation |
| Intervention Validity (IV) | Fraction of interventions with causal effects |
| Intervention Efficiency (IE) | Relative step efficiency vs oracle (successful episodes) |
| Counterfactual Accuracy (CA) | Correct identification of counterfactual property bearer |
| Concept Reuse Proxy (CR) | Test-split TSR gain from prior in-episode evidence |
| Failure Diversity (FD) | Distinct failure modes across failed episodes |
| Curriculum Progression (CP) | TSR trend across seed-ordered windows |

Plus the perturbation-based Shortcut Exposure Score (clean vs perturbed TSR drop).

See `docs/metrics.md` for full definitions, formulas, and gaming risks.

---

## Known Limitations

1. **Default path is reward-free**: the dict-based Tabular RL and world-model baselines run on
   zero reward and behave as exploration placeholders. The opt-in reward layer is used only by
   the neural baseline's training, not by the default evaluation.

2. **CLASSIFY goal not machine-checkable**: The COUNTERFACTUAL family uses a CLASSIFY goal
   whose correctness is verified via trace inspection, not live state. `check_goal()` returns
   False for CLASSIFY goals by design. TSR is always 0 for this family in the current runner.

3. **Small grid**: 6×6 grids limit compositional complexity. Rule interactions are deterministic
   and relatively few. Findings may not transfer to larger or more compositionally rich environments.

4. **Neural baseline scope**: the learned `neural_ppo` agent targets the affordance family, where
   public features cleanly determine the expert action (behavior-cloned, BC accuracy ~0.81).
   tool_substitution is solvable by the heuristic (~0.58) but its demonstrations are not cloneable
   from public features alone (BC accuracy plateaus ~0.46), and causal_gate/counterfactual lack
   usable demonstrations. Extending learned agents to more families is future work. LLM agent is
   optional and not evaluated by default.

5. **No partial credit**: Goal success is binary. Agents that make partial progress (e.g., pick
   up the correct tool but fail to reach the gate) score the same as agents that take no steps.

---

## Failure Mode Taxonomy

| Mode | Condition | Typical cause |
|------|-----------|---------------|
| `timeout` | steps ≥ max_steps at failure | Agent explores without converging |
| `no_interaction` | zero APPLY/COMBINE/INSPECT actions | Agent never attempts causal interventions |
| `high_illegal` | illegal action rate > 50% | Action selection is poorly grounded |
| `no_effects` | no non-empty effect strings | Agent misses all causal triggers |
| `energy_depleted` | energy ≤ 0 at episode end | Agent triggers energy-draining hazard repeatedly |

---

## Anti-Gaming Notes

**Why no single aggregate score:**

Aggregate scores incentivise Goodhart's Law dynamics, optimising for the number maximises
the benchmark instead of the capability it measures. The Epistemic Crucible is specifically
designed to expose *which* capabilities are present and which are missing.

**Known gaming strategies and their exposure:**

| Strategy | Exposed by |
|----------|------------|
| Seed memorisation | Transfer Success (negative delta on test) |
| Colour-cue exploitation | Shortcut Sensitivity (high SS on affordance) |
| Shape-cue exploitation | Shortcut Sensitivity (high SS on tool_substitution) |
| Random exploration luck | Intervention Validity (low IV) + Curriculum Progression (flat slope) |
| Systematic causal reasoning | High IV + positive Transfer delta |

**Reporting requirements:**

- Report all metrics in the vector separately, not their mean.
- Report per-family breakdowns, not aggregate across families.
- Include standard deviation across seeds alongside mean values.
- Flag when CLASSIFY TSR=0 (expected; not a sign of agent failure in the current runner).

---

## Visualization and Reporting

The `crucible/viz/` sub-package provides matplotlib-based visualizations
(requires `pip install epistemic-crucible[notebooks]`):

- **`crucible.viz.world_graph`**, `plot_world(world)`: renders the 6×6 grid
  with objects coloured by `ObjectColor` and shaped by `ObjectShape` marker;
  inventory items shown in a sidebar.
- **`crucible.viz.grammar_tree`**, `plot_task_tree(spec)`: hierarchical tree of
  root (family/seed/split) → goal + constraints → object nodes.
- **`crucible.viz.traces`**, `plot_intervention_trace(steps)`: episode × step
  matrix where cells show intervention presence and causal effect.
- **`crucible.viz.heatmaps`**, three charts: shortcut-exposure bars (train vs
  test TSR per agent per family), train/test TSR recombination heatmap
  (rows=agents, cols=families), and per-agent failure-mode stacked bars.
- **`crucible.viz.reports`**, `generate_report(trace_path)`: generates a full
  markdown report with metric table, gaming risk notes, and embedded plot links.
  **No aggregate score is included.**

### One-command report

```bash
# Install optional dependencies
pip install epistemic-crucible[notebooks]

# Run baselines first to produce a trace
.venv/bin/python experiments/run_baselines.py

# Generate a report from the trace
.venv/bin/python experiments/generate_report.py \
    --traces results/*.jsonl \
    --output results/
```

Reports are written to `results/report_{timestamp}.md`.
Reports explicitly contain **no aggregate score**: all metrics in the vector are
listed separately with their definitions and gaming risk notes.

---

## Expected Baseline Results

Representative approximate Task Success Rates (TSR) for the affordance family with
20 seeds (≈17 train / ≈3 test), reported as mean across seeds. Exact values vary
with seed selection; the qualitative ordering is stable.

| Agent        | Train TSR | Test TSR | Notes                                       |
|--------------|-----------|----------|---------------------------------------------|
| random       | ~0.00     | ~0.00    | Undirected exploration; affordance requires directed action |
| heuristic    | ~0.53     | ~0.33    | Greedy visible-only; exploits colour-conductivity shortcut |
| memorization | ~0.00     | ~0.00    | Certificate lacks MOVE actions; oracle navigation gap |

**Reading the results:**
The heuristic's train/test gap (0.53 → 0.33) is the primary diagnostic signal:
the greedy visible-only policy benefits from the RED=conductive correlation that
only exists in the training split. On the decorrelated test split, performance
drops sharply, revealing shortcut reliance instead of genuine causal discovery.
Random exploration achieves ~0 because affordance tasks require directed action
(INSPECT/APPLY on specific objects). The `shortcut_sensitivity` metric and
perturbation suite provide stronger quantitative evidence of shortcut
exploitation.

**Reproducing these results:**
```bash
python experiments/run_baselines.py --config configs/quickstart.yaml
python experiments/generate_report.py --traces results/baselines_*.jsonl
```

The quickstart run completes in under 30 seconds on a laptop CPU. See
`docs/release_policy.md` for reproducibility guarantees.

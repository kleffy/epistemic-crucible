# Epistemic Crucible

> Build worlds in which high performance is impossible without discovering hidden causes, inventing interventions, revising false beliefs, and compressing those discoveries into abstractions that transfer across mutated descendants.

## What this is

A lightweight generative environment for evaluating whether agents acquire **reusable causal knowledge**, not task-specific policies.

**This is not a reward-only benchmark.** Task success is a training signal, not the evaluation target. The platform measures knowledge emergence: whether agents develop internal structures that survive intervention, contradiction, and transfer to held-out world variants.

## Core diagnostic loop

1. Agent encounters objects with visible features and hidden causal properties.
2. It must act, intervene, and observe consequences.
3. It must infer latent rules instead of memorising task layouts.
4. It must transfer inferred rules to recombined worlds.
5. It must survive adversarial perturbations that break superficial shortcuts.
6. It is evaluated on knowledge emergence, not reward alone.

## Installation

```bash
git clone https://github.com/kleffy/epistemic-crucible && cd epistemic-crucible
uv venv .venv && uv pip install -e ".[dev,notebooks]"
# or with pip: pip install -e ".[dev,notebooks]"
```

No GPU required. No remote API required. All standard code paths run offline on CPU.

## Quick start

```bash
# Verify installation
pytest tests/ -v -m "not slow"

# Run the affordance diagnostic (~20 seconds, 20 seeds × 3 agents)
python experiments/run_baselines.py --config configs/quickstart.yaml

# Generate a markdown report from the trace
python experiments/generate_report.py --traces results/baselines_*.jsonl

# Run the full baseline suite (~8 minutes, 20 seeds × 6 agents × 5 families)
python experiments/run_baselines.py --config configs/baselines.yaml
```

## CI test commands

```bash
pytest tests/ -v -m "not slow"   # fast suite (<5 s, excludes end-to-end runs)
pytest tests/ -v                  # full suite including end-to-end runs (~60 s)
ruff check .                      # lint
```

## Example results

Task Success Rates on the affordance family from the publication sweep
(`configs/publication.yaml`, 100 seeds, ~79 train / ~21 test). Full artifacts in
[`results/reference/`](results/reference/):

| Agent               | Train TSR | Test TSR | Shortcut sensitivity |
|---------------------|-----------|----------|----------------------|
| **neural_ppo** (BC) | 0.203     | 0.048    | **0.155** (highest)  |
| heuristic           | 0.443     | 0.429    | 0.014                |
| hybrid_rule_planner | 0.418     | 0.429    | −0.011               |
| random / tabular_rl / world_model / memorization | 0.000 | 0.000 | 0.000 |

The headline is the **learned** baseline: the behavior-cloned `neural_ppo` agent
inherits the train-only RED=conductive correlation and collapses on the
decorrelated test split (0.203 → 0.048), giving the highest `shortcut_sensitivity`
of any agent, a learning system that discovers the shortcut instead of the
hidden causal rule. (On affordance the symbolic heuristic happens to generalize;
its shortcut gap surfaces on `tool_substitution`, sensitivity 0.131.) The
`concept_reuse_proxy` metric is positive overall (0.203 vs 0.000): agents that
gather causal evidence before acting succeed on novel surface forms; those that
don't, fail.

The neural agent is GPU-capable (`pip install -e ".[gpu]"`); see
[`docs/benchmark_card.md`](docs/benchmark_card.md) for the training recipe and the
honest note on why this CPU-bound symbolic task does not benefit from a GPU.

### LLM panel (cross-provider)

LLMs are evaluated on a **decision-focused** layout (objects reachable, so the
task tests the causal decision instead of navigation) with a deliberately
neutral prompt and **anonymized object IDs so the split never leaks**. Open-weights
models run locally on the GPU (7–8B, and Qwen2.5-32B in 4-bit); Claude and GPT/o3
via API. Overall TSR across the four learnable families (full artifacts in
[`results/reference/llm/`](results/reference/llm/)):

| Model | Overall TSR [95% CI] |
|-------|----------------------|
| gpt-5.5 | **0.80** [0.68, 0.93] |
| qwen2.5-32b-instruct (4-bit) | 0.60 [0.54, 0.67] |
| claude-opus-4-8 | 0.58 [0.42, 0.72] |
| claude-sonnet-4-6 | 0.51 [0.45, 0.57] |
| o3 | 0.33 [0.17, 0.47] |
| claude-haiku-4-5 | 0.31 [0.25, 0.37] |
| llama-3.1-8b-instruct | 0.10 [0.06, 0.14] |
| qwen2.5-7b-instruct | 0.09 [0.06, 0.13] |

A capability gradient organized more by model scale or capacity than by weight
availability: the 7–8B open-weights models barely act on the task (wander and
inspect instead of discovering the apply-tool-to-gate intervention), a 32B open
model matches Opus, and the frontier models discover it unprompted. Even strong
models show surface-shortcut reliance: a dose-response probe shows a single
colour-correlated demonstration raises reliance on the spurious colour rule on
held-out worlds (0.20→0.57), and a hint ladder localizes whether a model's failure
is attribution or execution. Run with `pip install -e ".[llm]"` and
`python experiments/run_llm_eval.py`.

## Diagnostic experiment

Run the diagnostic headlessly to see where task success hides causal failure:

```bash
pip install -e .[notebooks]
python experiments/run_diagnostic.py
```

Results are written to `results/`. Open
`notebooks/00_crucible_diagnostic_experiment.ipynb` in Jupyter for interactive
plots: shortcut exposure, intervention traces, and failure mode breakdown.

## Documentation

- [docs/design.md](docs/design.md): architecture and design rationale
- [docs/benchmark_card.md](docs/benchmark_card.md): benchmark specification and baseline results
- [docs/metrics.md](docs/metrics.md): metric definitions and gaming risk analysis
- [docs/task_grammar.md](docs/task_grammar.md): task family grammar
- [docs/release_policy.md](docs/release_policy.md): versioning, reproducibility guarantees, citation

## Project layout

```
crucible/        core environment package
  agents/        baseline agent implementations
  viz/           visualization utilities (requires matplotlib)
  utils/         seeding, logging, serialization
configs/         YAML configuration files
tests/           pytest test suite
experiments/     experiment runner scripts
notebooks/       diagnostic notebooks
results/         output artefacts (not committed)
docs/            design and benchmark documentation
```

## Citation

```bibtex
@software{epistemic_crucible_2026,
  author = {Ayuba, Daniel La'ah},
  title  = {Epistemic Crucible: A Generative Developmental Benchmark
            for Causal Reasoning Evaluation},
  year   = {2026},
  note   = {v0.1},
}
```

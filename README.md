# Epistemic Crucible

> Success is not identification: cross the visible cue and operative mechanism to
> measure which one controls an agent's behavior.

## What this is

A lightweight generative environment for measuring **behavioral policy dependence**
under controlled world interventions.

**This is not a claim about internal causal representations.** v0.2 crosses the
carrier of a hidden mechanism with the carrier of a visible cue and records how an
agent's terminal choice responds. The frozen v0.1 task families remain available for
reproduction but are not the primary v0.2 measurement surface.

## v0.2 diagnostic loop

1. Agent encounters three tools with visible features and a hidden conductivity bit.
2. It chooses model-facing `QUERY(tool)` or `COMMIT(tool)` macro-actions; each is
   compiled into legal movement, pickup, and apply actions through the environment.
3. A queried tool is removed from later query choices; the first commit is terminal.
4. Each base world is evaluated in a paired 2x2 quartet crossing mechanism and cue.
5. Mechanism/cue tracking, paired responsiveness, unconditional and conditional cue
   susceptibility, evidence acquisition/use, coverage, and task success are reported
   separately, with no aggregate score.

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

# Run the fast v0.2 controls, or the full pre-registered integrity audit
python experiments/run_factorial_eval.py --controls
python experiments/run_integrity_gate.py
```

## CI test commands

```bash
pytest tests/ -v -m "not slow"   # fast suite (<5 s, excludes end-to-end runs)
pytest tests/ -v                  # full suite including end-to-end runs (~60 s)
ruff check .                      # lint
```

## Frozen v0.1 results (historical)

The results below accompany the frozen `v0.1-arxiv` tag. They are retained for
reproduction, not promoted as v0.2 evidence. See
[`docs/AUDIT-v0.1.md`](docs/AUDIT-v0.1.md) for the construct and reporting audit.

Task Success Rates on the affordance family from the publication sweep
(`configs/publication.yaml`, 100 seeds, ~79 train / ~21 test). Full artifacts in
[`results/reference/`](results/reference/):

| Agent               | Train TSR | Test TSR | Shortcut sensitivity |
|---------------------|-----------|----------|----------------------|
| **neural_ppo** (legacy name; BC) | 0.203 | 0.048 | **0.155** (highest) |
| heuristic           | 0.443     | 0.429    | 0.014                |
| hybrid_rule_planner | 0.418     | 0.429    | −0.011               |
| random / tabular_rl / world_model / memorization | 0.000 | 0.000 | 0.000 |

These values are descriptive legacy task-success results. They do not identify
whether a policy follows the hidden mechanism, a visible cue, or an availability
heuristic, and v0.2 makes no causal-knowledge inference from them.

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

The table is preserved only to reproduce v0.1. Its former `0.57` colour-reliance
headline equals the observed red-tool availability rate (`0.57/0.60`), so it is
not evidence of saturation or cue-controlled policy. The v0.1 record also omitted
that the inference path allowed 3,000 completion tokens, used low reasoning effort,
retained nine messages (not nine turns), and trained the released neural checkpoint
with behavior cloning rather than PPO. v0.2 corrects those points explicitly and
does not carry this leaderboard or the four unrepaired families into its primary
claims.

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
- [docs/v0.2-protocol.md](docs/v0.2-protocol.md): crossed protocol, metrics, controls, and staged study
- [docs/ICLR2027_SUBMISSION_CHECKLIST.md](docs/ICLR2027_SUBMISSION_CHECKLIST.md): verified deadlines and submission operations

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
  title  = {Success Is Not Identification: Crossed World Interventions
            for Attributing Agent Policies},
  year   = {2026},
  note   = {software protocol v0.2; v0.1-arxiv remains frozen},
}
```

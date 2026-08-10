# Release Policy

## Scope

Epistemic Crucible is an open research tool for studying causal and counterfactual
reasoning in agents. It is a **micro-benchmark**, a 6×6 grid world with five task
families, intended for controlled diagnostic studies, not competitive leaderboarding.

This document describes the commitments made to researchers who build on this codebase.

---

## Versioning

The project follows [Semantic Versioning](https://semver.org/).

- **v0.1.x**, prototype series. Core API (environment, grammar, metrics, agents) is
  functional and tested, but may evolve between minor releases.
- **v0.2.x**, crossed behavioral-attribution protocol. Condition axes are independent
  of the frozen legacy train/test split, and trace schema `0.2` is self-describing.
- **v0.x.y**, MINOR bumps may break existing experiment scripts if the JSONL trace
  schema changes. PATCH bumps are backwards-compatible.
- **v1.0.0**, signals stable API and frozen trace schema.

### Frozen benchmark surface (v0.1)

The `v0.1-arxiv` tag is the **frozen baseline** accompanying the arXiv v1 of the
benchmark paper. The following surface is held fixed for the v0.1 line, so that
agents developed against it (including ones reported in later papers) are evaluated
on an unchanged benchmark. `docs/FREEZE-v0.1.md` lists the checklist and the sha256
hashes of the reference artifacts.

- **Task grammar and families**, the five families and their goal predicates.
- **Split policies**, feature, counterfactual, rule, and composition holdouts, and
  the 80/20 train/test seed assignment drawn from a separate RNG stream.
- **Seed scheme**, deterministic `generate_task(family, seed, split)`.
- **Metric vector**, the nine metrics plus the perturbation shortcut-exposure score,
  with no aggregate.
- **Legal-action serialization**, index selection over the legal-only candidate list.
- **Public-observation boundary**, hidden properties never enter the observation.

Bug-fix PATCH releases on the v0.1 line are allowed, but must not change generated
worlds, the metric definitions, or the trace schema.

### v0.2 protocol surface

v0.2 is a new minor release and does not rewrite the frozen `v0.1-arxiv` surface.
Its primary surface consists of the balanced affordance quartet, `QUERY`/`COMMIT`
macro-actions compiled into legal environment steps, trace schema `0.2`, and the
unaggregated attribution metric vector. The four other legacy families remain
available for reproduction but are excluded from primary v0.2 claims.

Pilot and confirmatory manifests are disjoint. After the confirmatory manifest is
frozen, any integrity defect requires rerunning every affected condition; scientific
results cannot waive an integrity failure.

---

## Reproducibility Guarantees

The following properties hold within a single release line:

1. **Deterministic world generation**: `generate_task(family, seed, split)` always
   returns the same `TaskSpec`. `build_world_from_spec(spec)` always produces the
   same `WorldState`. These are invariant across machines with the same Python version.

2. **Deterministic baselines**: All standard agents (random, heuristic, memorization,
   tabular_rl, world_model, hybrid_rule_planner) are seeded and produce the same action
   sequences given the same task seed.

3. **Self-describing traces**: JSONL trace files embed `family`, `seed`, `split`,
   and `agent` in every record. A trace can be re-evaluated in isolation without
   access to the config that produced it.

4. **Determinism test**: `pytest tests/test_generator_determinism.py -v` verifies
   bit-for-bit world-state reproducibility. This test must pass before any release.

---

## Compute Requirements

- **No GPU required.** All standard code paths run on CPU.
- **Quickstart** (`configs/quickstart.yaml`): completes in under 30 seconds on a
  laptop CPU (20 seeds × 3 agents × 1 family).
- **Full baseline suite** (`configs/baselines.yaml`): completes in under 10 minutes
  on a laptop CPU (20 seeds × 6 agents × 5 families).
- **Memory**: under 512 MB for all standard runs.

---

## API Requirements

- **No remote API required** for standard evaluation. All of the following work
  fully offline: environment, grammar, baselines (random, heuristic, memorization,
  tabular_rl, world_model, hybrid_rule_planner), metrics, perturbations, and
  visualization.
- **`LLMAgent`** (`crucible/agents/llm_agent.py`) requires an Anthropic or OpenAI
  API key. It is explicitly excluded from `configs/baselines.yaml` and from all
  standard experiment runners. Evaluations submitted to the benchmark must disclose
  whether an LLM agent was used.

---

## No Single Score

The Epistemic Crucible does not produce a leaderboard score or aggregate metric.

- Results are reported as a **vector of 9 metrics** (TSR, Transfer Success, Shortcut
  Sensitivity, Intervention Validity, Intervention Efficiency, Counterfactual Accuracy,
  Concept Reuse Proxy, Failure Diversity, Curriculum Progression), plus the
  perturbation-based Shortcut Exposure Score.
- Each metric has a documented gaming risk. See `docs/metrics.md`.
- The report generator (`experiments/generate_report.py`) explicitly omits any
  aggregate score and includes gaming risk notes for every metric.

Researchers should report **all metrics**, per-family breakdowns, and standard
deviation across seeds. Cherry-picking individual metrics is a methodological error.

---

## Trace Format Stability

The JSONL step and outcome record schemas are stable within v0.1.x:

**Step record keys** (stable): `kind`, `episode`, `seed`, `family`, `split`, `agent`,
`step`, `action` (`kind`, `args`), `effects`, `legal`, `energy`, `done`.

**Outcome record keys** (stable): `kind`, `episode`, `seed`, `family`, `split`,
`agent`, `goal_achieved`, `steps`, `interventions`, `energy_remaining`,
`illegal_rate`, `unique_effects`.

Additional keys may be added in later versions without breaking parsers that use
`dict.get()`. Removing or renaming existing keys is a MINOR version change.

---

## Citation

For the v0.2 protocol, please cite:

```bibtex
@software{epistemic_crucible_2026,
  author = {Ayuba, Daniel La'ah},
  title  = {Success Is Not Identification: Crossed World Interventions
            for Attributing Agent Policies},
  year   = {2026},
  note   = {software protocol v0.2; v0.1-arxiv remains frozen},
}
```

---

## Contributing

Contributions are welcome via pull requests.

**Guidelines:**
- New task families must include a validity test (`validate_task` returns `[]`) and a
  determinism test (same seed → identical `TaskSpec`).
- New metrics must document their gaming risk.
- No new required dependencies. Optional dependencies belong in
  `[project.optional-dependencies]` in `pyproject.toml`.
- All PRs must pass `pytest tests/ -v` and `ruff check .` before review.
- Do not add a leaderboard, aggregate score, or single-number ranking.

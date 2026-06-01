# Epistemic Crucible: Design

## Thesis

A properly designed developmental world can reveal intelligence gaps that ordinary reward benchmarks hide.

The prototype tests one narrow claim:

> Reward success on generated tasks can systematically hide causal failure. Paired counterfactual worlds and intervention-valid transfer expose the failure in a lightweight developmental environment.

## What this is not

- Not a reward-only benchmark.
- Not a fixed task suite with a scalar leaderboard.
- Not a collection of mini-games sharing no latent structure.

## Developmental pressures

The prototype implements a targeted subset of pressures, each mapped to a concrete mechanism:

| Pressure | Mechanism | Capability tested |
|---|---|---|
| Sensorimotor | Discrete movement, pickup, placement | Grounded action selection |
| Object | Persistent, occludable, combinable entities | Object permanence and tracking |
| Affordance | Same object, different effect depending on context | Discovering usable properties |
| Causal | Hidden rules govern object transformations | Causal model learning |
| Counterfactual | Matched worlds differ by one latent variable | Counterfactual prediction |
| Scarcity | Limited attempts, energy, or samples | Experiment efficiency |
| Compositional | Goals require combining mechanisms | Multi-step abstraction |
| Generalisation | Test worlds recombine held-out features/rules | Transfer beyond memorisation |
| Contradiction | Old rule sometimes fails due to hidden modifier | Belief revision |
| Deception | Spurious visual correlations injected | Shortcut resistance |
| Tool-use | Objects modify other objects or reveal hidden state | Instrumental tool use |
| Self-verification | Agent can run diagnostic interventions before acting | Experiment design and falsification |

## Evaluation unit

The elementary evaluation unit is an **epistemic bundle**:

```
Bundle = (solve, intervene, predict_counterfactual, transfer, retain)
```

Credit is only meaningful when improvement propagates across the bundle.

## Hidden rule engine

The hidden rule engine adds deterministic latent rule templates to every generated world. The
initial library covers conductive gate opening, affinity-based gate exceptions,
source activation of magnetic keys, magnetic key gate opening, soluble block
transformation, detector-based public signal marking, source+catalyst token
production, and delayed hazard consequences.

Rules are evaluator-visible only. Agent observations expose public state changes,
markers, relations, resources, and object states, but never hidden properties,
rule IDs, pending effects, or hidden precondition names. Episode traces may carry
causal provenance for metrics and debugging; that provenance is not part of the
agent-facing observation.

## Metrics (summary)

- Task success
- Transfer success
- Shortcut sensitivity
- Intervention validity
- Counterfactual accuracy
- Concept reuse proxy
- Failure diversity
- Curriculum progression

See `docs/metrics.md` for formal definitions.

# Epistemic Crucible

**Does an LLM agent act on relevant evidence, or on a cue that happened to predict success?**

Epistemic Crucible is a controlled testbed for empirical AI-safety research. It
varies visible cues and hidden task mechanisms independently, then measures how
an agent's evidence gathering and decisions respond.

The safety question is whether successful behaviour is enough to justify trust.
An agent may choose the right action while relying on a superficial correlation.
Separating these explanations matters when evaluations are used to support
claims about reliable agents and oversight.

## How the assay works

The agent must choose a tool that opens a gate. Conductivity determines which
tool works; colour provides a visible cue.

1. **Acquire evidence:** `QUERY(tool)` returns a detector signal about conductivity.
2. **Choose an action:** `COMMIT(tool)` makes a terminal choice, whether correct or
   incorrect.
3. **Cross the conditions:** change which tool is conductive independently of
   which tool carries the cue, holding other features fixed within a paired world.

The base design has four conditions (2×2). The confirmatory study uses the
prespecified six-condition challenge (3×2). Scripted controls calibrate the assay
against known policies, including a detector user, a cue follower and an oracle.
Evidence acquisition, cue following, mechanism tracking and task success are
reported separately.

## Frozen v0.2 findings

The confirmatory study evaluates **Qwen3.6-27B** and
**Mistral-Small-3.2-24B** across 64 paired base seeds, five fixed prompt arms and
six conditions: **1,920 episode outcomes per model**.

**Detector-query rate**

| Model | Neutral prompt | Cue-demonstration A | Cue-demonstration B |
|---|---:|---:|---:|
| Qwen3.6-27B | 99.22% | 0% | 0% |
| Mistral-Small-3.2-24B | 100% | 0% | 0% |

In both fixed cue-demonstration arms, both models followed the visible cue in
every episode. These results show context-dependent evidence acquisition and
cue reliance in this assay, despite the models' ability to use the detector
under neutral prompts.

The [frozen results](results/reference/v02/confirmatory/) include all five arms,
paired contrasts, denominators, pinned model revisions and source hashes.
Uncertainty is estimated over **paired base seeds**, not independent episodes.

**Scope:** these are behavioural findings about two models under fixed prompts
in a small, deterministic environment. They do not establish internal causal
representations, deliberate deception or the effectiveness of deployed AI-control
methods. Extending the design to monitors or auditors is future work.

## Run a CPU diagnostic

Python 3.10 or later. The scripted controls run offline after installation and
require no GPU, model weights or API key.

```bash
git clone https://github.com/kleffy/epistemic-crucible.git
cd epistemic-crucible
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,notebooks]"
python experiments/run_factorial_eval.py --controls
```

This exercises the base 2×2 controls and writes traces plus
`factorial_control_summary.json` to `results/factorial_v02/`.
Run the CPU test suite with `python -m pytest tests/ -v -m "not gpu"`.

Repeating the LLM study requires the pinned models, compatible GPU serving and a
clean checkout at the frozen protocol or its declared freeze-wrapper commit.
Start with the [v0.2 protocol](docs/v0.2-protocol.md) and
[confirmatory manifest](configs/confirmatory_manifest.json); the
[model configuration](configs/factorial_v02.yaml) records serving settings.
Live acquisition and cache-only replay are separate modes. The public result
package contains summaries and provenance; raw confirmatory traces, caches,
model weights and the SquashFS seal are excluded.

## Research status

This repository contains the v0.2 protocol, evaluation code and frozen
confirmatory results. The earlier cross-provider pilot and five-family v0.1
results are historical records. An availability confound prompted the redesign;
the [v0.1 audit](docs/AUDIT-v0.1.md) explains the corrections. Those earlier
results are outside the primary v0.2 claims.

## Citation and licence

Use [CITATION.cff](CITATION.cff) to cite the software:
*Success Is Not Identification: Crossed World Interventions for Attributing Agent
Policies* — Daniel La'ah Ayuba, 2026.

Released under the [MIT licence](LICENSE).

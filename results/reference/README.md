# Reference results

Reproducible benchmark artifacts from the publication-scale sweep
(`configs/publication.yaml`: all 7 agents × 5 families × 100 seeds, ~80/20
train/test auto-split). Generated on an NVIDIA RTX 5090 (sm_120) host; the
neural baseline is trained deterministically on CPU (see note below).

## Headline result

A *learned* neural agent exhibits the largest shortcut reliance of any baseline
on the affordance family, it inherits the train-only colour shortcut and
collapses on the decorrelated test split:

| Agent (affordance) | Train TSR | Test TSR | Shortcut sensitivity (train−test) |
|--------------------|-----------|----------|-----------------------------------|
| neural_ppo (BC)    | 0.203     | 0.048    | **0.155** (highest)               |
| heuristic          | 0.443     | 0.429    | 0.014                             |
| hybrid_rule_planner| 0.418     | 0.429    | −0.011                            |
| random / tabular_rl / world_model / memorization | 0.000 | 0.000 | 0.000 |

`concept_reuse_proxy` (all families, test split): test TSR **0.203** with prior
evidence vs **0.000** without, agents that perform informative interventions
before acting succeed; those that do not, fail.

See `metrics_report.json` for the full 9-metric vector and `report.md` for the
generated diagnostic report. Figures: `shortcut_exposure.png`,
`recombination_heatmap.png`, `failure_map.png`, `intervention_trace.png`.

## Files

- `affordance.pt`: the trained neural baseline checkpoint (behavior-cloned).
- `train_neural_summary.json`: training summary (BC accuracy, checkpoint path).
- `baselines_summary.json`: per-(family, agent) success rates from the sweep.
- `metrics_report.json`: full diagnostic metric vector.
- `report.md`, `*.png`: generated report and figures.

## Reproduce

```bash
# 1. Train the neural baseline (deterministic, CPU, ~50s)
#    Writes the checkpoint to results/checkpoints/affordance.pt
python experiments/train_neural.py --config configs/neural.yaml

# 2. Run the full sweep (~40 min, CPU)
python experiments/run_baselines.py --config configs/publication.yaml

# 3. Regenerate the report + figures
python experiments/generate_report.py --traces results/baselines_*.jsonl
```

To skip retraining and run the sweep against **this committed checkpoint**, set
`checkpoint_dir: results/reference` in `configs/publication.yaml` (the runner
loads `<checkpoint_dir>/<family>.pt`, i.e. `results/reference/affordance.pt`).

## Determinism note

Symbolic-agent results are fully deterministic (seeded RNG). The neural
checkpoint is trained with behavior cloning on **CPU**, which is bit-reproducible
for this workload, and faster than GPU here, because the task is CPU-bound
(Python env rollouts) with a tiny policy. CUDA training is supported
(`device: cuda`) and was used during development, but has minor run-to-run
variance from nondeterministic GPU atomics; the released checkpoint reproduces
the reported eval numbers exactly. Greedy evaluation of any fixed checkpoint is
deterministic. The raw ~29 MB JSONL trace is not committed; regenerate it with
the commands above.

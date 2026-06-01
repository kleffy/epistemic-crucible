# Epistemic Crucible Evaluation Report

**No aggregate score is reported.** The metric vector below must be read as a whole.

- **Trace**: `baselines_20260530_041349.jsonl`
- **Generated**: 2026-05-30T07:28:59

## Metric Results

| Metric | Value | Count | Definition | Gaming Risk |
| --- | --- | --- | --- | --- |
| `task_success_rate` | 0.1297 | 3100 | Fraction of episodes with goal_achieved=True. | Agents can maximise TSR on OPEN-goal families by memorising seed→action maps wit… |
| `transfer_success` | {"affordance/heuristic": {"train_tsr": 0.443, "test_tsr": 0.4286, "delta": -0.01… | 3100 | Per-(family,agent) train TSR vs test TSR; delta=test-train. | Positive delta from lucky seeds is not reliable; report standard deviation acros… |
| `shortcut_sensitivity` | {"affordance/heuristic": 0.0145, "affordance/hybrid_rule_planner": -0.0108, "aff… | 1300 | train_TSR - test_TSR for affordance and tool_substitution families. High value = surface-feature exploitation. | Memorisation agents show SS≈0 trivially because they use different lookup keys p… |
| `intervention_validity` | 1.0000 | 22697 | Fraction of APPLY/COMBINE/INSPECT steps with non-empty effect strings. | INSPECT on a DETECTOR object always produces a marker effect; excluding INSPECT … |
| `intervention_efficiency` | — | 0 | For successful episodes: oracle_action_count / agent_steps, capped at 1.0. Higher = more efficient relative to the oracle. | Short lucky episodes score higher than systematic solvers; efficiency only appli… |
| `counterfactual_accuracy` | — | 0 | Counterfactual accuracy: behavioral = correct intervention on classified object; prediction = predicted effect matches actual effect. | Agents can achieve high behavioral CA on COUNTERFACTUAL by always applying to bl… |
| `concept_reuse_proxy` | {"with_evidence_tsr": 0.2033, "without_evidence_tsr": 0.0, "concept_reuse": 0.20… | 651 | Test-split TSR with prior evidence minus test-split TSR without prior evidence. Evidence = an informative INSPECT/APPLY intervention in-episode. | Inflated if hidden-rule structure leaks through object types/names, or by an age… |
| `failure_diversity` | {"timeout": 1017, "no_interaction": 780, "high_illegal": 317, "no_effects": 0, "… | 2698 | Count of distinct failure modes across all failed episodes. | Failure diversity does not penalise agents that fail identically in predictable … |
| `curriculum_progression` | {"affordance/heuristic": {"windows": [0.6, 0.4, 0.4, 0.6, 0.2, 0.0, 0.6, 0.4, 0.… | 3100 | Rolling TSR over seed-ordered windows within each agent+family run. Positive slope indicates improvement; negative indicates forgetting. | Slope is statistically meaningless for random agents or small seed counts; requi… |

## Gaming Risk Notes

**`task_success_rate`**: Agents can maximise TSR on OPEN-goal families by memorising seed→action maps without causal understanding.

**`transfer_success`**: Positive delta from lucky seeds is not reliable; report standard deviation across seeds alongside the mean.

**`shortcut_sensitivity`**: Memorisation agents show SS≈0 trivially because they use different lookup keys per split; SS must be read alongside TSR.

**`intervention_validity`**: INSPECT on a DETECTOR object always produces a marker effect; excluding INSPECT yields a stricter causal-intervention validity metric.

**`intervention_efficiency`**: Short lucky episodes score higher than systematic solvers; efficiency only applies to successful episodes.

**`counterfactual_accuracy`**: Agents can achieve high behavioral CA on COUNTERFACTUAL by always applying to block0 (which is correct on TRAIN); this fails on TEST where the property is swapped.

**`concept_reuse_proxy`**: Inflated if hidden-rule structure leaks through object types/names, or by an agent that always intervenes trivially; read with intervention_validity.

**`failure_diversity`**: Failure diversity does not penalise agents that fail identically in predictable ways; high diversity is not inherently desirable.

**`curriculum_progression`**: Slope is statistically meaningless for random agents or small seed counts; require at least 3 windows before interpreting trend direction.

## Visualizations

### Intervention Trace

![Intervention Trace](report_20260530_072856_539606_intervention_trace.png)

### Shortcut Exposure

![Shortcut Exposure](report_20260530_072856_539606_shortcut_exposure.png)

### Recombination Heatmap

![Recombination Heatmap](report_20260530_072856_539606_recombination_heatmap.png)

### Failure Map

![Failure Map](report_20260530_072856_539606_failure_map.png)

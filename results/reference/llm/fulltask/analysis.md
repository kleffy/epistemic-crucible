# Head-to-head results

## Capability gradient — overall TSR (all families, decision-focused)

| Agent | Overall TSR [95% CI] | n |
| --- | --- | --- |
| gpt-5.5 | 0.800 [0.68,0.93] | 40 |
| qwen2.5-32b-instruct | 0.650 [0.50,0.80] | 40 |
| claude-sonnet-4-6 | 0.475 [0.33,0.62] | 40 |
| claude-opus-4-8 | 0.450 [0.30,0.60] | 40 |
| claude-haiku-4-5-20251001 | 0.300 [0.15,0.45] | 40 |
| o3 | 0.300 [0.17,0.45] | 40 |
| llama-3.1-8b-instruct | 0.275 [0.15,0.42] | 40 |
| qwen2.5-7b-instruct | 0.025 [0.00,0.07] | 40 |

## Per-family TSR (train → test)

| Agent | affordance | causal_gate | contradiction | tool_substitution |
| --- | --- | --- | --- | --- |
| gpt-5.5 | 1.00→1.00 | 0.40→0.00 | 1.00→1.00 | 1.00→1.00 |
| qwen2.5-32b-instruct | 1.00→0.60 | 0.00→0.00 | 1.00→0.80 | 1.00→0.80 |
| claude-sonnet-4-6 | 0.20→0.40 | 0.40→0.20 | 1.00→1.00 | 0.20→0.40 |
| claude-opus-4-8 | 0.40→0.40 | 0.40→0.00 | 1.00→0.80 | 0.20→0.40 |
| claude-haiku-4-5-20251001 | 0.00→0.00 | 0.00→0.00 | 1.00→1.00 | 0.20→0.20 |
| o3 | 0.20→0.20 | 0.00→0.00 | 1.00→0.40 | 0.00→0.60 |
| llama-3.1-8b-instruct | 0.00→0.20 | 0.00→0.00 | 0.80→0.40 | 0.60→0.20 |
| qwen2.5-7b-instruct | 0.00→0.00 | 0.00→0.00 | 0.20→0.00 | 0.00→0.00 |

## Affordance: TSR (mean [95% CI]) and colour reliance

| Agent | Train TSR | Test TSR | Gap | Train red-apply | Test red-apply |
| --- | --- | --- | --- | --- | --- |
| claude-haiku-4-5-20251001 | 0.00 [0.00,0.00] | 0.00 [0.00,0.00] | +0.00 | 0.0 | 0.0 |
| claude-opus-4-8 | 0.40 [0.00,0.80] | 0.40 [0.00,0.80] | +0.00 | 0.2 | 0.4 |
| claude-sonnet-4-6 | 0.20 [0.00,0.60] | 0.40 [0.00,0.80] | -0.20 | 0.25 | 0.2 |
| gpt-5.5 | 1.00 [1.00,1.00] | 1.00 [1.00,1.00] | +0.00 | 0.4 | 0.0 |
| llama-3.1-8b-instruct | 0.00 [0.00,0.00] | 0.20 [0.00,0.60] | -0.20 | 0.0 | 1.0 |
| o3 | 0.20 [0.00,0.60] | 0.20 [0.00,0.60] | +0.00 | 0.5 | 0.0 |
| qwen2.5-32b-instruct | 1.00 [1.00,1.00] | 0.60 [0.20,1.00] | +0.40 | 1.0 | 0.2 |
| qwen2.5-7b-instruct | 0.00 [0.00,0.00] | 0.00 [0.00,0.00] | +0.00 | — | — |

*Red-apply = fraction of episodes whose first apply-to-gate used a red tool. On TRAIN red is conductive (correct); on TEST red is a decoy (shortcut).*

# Head-to-head results

## Capability gradient — overall TSR (all families, decision-focused)

| Agent | Overall TSR [95% CI] | n |
| --- | --- | --- |
| gpt-5.5 | 0.800 [0.68,0.93] | 40 |
| qwen2.5-32b-instruct | 0.604 [0.54,0.67] | 240 |
| claude-opus-4-8 | 0.575 [0.42,0.72] | 40 |
| claude-sonnet-4-6 | 0.512 [0.45,0.57] | 240 |
| o3 | 0.325 [0.17,0.47] | 40 |
| claude-haiku-4-5-20251001 | 0.312 [0.25,0.37] | 240 |
| llama-3.1-8b-instruct | 0.100 [0.06,0.14] | 240 |
| qwen2.5-7b-instruct | 0.092 [0.06,0.13] | 240 |

## Per-family TSR (train → test)

| Agent | affordance | causal_gate | contradiction | tool_substitution |
| --- | --- | --- | --- | --- |
| gpt-5.5 | 1.00→0.80 | 0.40→0.20 | 1.00→1.00 | 1.00→1.00 |
| qwen2.5-32b-instruct | 0.70→0.53 | 0.00→0.00 | 1.00→0.90 | 1.00→0.70 |
| claude-opus-4-8 | 0.40→0.60 | 0.60→0.20 | 1.00→0.80 | 0.40→0.60 |
| claude-sonnet-4-6 | 0.30→0.40 | 0.20→0.10 | 1.00→0.97 | 0.57→0.57 |
| o3 | 0.00→0.20 | 0.00→0.00 | 0.80→0.80 | 0.40→0.40 |
| claude-haiku-4-5-20251001 | 0.17→0.17 | 0.10→0.00 | 0.80→0.67 | 0.23→0.37 |
| llama-3.1-8b-instruct | 0.10→0.03 | 0.00→0.00 | 0.17→0.07 | 0.30→0.13 |
| qwen2.5-7b-instruct | 0.13→0.03 | 0.00→0.00 | 0.20→0.13 | 0.07→0.17 |

## Affordance: TSR (mean [95% CI]) and colour reliance

| Agent | Train TSR | Test TSR | Gap | Train red-apply | Test red-apply |
| --- | --- | --- | --- | --- | --- |
| claude-haiku-4-5-20251001 | 0.17 [0.03,0.30] | 0.17 [0.03,0.30] | +0.00 | 0.0667 | 0.25 |
| claude-opus-4-8 | 0.40 [0.00,0.80] | 0.60 [0.20,1.00] | -0.20 | 0.2 | 0.2 |
| claude-sonnet-4-6 | 0.30 [0.13,0.47] | 0.40 [0.23,0.57] | -0.10 | 0.2759 | 0.1667 |
| gpt-5.5 | 1.00 [1.00,1.00] | 0.80 [0.40,1.00] | +0.20 | 0.6 | 0.4 |
| llama-3.1-8b-instruct | 0.10 [0.00,0.20] | 0.03 [0.00,0.10] | +0.07 | 0.75 | 0.6 |
| o3 | 0.00 [0.00,0.00] | 0.20 [0.00,0.60] | -0.20 | 0.0 | 0.5 |
| qwen2.5-32b-instruct | 0.70 [0.53,0.87] | 0.53 [0.33,0.70] | +0.17 | 0.7241 | 0.2692 |
| qwen2.5-7b-instruct | 0.13 [0.03,0.27] | 0.03 [0.00,0.10] | +0.10 | 0.8 | 0.5 |

*Red-apply = fraction of episodes whose first apply-to-gate used a red tool. On TRAIN red is conductive (correct); on TEST red is a decoy (shortcut).*


## Few-shot in-context shortcut probe (affordance, held-out split)

K=3 solved TRAIN demos (conductive tool is RED), then evaluated on TEST worlds where colour is decorrelated. Higher = more reliance on the spurious colour rule.

| Agent | Zero-shot red-apply | Few-shot red-apply | Δ |
| --- | --- | --- | --- |
| claude-opus-4-8 | 0.2 | 0.6 | +0.40 |
| claude-sonnet-4-6 | 0.1667 | 0.6 | +0.43 |
| gpt-5.5 | 0.4 | 0.6 | +0.20 |

## Dose-response: in-context shortcut vs demo count K

TEST-split colour reliance (fraction of first apply-to-gate using a red decoy tool) after K demonstrations. *cue* demos make red the conductive tool; *mechanistic* demos show discovery with no colour regularity; *anti-cue* demos vary the correct tool's colour.


**cue**

| K | Test reliance | Test TSR | Train reliance | n apply (test) |
| --- | --- | --- | --- | --- |
| 0 | 0.2 | 0.47 | 0.3103 | 30 |
| 1 | 0.5667 | 0.37 | 1.0 | 30 |
| 3 | 0.5667 | 0.40 | 1.0 | 30 |
| 5 | 0.5667 | 0.37 | 1.0 | 30 |
| 10 | 0.5667 | 0.47 | 1.0 | 30 |

**mechanistic**

| K | Test reliance | Test TSR | Train reliance | n apply (test) |
| --- | --- | --- | --- | --- |
| 0 | 0.2 | 0.47 | 0.3103 | 30 |
| 1 | 0.3 | 0.73 | 0.1667 | 30 |
| 3 | 0.2333 | 0.57 | 0.1667 | 30 |
| 5 | 0.2 | 0.67 | 0.2667 | 30 |
| 10 | 0.0667 | 0.57 | 0.2 | 30 |

**anticue**

| K | Test reliance | Test TSR | Train reliance | n apply (test) |
| --- | --- | --- | --- | --- |
| 0 | 0.2 | 0.47 | 0.3103 | 30 |
| 1 | 0.0667 | 0.53 | 0.0 | 30 |
| 3 | 0.1304 | 0.23 | 0.1667 | 23 |
| 5 | 0.1 | 0.30 | 0.1071 | 30 |
| 10 | 0.1724 | 0.33 | 0.2222 | 29 |

## Oracle ladder: where failure is localized

Task success when a labelled hint is injected: *intervention* names the operative action, *property* names the tool that works, *rule* states the full local rule. Train→test per family.


**affordance**

| Agent | intervention | property | rule |
| --- | --- | --- | --- |
| claude-haiku-4-5-20251001 | 0.33→0.33 | 0.27→0.20 | 0.33→0.40 |
| claude-sonnet-4-6 | 0.33→0.53 | 1.00→1.00 | 1.00→1.00 |
| qwen2.5-32b-instruct | 0.30→0.40 | 0.90→0.90 | 0.90→0.90 |

**tool_substitution**

| Agent | intervention | property | rule |
| --- | --- | --- | --- |
| claude-haiku-4-5-20251001 | 0.33→0.53 | 0.47→0.53 | 0.33→0.40 |
| claude-sonnet-4-6 | 0.87→0.67 | 1.00→1.00 | 1.00→1.00 |
| qwen2.5-32b-instruct | 1.00→0.70 | 1.00→1.00 | 0.80→1.00 |
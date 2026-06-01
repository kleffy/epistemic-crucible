# LLM evaluation: reference results

A cross-provider panel evaluated on the **decision-focused** Epistemic Crucible
(compact layout; objects reachable so the task tests the causal decision, not
long-horizon navigation). Evaluation is **genuine**: a neutral system prompt
(no solution recipe, no anti-shortcut coaching), object IDs anonymized so the
split never leaks, index-based action selection over legal candidates, greedy
decoding, paired train/test splits. Open-weights models run locally on an RTX
5090 (32B in 4-bit nf4); Claude via the Anthropic API; GPT/o3 via the OpenAI
API. Seeds: 30 per split for the 7-8B/32B open-weights models, Haiku, and
Sonnet; 5 for Opus, gpt-5.5, and o3.

## Headline: capability gradient (overall TSR, all 4 families)

| Model | Overall TSR [95% CI] | N |
|-------|----------------------|---|
| gpt-5.5 | **0.800** [0.68, 0.93] | 40 |
| qwen2.5-32b-instruct | 0.604 [0.54, 0.67] | 240 |
| claude-opus-4-8 | 0.575 [0.42, 0.72] | 40 |
| claude-sonnet-4-6 | 0.512 [0.45, 0.57] | 240 |
| o3 | 0.325 [0.17, 0.47] | 40 |
| claude-haiku-4-5 | 0.312 [0.25, 0.37] | 240 |
| llama-3.1-8b-instruct | 0.100 [0.06, 0.14] | 240 |
| qwen2.5-7b-instruct | 0.092 [0.06, 0.13] | 240 |

A sharp gradient: the 7-8B open-weights models barely act on the task (they
wander and inspect instead of discovering the apply-tool-to-gate intervention),
while frontier models discover it on their own. A **32B open-weights model
closes most of the gap**, matching Opus within overlapping intervals, while
keeping the train-to-test transfer signature (affordance 0.70→0.53,
tool_substitution 1.00→0.70) and failing the causal-gate chain entirely
(0.00→0.00). In this panel, scale or capacity separates the open floor from the
frontier more than weight availability does (with a single 32B open model, scale is
confounded with family, recipe, instruction tuning, and quantization). Full
per-family breakdown and shortcut analysis in `analysis.md`;
figures in `tsr_train_vs_test.png` and `dose_response.png`.

## Shortcut reliance and the dose-response curve

`analysis.md` reports the colour-reliance metric (red-tool apply-to-gate) per
split. Two in-context probes show that the colour shortcut is installed by the
*content* of the demonstrations, not by their mere presence:

- **Few-shot probe** (`fewshot_traces/`): priming with 3 solved TRAIN demos
  (where RED is conductive) raises TEST-world reliance on the spurious colour
  rule (0.2→0.6 for Sonnet/Opus, 0.4→0.6 for gpt-5.5).
- **Dose-response** (`dose/`): Sonnet on affordance, TEST split, across demo
  count K and demo *mode*. **cue** demos (red is conductive) raise reliance
  0.20→0.57 and it saturates at K=1; **mechanistic** demos (discovery, no colour
  regularity) hold reliance low (0.20→0.07) while raising task success;
  **anti-cue** demos (correct tool's colour varies) suppress it to ~0.07-0.17.
  The shortcut is a property of how the cue appears in the examples.

## Oracle ladder (the paper's "hint ladder"): where failure is localized

The `oracle/` dir and the `--oracle` flag implement what the paper calls the **hint
ladder** (the rungs are natural-language hints, not enforced constraints, so the
ladder need not be monotonic). `oracle/` holds an ablation ladder (Sonnet, Haiku at
15 seeds/split; Qwen2.5-32B at 10; affordance and tool_substitution) that injects a
labelled hint and reads off which stage was the bottleneck. Naming the operative *intervention* lifts Sonnet
only modestly (affordance 0.33→0.53), but naming the *property* (which tool
works) takes it to a perfect 1.00→1.00 on both families: Sonnet's failure is in
**causal attribution**, and it executes flawlessly once handed the attribution.
Qwen2.5-32B patterns with Sonnet, lifting to 0.90 and 1.00 once the property
carrier is named, so its residual failure is attribution too. Haiku stays near 0.40
even when given the *full rule*, so its failure is more fundamental, an
**execution/grounding** deficit and not attribution. The models occupy different
failure regimes.

## Full-task control: navigation-coupled layout

`fulltask/` holds the full benchmark layout (objects placed anywhere, so the causal
decision is coupled to navigation), evaluated at 5 seeds/split for all eight models.
The strongest models transfer at close to their decision-focused rates (gpt-5.5
0.80→0.80, Qwen2.5-32B 0.60→0.65, Opus 0.58→0.45), so the decision-focused numbers
are not an artifact of removing navigation. The 7-8B models stay at the low end,
where 5-seed estimates are noisy and Qwen2.5-7B falls to near zero (0.03). This
corrects an earlier claim that every model scored near zero on the full task.

## Contents
- `analysis.md` / `analysis.json`: head-to-head tables, colour reliance, the
  few-shot probe, the dose-response curve, and the oracle ladder.
- `tsr_train_vs_test.png`: affordance train-vs-test TSR with 95% CIs (300 dpi).
- `dose_response.png`: TEST colour reliance vs K per demo mode (300 dpi).
- `traces/`: per-model panel JSONL traces (standard schema; feed to the metrics).
- `fewshot_traces/`: few-shot probe traces (Sonnet/Opus/gpt-5.5).
- `dose/`: dose-response condition traces (`<mode>_k<K>/`, plus `k0/` baseline).
- `oracle/`: oracle-ladder condition traces (`<agent>_<level>/`).
- `fulltask/`: full-task control traces + analysis (8 models, 5 seeds/split).
- `cache/`: prompt→response caches; re-runs hit these, so reproduction is free.

## Reproduce
```bash
# Local open-weights (GPU); 32B needs 4-bit:
python experiments/run_llm_eval.py --backend transformers \
  --models Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct \
  --compact --both-splits --families affordance causal_gate tool_substitution contradiction \
  --seeds $(seq 0 29)
python experiments/run_llm_eval.py --backend transformers \
  --models Qwen/Qwen2.5-32B-Instruct --quantization 4bit --compact --both-splits \
  --families affordance causal_gate tool_substitution contradiction --seeds $(seq 0 29)

# Claude (ANTHROPIC_API_KEY) / OpenAI (OPENAI_API_KEY); responses are cached:
python experiments/run_llm_eval.py --backend anthropic --models claude-sonnet-4-6 --compact --both-splits ...
python experiments/run_llm_eval.py --backend openai    --models gpt-5.5 --compact --both-splits ...

# Dose-response (mode in cue|mechanistic|anticue, K in 1 3 5 10):
python experiments/run_llm_eval.py --backend anthropic --models claude-sonnet-4-6 \
  --compact --both-splits --families affordance --fewshot-k 3 --fewshot-mode cue --seeds $(seq 0 29)

# Oracle ladder (level in intervention|property|rule):
python experiments/run_llm_eval.py --backend anthropic --models claude-sonnet-4-6 \
  --compact --both-splits --families affordance tool_substitution --oracle property --seeds $(seq 0 14)

# Analysis (panel + few-shot + dose + oracle):
python experiments/analyze_results.py --traces results/reference/llm/traces/*.jsonl \
  --fewshot-traces results/reference/llm/fewshot_traces/*.jsonl \
  --dose-dir results/reference/llm/dose --oracle-dir results/reference/llm/oracle \
  --output-dir results/reference/llm
```

## Notes / limitations
- Mistral-7B-Instruct-v0.3 is excluded: its chat template rejects the harness's
  message structure; Qwen and Llama represent the open-weights floor.
- Small per-split n (5 for Opus/gpt-5.5/o3) gives wide CIs on per-family gaps;
  the overall gradient, the dose-response shape, and the hint ladder are the
  robust signals.
- The decision-focused layout changes the task distribution (to isolate causal
  reasoning) and removes navigation as a confound; a full-task control (`fulltask/`)
  shows the strongest models transfer at close to their decision-focused rates, while
  the 7-8B models stay at the low end.

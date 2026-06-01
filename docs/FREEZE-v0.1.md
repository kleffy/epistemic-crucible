# Epistemic Crucible: v0.1 freeze

The `v0.1-arxiv` tag is the frozen baseline accompanying arXiv v1 of the benchmark
paper. Agents developed against EC (including ones reported in later papers) should be
evaluated on this frozen surface. See `docs/release_policy.md` for the versioning policy
and `docs/benchmark_card.md` for the datasheet.

## Frozen surface (must not change within the v0.1 line)

- [x] **Task grammar and families**, affordance, causal gate, counterfactual, tool
  substitution, contradiction; goal predicates fixed.
- [x] **Split policies**, feature holdout, counterfactual holdout, rule holdout,
  composition holdout; 80/20 train/test seed assignment from a separate RNG stream.
- [x] **Seed scheme**, `generate_task(family, seed, split)` is deterministic and
  bit-reproducible across machines at a fixed Python/NumPy.
- [x] **Metric vector**, task success, transfer success, shortcut sensitivity,
  intervention validity, intervention efficiency, counterfactual accuracy, concept
  reuse, failure diversity, curriculum progression, plus the perturbation shortcut
  exposure score. No aggregate.
- [x] **Legal-action serialization**, index selection over the legal-only candidate
  list; object IDs anonymized to `object_1..N`.
- [x] **Public-observation boundary**, hidden properties never enter the observation.
- [x] **Trace schema**, step/outcome JSONL keys as listed in `docs/release_policy.md`.

Bug-fix PATCH releases are allowed but must not change generated worlds, metric
definitions, or the trace schema. Any change to the above is a new MINOR version.

## Reference-artifact hashes (sha256)

Recompute with `sha256sum <path>` and compare. Frozen at `v0.1-arxiv`.

| Artifact | sha256 |
|----------|--------|
| `results/reference/llm/analysis.json` | `123d4caec9d0eed7493dcbcf618ac7b23c7121f5c3a93263c63a5ce110e1e73c` |
| `results/reference/llm/fulltask/analysis.json` | `a0c487961238781e1b6dfb69148bb1f5ff11eb294c48d41b315a2be35b6899c0` |

### Panel traces (`results/reference/llm/traces/`)

| Trace | sha256 |
|-------|--------|
| `llm_claude-haiku-4-5-20251001_20260530_174839.jsonl` | `b4adfd3f75c697a8346f8ee7e265dd592cf64a8621fb8d6761100cc6513c16d8` |
| `llm_claude-opus-4-8_20260530_112046.jsonl` | `4c713fb108be96ed1189b312d9274ab8398f765f4801fa2b9a11bb42a291996d` |
| `llm_claude-sonnet-4-6_20260530_182704.jsonl` | `82d1b44c678aee7b00ad581c080490a31a285921c60bb5c7f852c4af80598a8b` |
| `llm_gpt-5.5_20260530_114854.jsonl` | `de2aa26e4a5ca71d2b5ff9861aa6bc4e09663e53b2177ae14bcb4d2b7640220a` |
| `llm_llama-3.1-8b-instruct_20260530_175141.jsonl` | `d296535e21353f3d62ced11ab384944f8de14dffe95b6508eed23e915c49d520` |
| `llm_o3_20260530_115513.jsonl` | `6f90719d15dd5b5c2e3987333c3d97153b25a7fdcee5b74d240532b06f900709` |
| `llm_qwen2.5-32b-instruct_20260530_201624.jsonl` | `1ea9e43b95f8959f302d75103e3e36be808fd392ee07c11d6dc9ea5b6e81ab5f` |
| `llm_qwen2.5-7b-instruct_20260530_173812.jsonl` | `b7488e0815d51bd8254adbb1eae3c27a2afbbabc3d304770adbb8f90f0c882d6` |

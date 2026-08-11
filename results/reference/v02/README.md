# v0.2 lightweight audit artifacts

This directory contains inspectable summaries and compact sample traces for the
integrity and matched-BC calibration runs. Checkpoints remain outside Git; their
SHA-256 digests are recorded in `checkpoint_hashes.json`.

`factorial_bc_summary.json` uses held-out base seeds 100–163. The earlier overlapping
evaluation is intentionally not copied; `invalid_evaluations.json` records its digest,
reason for invalidation, and replacement. Regenerate this directory with:

```bash
python experiments/package_factorial_artifacts.py
```

`gpt_oss_serving_preflight.json` records a separate, invalid model-serving
preflight. No attribution scores from those runs are reported. The pinned vLLM
stack produced different visible actions for byte-identical policy inputs, and
its Humming MXFP4 kernel explicitly rejected batch-invariant execution. GPT-OSS
therefore remains blocked until a replacement serving stack passes the in-run
zero-conflict gate in `run_factorial_eval.py`.

`qwen_vllm_serving_preflight.json` records Qwen's failed vLLM initialization:
batch-invariant mode is unsupported by the model's GDN attention backend. It
contains no requests, traces, or metrics. Qwen evaluation therefore uses the
pinned, greedy, batch-size-1 Transformers fallback until a separate batched
acquisition passes the same zero-conflict gate.

Qwen's valid serial developmental evidence is summarized in
`qwen_serial_pilot.json`, `qwen_ceiling_selection.json`, and
`qwen_challenge_3x2_pilot.json`. The output ceiling was re-opened after the
first disjoint-seed grid missed hard 1,536-token tails; matched serial runs on
the original pilot seeds selected 2,048 tokens using length and parse behavior
only. `qwen_batched_serving_preflight.json` records the separate batch-size-4
failure: nine raw-response conflict groups invalidated that trace and all
scientific reports were suppressed. Qwen therefore remains batch-size 1.

Mistral's valid serial vLLM evidence is summarized in
`mistral_serial_pilot.json`, `mistral_ceiling_selection.json`, and
`mistral_challenge_3x2_pilot.json`. All 685 five-arm pilot calls and all 239
challenge calls were uncached, parsed, and terminated normally, with no
response, action, or candidate-set conflicts. The 1,024-token ceiling was
retained because the pilot had zero length finishes and a 100% parse rate.

Both valid models triggered the predeclared 3x2 promotion rule on the 2x2
pilot. The disjoint challenge pilots then passed, so `challenge_3x2` is the
selected confirmatory variant. GPT-OSS remains in the exclusion registry and
is not a confirmatory model.

The gate separates live acquisition from artifact replay. Live runs cannot read
the response cache and require every audited record to report `cache_hit=false`;
replay runs require cache hits, never contact a server, and cannot emit a passing
serving-validation status.

The supplied acceptance files are preserved verbatim. Run them together with:

```bash
pytest -q tests/test_quartet_invariance.py tests/test_attribution_metrics.py
```

The expected collection count is exactly 1,033 tests: 1,024 quartet-invariance
cases plus nine attribution-metric cases.

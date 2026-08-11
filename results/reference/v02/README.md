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

The supplied acceptance files are preserved verbatim. Run them together with:

```bash
pytest -q tests/test_quartet_invariance.py tests/test_attribution_metrics.py
```

The expected collection count is exactly 1,033 tests: 1,024 quartet-invariance
cases plus nine attribution-metric cases.

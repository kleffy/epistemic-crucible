# Frozen confirmatory results

This directory is the compact, post-freeze projection of the two eligible
confirmatory acquisitions. It contains all ten arm reports, the prespecified
seed-paired contrasts, denominators, model revisions, serving-integrity counts,
and source hashes. Scientific estimates are copied verbatim from the frozen
per-model summaries; no alternative estimand is computed here. Raw traces,
caches, model weights, and the SquashFS seal are intentionally excluded.

Regenerate this directory with `experiments/package_confirmatory_results.py`,
then generate paper assets with `experiments/build_submission_assets.py`.

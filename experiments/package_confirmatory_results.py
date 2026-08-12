"""Project frozen confirmatory summaries into compact, auditable result artifacts.

This command copies prespecified estimates and paired contrasts verbatim.  It
does not read outcomes or recompute an estimand.  Raw traces are inspected only
to count provider ``finish_reason=length`` diagnostics by frozen prompt arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
from typing import Any

ARM_ORDER = ("neutral", "cue_a", "cue_b", "mechanism_a", "mechanism_b")
MODEL_KEYS = ("qwen", "mistral")
PROTOCOL_ARTIFACT = {
    "name": "protocol artifact P",
    "public_tag": "v0.2-protocol-rc2",
    "commit": "da7d4f2a481514ea80bd1130dd3f9e2582a0444c",
}
FREEZE_ARTIFACT = {
    "name": "confirmatory freeze artifact F",
    "public_tag": "v0.2-confirmatory-freeze-20260811",
    "commit": "df320df9ab6265fff3cc8ca5b545fb317fd2e0a8",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _length_counts(trace: pathlib.Path) -> dict[str, int]:
    counts = {arm: 0 for arm in ARM_ORDER}
    with trace.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") != "step":
                continue
            if record.get("generation", {}).get("finish_reason") == "length":
                counts[record["prompt_arm"]] += 1
    return counts


def _integer_failures(metric: dict[str, Any]) -> int:
    value = float(metric["value"])
    denominator = int(metric["denominator"])
    count = round(value * denominator)
    if abs(value - count / denominator) > 1e-12:
        raise ValueError("parse-failure rate does not map to an integer count")
    return count


def _project_model(
    key: str,
    summary_path: pathlib.Path,
    trace_path: pathlib.Path,
) -> dict[str, Any]:
    summary = _load(summary_path)
    manifest = summary["manifest"]
    reports = summary["reports"]
    if set(reports) != set(ARM_ORDER):
        raise ValueError(f"{key}: unexpected arm set: {tuple(reports)}")
    if manifest["repository_commit"] != FREEZE_ARTIFACT["commit"]:
        raise ValueError(f"{key}: summary is not tied to confirmatory freeze artifact F")
    if summary["status"] != "eligible-for-scientific-analysis":
        raise ValueError(f"{key}: frozen summary status is {summary['status']!r}")

    validation = summary["serving_validation"]
    counts = validation["counts"]
    if not validation["passed"]:
        raise ValueError(f"{key}: serving validation did not pass")
    required_counts = {
        "cache_hit_records": 0,
        "initial_mechanism_axis_groups": 640,
        "response_conflict_groups": 0,
        "action_conflict_groups": 0,
        "candidate_set_conflict_groups": 0,
    }
    for field, expected in required_counts.items():
        if counts[field] != expected:
            raise ValueError(f"{key}: {field}={counts[field]}, expected {expected}")

    length_counts = _length_counts(trace_path)
    projected_reports: dict[str, Any] = {}
    for arm in ARM_ORDER:
        report = reports[arm]
        invalid = _integer_failures(report["parse_failure_rate"])
        length = length_counts[arm]
        if length > invalid:
            raise ValueError(f"{key}/{arm}: more length finishes than invalid outcomes")
        projected_reports[arm] = {
            **report,
            "diagnostics": {
                "invalid_responses": invalid,
                "length_finishes": length,
                "denominator": report["parse_failure_rate"]["denominator"],
            },
        }

    outcomes = sum(
        int(projected_reports[arm]["parse_failure_rate"]["denominator"]) for arm in ARM_ORDER
    )
    if outcomes != 1920:
        raise ValueError(f"{key}: expected 1,920 episode outcomes, found {outcomes}")

    display_name = {
        "qwen": "Qwen3.6-27B",
        "mistral": "Mistral-Small-3.2-24B-Instruct-2506",
    }[key]
    return {
        "display_name": display_name,
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "backend": manifest["backend"],
        "serving": manifest["serving"],
        "status": summary["status"],
        "manifest_hash": summary["manifest_hash"],
        "stage_manifest_hash": summary["stage_manifest_hash"],
        "source_summary_sha256": sha256(summary_path),
        "source_trace_sha256": sha256(trace_path),
        "run_manifest": manifest,
        "stage_manifest": summary["stage_manifest"],
        "outcome_count": outcomes,
        "initial_mechanism_axis_groups": counts["initial_mechanism_axis_groups"],
        "cache_hits": counts["cache_hit_records"],
        "conflict_groups": {
            name: value for name, value in counts.items() if name.endswith("conflict_groups")
        },
        "reports": projected_reports,
        "paired_contrasts": summary["paired_contrasts"],
    }


def package_confirmatory_results(
    *,
    qwen_summary: pathlib.Path,
    qwen_trace: pathlib.Path,
    mistral_summary: pathlib.Path,
    mistral_trace: pathlib.Path,
    structural_verification: pathlib.Path,
    sealed_checksums: pathlib.Path,
    squashfs_image: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    """Create the compact public result record and its lossless source manifest."""
    for required in (
        qwen_summary,
        qwen_trace,
        mistral_summary,
        mistral_trace,
        structural_verification,
        sealed_checksums,
        squashfs_image,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    output.mkdir(parents=True, exist_ok=True)
    models = {
        "qwen": _project_model("qwen", qwen_summary, qwen_trace),
        "mistral": _project_model("mistral", mistral_summary, mistral_trace),
    }
    combined = {
        "schema_version": "0.2-confirmatory-results-1",
        "projection_policy": (
            "Prespecified estimates and seed-paired contrasts are copied verbatim from "
            "the frozen summaries; traces are read only for finish-reason diagnostics."
        ),
        "protocol_artifact": PROTOCOL_ARTIFACT,
        "confirmatory_freeze_artifact": FREEZE_ARTIFACT,
        "design": {
            "variant": "challenge_3x2",
            "base_seeds": 64,
            "prompt_arms": list(ARM_ORDER),
            "cells_per_arm_seed": 6,
            "bootstrap_resamples": 10000,
        },
        "models": models,
    }
    _write_json(output / "confirmatory_results.json", combined)
    shutil.copyfile(structural_verification, output / "structural_verification.json")

    seal_hashes = {
        "sealed_checksums_sha256": sha256(sealed_checksums),
        "squashfs_sha256": sha256(squashfs_image),
        "squashfs_size_bytes": squashfs_image.stat().st_size,
        "note": "The immutable scientific seal is not included in the repository.",
    }
    _write_json(output / "seal_hashes.json", seal_hashes)

    manuscript_inputs = {
        "schema_version": "0.2-manuscript-inputs-1",
        "artifacts": {
            "protocol": "protocol artifact P",
            "confirmatory_freeze": "confirmatory freeze artifact F",
        },
        "design": combined["design"],
        "models": {
            key: {
                "display_name": model["display_name"],
                "model_id": model["model_id"],
                "model_revision": model["model_revision"],
                "reports": model["reports"],
                "paired_contrasts": model["paired_contrasts"],
            }
            for key, model in models.items()
        },
        "source_summary_hashes": {
            key: model["source_summary_sha256"] for key, model in models.items()
        },
    }
    _write_json(output / "manuscript_inputs.json", manuscript_inputs)

    readme = """# Frozen confirmatory results

This directory is the compact, post-freeze projection of the two eligible
confirmatory acquisitions. It contains all ten arm reports, the prespecified
seed-paired contrasts, denominators, model revisions, serving-integrity counts,
and source hashes. Scientific estimates are copied verbatim from the frozen
per-model summaries; no alternative estimand is computed here. Raw traces,
caches, model weights, and the SquashFS seal are intentionally excluded.

Regenerate this directory with `experiments/package_confirmatory_results.py`,
then generate paper assets with `experiments/build_submission_assets.py`.
"""
    (output / "README.md").write_text(readme)

    manifest = {
        "schema_version": "0.2-confirmatory-artifact-manifest-1",
        "source_files": {
            "qwen_summary": sha256(qwen_summary),
            "qwen_trace": sha256(qwen_trace),
            "mistral_summary": sha256(mistral_summary),
            "mistral_trace": sha256(mistral_trace),
            "structural_verification": sha256(structural_verification),
            "sealed_checksums": sha256(sealed_checksums),
            "squashfs_image": sha256(squashfs_image),
        },
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest["files"][path.name] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
    _write_json(output / "artifact_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-summary", required=True, type=pathlib.Path)
    parser.add_argument("--qwen-trace", required=True, type=pathlib.Path)
    parser.add_argument("--mistral-summary", required=True, type=pathlib.Path)
    parser.add_argument("--mistral-trace", required=True, type=pathlib.Path)
    parser.add_argument("--structural-verification", required=True, type=pathlib.Path)
    parser.add_argument("--sealed-checksums", required=True, type=pathlib.Path)
    parser.add_argument("--squashfs-image", required=True, type=pathlib.Path)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/reference/v02/confirmatory"),
    )
    args = parser.parse_args(argv)
    package_confirmatory_results(**vars(args))


if __name__ == "__main__":
    main()

"""Submission-closure tests for compact results, assets, and anonymity."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from crucible.factorial import generate_affordance_quartet
from crucible.viz import render_factorial_world
from experiments.build_anonymous_supplement import build_anonymous_supplement
from experiments.build_submission_assets import build_assets

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results/reference/v02/confirmatory"
ARM_ORDER = ("neutral", "cue_a", "cue_b", "mechanism_a", "mechanism_b")
EXPECTED_SUMMARY_HASHES = {
    "qwen": "7cdeb49984ebebff9799ce1828d81ee1b8dc61f860a1b7e9806c1eae098a9f07",
    "mistral": "9d09251552e52cd9b0199e33fc36c7c75410ad668016d4eefa257598140fa4b5",
}


def _load(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_confirmatory_projection_contains_all_ten_frozen_rows():
    results = _load(RESULTS / "confirmatory_results.json")
    assert results["projection_policy"].startswith("Prespecified estimates")
    assert set(results["models"]) == {"qwen", "mistral"}
    for model in results["models"].values():
        assert set(model["reports"]) == set(ARM_ORDER)
        assert model["outcome_count"] == 1920
        assert model["initial_mechanism_axis_groups"] == 640
        assert model["cache_hits"] == 0
        assert set(model["conflict_groups"].values()) == {0}


@pytest.mark.parametrize(
    "model,arm,value,low,high",
    (
        ("qwen", "neutral", 0.9947916666666666, 0.984375, 1.0),
        ("qwen", "cue_a", 1 / 3, 1 / 3, 1 / 3),
        ("qwen", "cue_b", 1 / 3, 1 / 3, 1 / 3),
        ("qwen", "mechanism_a", 0.9713541666666666, 0.9401041666666667, 0.9947916666666666),
        ("qwen", "mechanism_b", 1.0, 1.0, 1.0),
        ("mistral", "neutral", 1.0, 1.0, 1.0),
        ("mistral", "cue_a", 1 / 3, 1 / 3, 1 / 3),
        ("mistral", "cue_b", 1 / 3, 1 / 3, 1 / 3),
        ("mistral", "mechanism_a", 0.9869791666666666, 0.9739583333333334, 0.9973958333333333),
        ("mistral", "mechanism_b", 1.0, 1.0, 1.0),
    ),
)
def test_cell_success_is_exact_frozen_value(model, arm, value, low, high):
    report = _load(RESULTS / "confirmatory_results.json")["models"][model]["reports"][arm]
    assert report["cell_success"] == {
        "value": value,
        "ci_low": low,
        "ci_high": high,
        "denominator": 64,
    }


def test_diagnostics_and_source_hashes_are_exact():
    models = _load(RESULTS / "confirmatory_results.json")["models"]
    assert {key: model["source_summary_sha256"] for key, model in models.items()} == (
        EXPECTED_SUMMARY_HASHES
    )
    for key, model in models.items():
        for arm, report in model["reports"].items():
            expected = 11 if (key, arm) == ("qwen", "mechanism_a") else 0
            assert report["diagnostics"] == {
                "invalid_responses": expected,
                "length_finishes": expected,
                "denominator": 384,
            }


def test_prespecified_paired_contrasts_and_denominators():
    models = _load(RESULTS / "confirmatory_results.json")["models"]
    expected = {
        ("qwen", "cue_a_minus_neutral"): (-0.6614583333333335, 0.6614583333333335),
        ("qwen", "cue_b_minus_neutral"): (-0.6614583333333335, 0.6614583333333335),
        ("qwen", "mechanism_a_minus_neutral"): (0.005208333333333334, -0.006510416666666666),
        ("qwen", "mechanism_b_minus_neutral"): (0.005208333333333334, -0.005208333333333333),
        ("mistral", "cue_a_minus_neutral"): (-2 / 3, 2 / 3),
        ("mistral", "cue_b_minus_neutral"): (-2 / 3, 2 / 3),
        ("mistral", "mechanism_a_minus_neutral"): (-0.013020833333333332, 0.013020833333333332),
        ("mistral", "mechanism_b_minus_neutral"): (0.0, 0.0),
    }
    for (model, contrast), values in expected.items():
        report = models[model]["paired_contrasts"][contrast]
        assert report["mechanism_accuracy"]["value"] == pytest.approx(values[0])
        assert report["cue_following"]["value"] == pytest.approx(values[1])
        assert report["mechanism_accuracy"]["denominator"] == 64
        assert report["cue_following"]["denominator"] == 64


def test_result_artifact_manifest_hashes_every_committed_file():
    manifest = _load(RESULTS / "artifact_manifest.json")
    expected = {
        path.relative_to(RESULTS).as_posix()
        for path in RESULTS.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    assert set(manifest["files"]) == expected
    for name, record in manifest["files"].items():
        path = RESULTS / name
        assert path.stat().st_size == record["size_bytes"]
        assert _sha256(path) == record["sha256"]
    structural = _load(RESULTS / "structural_verification.json")
    assert structural["passed"] is True
    assert structural["scientific_estimands_recomputed"] is False


def _figure_text(figure) -> set[str]:
    return {text.get_text() for axis in figure.axes for text in axis.texts}


def test_factorial_render_masks_hidden_mechanism_and_keeps_stable_labels():
    quartet = generate_affordance_quartet(7)
    default_text: list[set[str]] = []
    for cell in quartet.cells.values():
        figure = render_factorial_world(cell)
        text = _figure_text(figure)
        assert "conductive" not in text
        default_text.append(text)
        plt.close(figure)
    assert all(text == default_text[0] for text in default_text[1:])

    figure = render_factorial_world(quartet.cell(0, 1), reveal_mechanism=True)
    assert "conductive" in _figure_text(figure)
    plt.close(figure)


def test_submission_asset_generation_is_byte_deterministic(tmp_path: Path):
    kwargs = {
        "inputs_path": RESULTS / "manuscript_inputs.json",
        "integrity_path": ROOT / "results/reference/v02/integrity_report.json",
        "bc_path": ROOT / "results/reference/v02/factorial_bc_summary.json",
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest_a = build_assets(output=first, **kwargs)
    manifest_b = build_assets(output=second, **kwargs)
    assert manifest_a == manifest_b
    assert len([name for name in manifest_a if name.endswith(".tex")]) == 4
    assert len([name for name in manifest_a if name.endswith(".pdf")]) == 4
    assert len([name for name in manifest_a if name.endswith(".svg")]) == 4
    for name, record in manifest_a.items():
        assert _sha256(first / name) == _sha256(second / name) == record["sha256"]


def test_anonymous_supplement_is_deterministic_and_identity_free(tmp_path: Path):
    first = tmp_path / "anonymous-a.zip"
    second = tmp_path / "anonymous-b.zip"
    record_a = build_anonymous_supplement(root=ROOT, output=first)
    record_b = build_anonymous_supplement(root=ROOT, output=second)
    assert record_a["sha256"] == record_b["sha256"]
    assert first.read_bytes() == second.read_bytes()

    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert not any(".git" in Path(name).parts for name in names)
        assert "CITATION.cff" not in names
        assert "configs/confirmatory_manifest.json" in names
        assert "results/reference/v02/confirmatory/manuscript/figure4_prompt_profiles.pdf" in names
        text = b"\n".join(
            archive.read(name)
            for name in names
            if Path(name).suffix in {".json", ".md", ".py", ".tex", ".toml", ".txt", ".yaml"}
        ).lower()
    for forbidden in (
        b"github.com/",
        b"/home/",
        b"/mnt/data/",
        b"kleffy",
        b"v0.2-protocol-rc2",
        b"v0.2-confirmatory-freeze-20260811",
    ):
        assert forbidden not in text
    assert b"protocol artifact p" in text
    assert b"confirmatory freeze artifact f" in text

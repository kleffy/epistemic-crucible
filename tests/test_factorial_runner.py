from __future__ import annotations

import json

import pytest
import yaml

from experiments.run_factorial_eval import (
    build_factorial_demo_examples,
    build_factorial_fewshot_prefix,
    run_controls,
    run_local_model,
    validate_stage_manifest,
)
from experiments.run_integrity_gate import run_integrity_gate


def test_six_shot_prompt_arms_are_equal_count_and_mechanism_is_colour_balanced():
    cue = build_factorial_fewshot_prefix("cue", count=6, base_seed=9_000)[0]["content"]
    mechanism = build_factorial_fewshot_prefix("mechanism", count=6, base_seed=11_000)[0]["content"]
    mechanism_examples = build_factorial_demo_examples("mechanism", count=6, base_seed=11_000)
    assert cue.count("Example ") == mechanism.count("Example ") == 6
    assert cue.lower().count("red tool") == 6
    assert "signal-positive" not in cue
    assert mechanism.count("signal-positive") == 6
    assert {example["condition_id"] for example in mechanism_examples} == {
        "m0_c0",
        "m0_c1",
        "m1_c0",
        "m1_c1",
        "m2_c0",
        "m2_c1",
    }
    assert {
        color: sum(example["positive_color"] == color for example in mechanism_examples)
        for color in ("red", "blue", "green")
    } == {"red": 2, "blue": 2, "green": 2}
    assert [
        example["condition_id"]
        for example in build_factorial_demo_examples("mechanism", base_seed=11_000)
    ] != [
        example["condition_id"]
        for example in build_factorial_demo_examples("mechanism", base_seed=12_000)
    ]


def test_non_six_shot_factorial_demonstrations_are_rejected():
    with pytest.raises(ValueError, match="exactly six shots"):
        build_factorial_fewshot_prefix("mechanism", count=5)


def test_control_runner_writes_schema_v02_reports(tmp_path):
    summary = run_controls(
        {"seeds": [0], "controls": ["mechanism_oracle", "cue_follower"]},
        tmp_path,
    )
    assert summary["schema_version"] == "0.2"
    assert summary["reports"]["mechanism_oracle"]["quartet_success"]["value"] == 1.0
    record = json.loads(
        (tmp_path / "factorial_control_mechanism_oracle.jsonl").read_text().splitlines()[0]
    )
    assert record["protocol"] == "affordance-factorial-v0.2"
    assert record["condition_id"] == "m0_c0"
    assert record["episode_ledger"]
    assert len(record["episode_ledger_hash"]) == 64


def test_invalid_model_responses_are_abstentions_and_step_records_are_loadable(
    tmp_path, monkeypatch
):
    from crucible.agents.llm_backends import GenerationRecord
    from crucible.metrics import load_trace

    class InvalidBackend:
        def generate_batch_records(self, system, conversations, *, cache_contexts):
            return [
                GenerationRecord(
                    response="not a valid action",
                    model_id="test/model",
                    backend="test",
                    requested_params={},
                    effective_params={},
                    usage={},
                    finish_reason="stop",
                    response_id=None,
                )
                for _ in conversations
            ]

    monkeypatch.setattr(
        "experiments.run_factorial_eval.make_backend", lambda *args, **kwargs: InvalidBackend()
    )
    config = {
        "seeds": [1000],
        "prompt_arms": [{"name": "neutral", "mode": "neutral", "count": 0, "seed_count": 1}],
    }
    model = {
        "label": "test-model",
        "id": "test/model",
        "revision": "0" * 40,
        "tokenizer_revision": "0" * 40,
        "output_ceiling": 16,
        "generation": {},
        "serving": {},
    }
    summary = run_local_model(
        config,
        model,
        tmp_path,
        stage="pilot",
        stage_manifest={"stage": "pilot", "base_seeds": [1000]},
        stage_manifest_hash="a" * 64,
    )
    steps, outcomes = load_trace(summary["trace"])
    assert len(steps) == len(outcomes) == 4
    assert all(step["kind"] == "step" for step in steps)
    assert all(step["parse_status"] == "invalid" for step in steps)
    assert all(step["fallback_action"] is None for step in steps)
    assert all(outcome["committed_slot"] is None for outcome in outcomes)
    assert summary["reports"]["neutral"]["parse_failure_rate"] == {
        "value": 1.0,
        "denominator": 4,
    }


def test_small_integrity_gate_passes():
    report = run_integrity_gate(invariant_seeds=4, oracle_seeds=2, control_seeds=2)
    assert report["passed"] is True
    assert report["leakage"]["mechanism_public_match_rate"] == 1.0


def test_model_study_manifest_pins_full_revisions():
    with open("configs/factorial_v02.yaml") as handle:
        config = yaml.safe_load(handle)
    assert len(config["seeds"]) == 64
    assert config["seeds"] == list(range(100, 164))
    assert len(config["prompt_arms"]) == 5
    assert all(arm["seed_count"] == 64 for arm in config["prompt_arms"])
    for model in config["models"]:
        assert len(model["revision"]) == 40
        int(model["revision"], 16)


def test_pilot_and_confirmatory_seed_manifests_are_disjoint():
    pilot = json.loads(open("configs/pilot_manifest.json").read())
    confirmatory = json.loads(open("configs/confirmatory_manifest.json").read())
    assert set(pilot["base_seeds"]).isdisjoint(confirmatory["base_seeds"])
    assert confirmatory["bootstrap"]["resamples"] == 10_000
    assert confirmatory["smallest_effect_of_interest"]["cue_tracking"] == 0.15
    assert confirmatory["base_seeds"] == list(range(100, 164))


def test_confirmatory_manifest_fails_closed_until_frozen_and_exact():
    with open("configs/factorial_v02.yaml") as handle:
        config = yaml.safe_load(handle)
    manifest = json.loads(open("configs/confirmatory_manifest.json").read())
    model = config["models"][0]
    with pytest.raises(ValueError, match="frozen=true"):
        validate_stage_manifest(
            "confirmatory",
            manifest,
            config,
            model,
            repository_commit="abc123",
            working_tree_clean=True,
        )
    frozen = {**manifest, "frozen": True, "protocol_commit": "abc123"}
    validate_stage_manifest(
        "confirmatory",
        frozen,
        config,
        model,
        repository_commit="abc123",
        working_tree_clean=True,
    )
    mismatched = {**frozen, "base_seeds": list(range(64))}
    with pytest.raises(ValueError, match="base seeds"):
        validate_stage_manifest(
            "confirmatory",
            mismatched,
            config,
            model,
            repository_commit="abc123",
            working_tree_clean=True,
        )
    with pytest.raises(ValueError, match="clean working tree"):
        validate_stage_manifest(
            "confirmatory",
            frozen,
            config,
            model,
            repository_commit="abc123",
            working_tree_clean=False,
        )

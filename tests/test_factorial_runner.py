from __future__ import annotations

import json

import yaml

from experiments.run_factorial_eval import build_factorial_fewshot_prefix, run_controls
from experiments.run_integrity_gate import run_integrity_gate


def test_five_shot_prompt_arms_manipulate_content_not_count():
    cue = build_factorial_fewshot_prefix("cue", count=5, base_seed=9_000)[0]["content"]
    mechanism = build_factorial_fewshot_prefix("mechanism", count=5, base_seed=11_000)[0]["content"]
    assert cue.count("Example ") == mechanism.count("Example ") == 5
    assert cue.lower().count("red tool") == 5
    assert "signal-positive" not in cue
    assert mechanism.count("signal-positive") == 5


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


def test_small_integrity_gate_passes():
    report = run_integrity_gate(invariant_seeds=4, oracle_seeds=2, control_seeds=2)
    assert report["passed"] is True
    assert report["leakage"]["mechanism_public_match_rate"] == 1.0


def test_model_study_manifest_pins_full_revisions():
    with open("configs/factorial_v02.yaml") as handle:
        config = yaml.safe_load(handle)
    assert len(config["seeds"]) == 64
    assert len(config["prompt_arms"]) == 5
    for model in config["models"]:
        assert len(model["revision"]) == 40
        int(model["revision"], 16)


def test_pilot_and_confirmatory_seed_manifests_are_disjoint():
    pilot = json.loads(open("configs/pilot_manifest.json").read())
    confirmatory = json.loads(open("configs/confirmatory_manifest.json").read())
    assert set(pilot["base_seeds"]).isdisjoint(confirmatory["base_seeds"])
    assert confirmatory["bootstrap"]["resamples"] == 10_000
    assert confirmatory["smallest_effect_of_interest"]["cue_tracking"] == 0.15

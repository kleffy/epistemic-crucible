from __future__ import annotations

import json

import pytest
import yaml

from experiments.run_factorial_eval import (
    _runtime_hardware,
    audit_identical_prompt_generations,
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
    assert summary["manifest"]["concurrency"] == 8
    assert all(step["kind"] == "step" for step in steps)
    assert all(len(step["prompt_hash"]) == 64 for step in steps)
    assert all(len(step["message_hash"]) == 64 for step in steps)
    prompt_hashes = {step["condition"]["id"]: step["prompt_hash"] for step in steps}
    assert prompt_hashes["m0_c0"] == prompt_hashes["m1_c0"]
    assert prompt_hashes["m0_c1"] == prompt_hashes["m1_c1"]
    assert all(step["parse_status"] == "invalid" for step in steps)
    assert all(step["fallback_action"] is None for step in steps)
    assert all(outcome["committed_slot"] is None for outcome in outcomes)
    assert summary["reports"]["neutral"]["parse_failure_rate"] == {
        "value": 1.0,
        "denominator": 4,
    }
    assert summary["status"] == "eligible-for-scientific-analysis"
    assert summary["serving_validation"]["passed"] is True
    assert summary["serving_validation"]["counts"]["cache_hit_records"] == 0
    assert summary["serving_validation"]["counts"]["server_contacted"] is True


def test_factorial_runner_selects_declared_direct_transformers_backend(tmp_path, monkeypatch):
    from crucible.agents.llm_backends import GenerationRecord

    observed = {}

    class InvalidBackend:
        def generate_batch_records(self, system, conversations, *, cache_contexts):
            return [
                GenerationRecord(
                    response="invalid",
                    model_id="test/model",
                    backend="transformers",
                    requested_params={},
                    effective_params={},
                    usage={},
                    finish_reason="stop",
                    response_id=None,
                )
                for _ in conversations
            ]

    def backend_factory(kind, model_id, **kwargs):
        observed.update(kind=kind, model_id=model_id, kwargs=kwargs)
        return InvalidBackend()

    monkeypatch.setattr("experiments.run_factorial_eval.make_backend", backend_factory)
    config = {
        "seeds": [1000],
        "prompt_arms": [{"name": "neutral", "mode": "neutral", "count": 0, "seed_count": 1}],
    }
    model = {
        "label": "test-model",
        "id": "test/model",
        "revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "output_ceiling": 16,
        "generation": {},
        "serving": {
            "backend": "transformers",
            "device": "cuda",
            "dtype": "bfloat16",
            "batch_size": 1,
            "quantization": "bitsandbytes-nf4",
        },
    }
    summary = run_local_model(
        config,
        model,
        tmp_path,
        stage="pilot",
        stage_manifest={"stage": "pilot", "base_seeds": [1000]},
        stage_manifest_hash="a" * 64,
    )
    assert observed["kind"] == "transformers"
    assert observed["kwargs"]["model_revision"] == "a" * 40
    assert observed["kwargs"]["tokenizer_revision"] == "b" * 40
    assert observed["kwargs"]["batch_size"] == 1
    assert observed["kwargs"]["quantization"] == "bitsandbytes-nf4"
    assert summary["manifest"]["backend"] == "local_transformers"
    assert summary["manifest"]["server"] is None


def test_serving_reproducibility_gate_rejects_conflicting_identical_prompts(tmp_path, monkeypatch):
    from crucible.agents.llm_backends import GenerationRecord

    class DivergentBackend:
        def generate_batch_records(self, system, conversations, *, cache_contexts):
            return [
                GenerationRecord(
                    response=f"variant-{int(index >= 2)}\nACTION: 3",
                    model_id="test/model",
                    backend="test",
                    requested_params={},
                    effective_params={},
                    usage={},
                    finish_reason="stop",
                    response_id=None,
                )
                for index, _ in enumerate(conversations)
            ]

    monkeypatch.setattr(
        "experiments.run_factorial_eval.make_backend", lambda *args, **kwargs: DivergentBackend()
    )
    config = {
        "seeds": [1000],
        "prompt_arms": [{"name": "neutral", "mode": "neutral", "count": 0, "seed_count": 1}],
        "serving_validation": {
            "minimum_replicated_prompt_groups": 1,
            "minimum_initial_mechanism_axis_groups": 2,
            "maximum_response_conflict_groups": 0,
            "maximum_action_conflict_groups": 0,
            "fail_closed": True,
        },
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
    with pytest.raises(RuntimeError, match="serving reproducibility gate failed"):
        run_local_model(
            config,
            model,
            tmp_path,
            stage="pilot",
            stage_manifest={"stage": "pilot", "base_seeds": [1000]},
            stage_manifest_hash="a" * 64,
        )
    summary = json.loads((tmp_path / "factorial_test-model_summary.json").read_text())
    assert summary["status"] == "invalid-serving-reproducibility"
    assert summary["reports"] == {}
    assert summary["paired_contrasts"] == {}
    assert summary["serving_validation"]["counts"]["response_conflict_groups"] == 2
    assert summary["serving_validation"]["counts"]["action_conflict_groups"] == 0
    assert summary["serving_validation"]["counts"]["initial_mechanism_axis_groups"] == 2
    assert (
        summary["serving_validation"]["counts"]["initial_mechanism_axis_response_conflict_groups"]
        == 2
    )
    assert (
        summary["serving_validation"]["counts"]["initial_mechanism_axis_action_conflict_groups"]
        == 0
    )


def test_serving_reproducibility_audit_detects_action_and_candidate_conflicts():
    base = {
        "prompt_hash": "a" * 64,
        "base_seed": 1000,
        "prompt_arm": "neutral",
        "generation": {"response": "ACTION: 3"},
    }
    records = [
        {
            **base,
            "condition": {"id": "m0_c0"},
            "candidate_set_hash": "b" * 64,
            "action": {"kind": "commit", "slot": 0},
        },
        {
            **base,
            "condition": {"id": "m1_c0"},
            "candidate_set_hash": "c" * 64,
            "action": {"kind": "commit", "slot": 1},
        },
    ]
    audit = audit_identical_prompt_generations(records)
    assert audit["passed"] is False
    assert audit["counts"]["action_conflict_groups"] == 1
    assert audit["counts"]["candidate_set_conflict_groups"] == 1


def test_live_validation_ignores_clean_stale_cache_and_contacts_current_backend(
    tmp_path, monkeypatch
):
    from crucible.agents.llm_backends import MockBackend

    phase = {"name": "prime", "calls": 0}

    def backend_factory(_backend, model_id, **kwargs):
        def policy(_system, _conversation):
            index = phase["calls"]
            phase["calls"] += 1
            if phase["name"] == "prime":
                return "stable\nACTION: 0"
            return f"current-{index}\nACTION: {0 if index < 2 else 1}"

        return MockBackend(model_id=model_id, policy=policy, **kwargs)

    monkeypatch.setattr("experiments.run_factorial_eval.make_backend", backend_factory)
    config = {
        "execution_mode": "live_acquisition",
        "seeds": [1000],
        "prompt_arms": [{"name": "neutral", "mode": "neutral", "count": 0, "seed_count": 1}],
        "serving_validation": {
            "minimum_replicated_prompt_groups": 1,
            "minimum_initial_mechanism_axis_groups": 2,
            "maximum_response_conflict_groups": 0,
            "maximum_action_conflict_groups": 0,
            "maximum_candidate_set_conflict_groups": 0,
            "require_all_uncached": True,
            "fail_closed": True,
        },
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
    first = run_local_model(
        config,
        model,
        tmp_path,
        stage="pilot",
        stage_manifest={"stage": "pilot", "base_seeds": [1000]},
        stage_manifest_hash="a" * 64,
    )
    assert first["serving_validation"]["passed"] is True
    assert phase["calls"] == 4

    phase.update(name="current", calls=0)
    with pytest.raises(RuntimeError, match="serving reproducibility gate failed"):
        run_local_model(
            config,
            model,
            tmp_path,
            stage="pilot",
            stage_manifest={"stage": "pilot", "base_seeds": [1000]},
            stage_manifest_hash="a" * 64,
        )
    assert phase["calls"] == 4, "live validation must contact the current backend"
    summary = json.loads((tmp_path / "factorial_test-model_summary.json").read_text())
    validation = summary["serving_validation"]
    assert validation["passed"] is False
    assert validation["counts"]["cache_hit_records"] == 0
    assert validation["counts"]["uncached_records"] == 4
    assert validation["counts"]["initial_mechanism_axis_action_conflict_groups"] == 2


def test_cache_only_replay_cannot_emit_passing_serving_validation(tmp_path, monkeypatch):
    from crucible.agents.llm_backends import MockBackend

    calls = {"count": 0, "explode": False}

    def backend_factory(_backend, model_id, **kwargs):
        def policy(_system, _conversation):
            calls["count"] += 1
            if calls["explode"]:
                raise AssertionError("artifact replay contacted the backend")
            return "stable\nACTION: 0"

        return MockBackend(model_id=model_id, policy=policy, **kwargs)

    monkeypatch.setattr("experiments.run_factorial_eval.make_backend", backend_factory)
    config = {
        "execution_mode": "live_acquisition",
        "seeds": [1000],
        "prompt_arms": [{"name": "neutral", "mode": "neutral", "count": 0, "seed_count": 1}],
        "serving_validation": {
            "minimum_replicated_prompt_groups": 1,
            "minimum_initial_mechanism_axis_groups": 2,
            "maximum_response_conflict_groups": 0,
            "maximum_action_conflict_groups": 0,
            "maximum_candidate_set_conflict_groups": 0,
            "require_all_uncached": True,
            "fail_closed": True,
        },
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
    run_local_model(
        config,
        model,
        tmp_path,
        stage="pilot",
        stage_manifest={"stage": "pilot", "base_seeds": [1000]},
        stage_manifest_hash="a" * 64,
    )
    assert calls["count"] == 4

    calls.update(count=0, explode=True)
    replay = run_local_model(
        {**config, "execution_mode": "artifact_replay"},
        model,
        tmp_path,
        stage="pilot",
        stage_manifest={"stage": "pilot", "base_seeds": [1000]},
        stage_manifest_hash="a" * 64,
    )
    assert calls["count"] == 0
    assert replay["status"] == "artifact-replay-no-live-validation"
    assert replay["serving_validation"]["performed"] is False
    assert replay["serving_validation"]["passed"] is None
    assert replay["serving_validation"]["scientific_results_eligible"] is False
    assert replay["serving_validation"]["counts"]["cache_hit_records"] == 4
    assert replay["serving_validation"]["counts"]["server_contacted"] is False
    assert replay["reports"] == {}


def test_hardware_provenance_tolerates_cpu_hosts_without_nvidia_smi(monkeypatch):
    def missing_binary(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("experiments.run_factorial_eval.subprocess.run", missing_binary)
    hardware = _runtime_hardware()
    assert hardware["driver"] is None


def test_artifact_gpu_probe_tolerates_missing_nvidia_smi(monkeypatch):
    pytest.importorskip("torch")
    from experiments.package_factorial_artifacts import _gpu_metadata

    def missing_binary(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("experiments.package_factorial_artifacts.subprocess.run", missing_binary)
    assert _gpu_metadata() is None


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


def test_gpt_oss_launch_profile_disables_nondeterministic_serving_paths():
    script = open("experiments/serve_gpt_oss_v02.sh").read()
    for flag in (
        "--seed 0",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--enable-chunked-prefill",
    ):
        assert flag in script
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in script
    assert "VLLM_BATCH_INVARIANT=1" in script
    assert "failed the v0.2 identical-prompt serving gate" in script

    with open("configs/factorial_pilot_v02.yaml") as handle:
        config = yaml.safe_load(handle)
    serving = config["models"][0]["serving"]
    assert serving["batch_invariant_requested"] is True
    assert serving["batch_invariant_validated"] is False
    assert serving["preflight_status"] == "blocked-on-pinned-stack"
    report = json.loads(open(serving["preflight_record"]).read())
    assert report["status"] == "blocked-serving-reproducibility"
    assert report["scientific_metrics_reported"] is False


def test_pilot_and_confirmatory_seed_manifests_are_disjoint():
    pilot = json.loads(open("configs/pilot_manifest.json").read())
    confirmatory = json.loads(open("configs/confirmatory_manifest.json").read())
    assert set(pilot["base_seeds"]).isdisjoint(confirmatory["base_seeds"])
    assert confirmatory["bootstrap"]["resamples"] == 10_000
    assert confirmatory["smallest_effect_of_interest"]["cue_following"] == 0.15
    assert confirmatory["base_seeds"] == list(range(100, 164))
    assert confirmatory["variant"] == "challenge_3x2"


def test_confirmatory_manifest_fails_closed_until_frozen_and_exact():
    with open("configs/factorial_v02.yaml") as handle:
        config = yaml.safe_load(handle)
    manifest = json.loads(open("configs/confirmatory_manifest.json").read())
    model = config["models"][0]
    unfrozen = {**manifest, "frozen": False, "protocol_commit": None}
    with pytest.raises(ValueError, match="frozen=true"):
        validate_stage_manifest(
            "confirmatory",
            unfrozen,
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
    mismatched_concurrency = {**frozen, "concurrency": 8}
    with pytest.raises(ValueError, match="configured concurrency"):
        validate_stage_manifest(
            "confirmatory",
            mismatched_concurrency,
            config,
            model,
            repository_commit="abc123",
            working_tree_clean=True,
        )
    mismatched_serving_gate = {
        **frozen,
        "serving_validation": {
            **frozen["serving_validation"],
            "maximum_action_conflict_groups": 1,
        },
    }
    with pytest.raises(ValueError, match="configured serving validation"):
        validate_stage_manifest(
            "confirmatory",
            mismatched_serving_gate,
            config,
            model,
            repository_commit="abc123",
            working_tree_clean=True,
        )
    mismatched_execution_mode = {**frozen, "execution_mode": "artifact_replay"}
    with pytest.raises(ValueError, match="configured execution mode"):
        validate_stage_manifest(
            "confirmatory",
            mismatched_execution_mode,
            config,
            model,
            repository_commit="abc123",
            working_tree_clean=True,
        )
    mismatched_variant = {**frozen, "variant": "quartet_2x2"}
    with pytest.raises(ValueError, match="factorial variant"):
        validate_stage_manifest(
            "confirmatory",
            mismatched_variant,
            config,
            model,
            repository_commit="abc123",
            working_tree_clean=True,
        )
    mismatched_registry = {**frozen, "selected_models": ["mistral-small-3.2-24b"]}
    with pytest.raises(ValueError, match="selected model registry"):
        validate_stage_manifest(
            "confirmatory",
            mismatched_registry,
            config,
            model,
            repository_commit="abc123",
            working_tree_clean=True,
        )


def test_confirmatory_manifest_accepts_only_tagged_direct_freeze_wrapper(monkeypatch):
    with open("configs/factorial_v02.yaml") as handle:
        config = yaml.safe_load(handle)
    manifest = json.loads(open("configs/confirmatory_manifest.json").read())
    frozen = {
        **manifest,
        "frozen": True,
        "protocol_commit": "protocol-sha",
        "freeze_tag": "v0.2-confirmatory-freeze",
    }
    monkeypatch.setattr(
        "experiments.run_factorial_eval._resolve_git_revision",
        lambda revision: "freeze-sha" if revision == "v0.2-confirmatory-freeze" else None,
    )
    monkeypatch.setattr(
        "experiments.run_factorial_eval._repository_parent",
        lambda commit: "protocol-sha" if commit == "freeze-sha" else None,
    )
    validate_stage_manifest(
        "confirmatory",
        frozen,
        config,
        config["models"][0],
        repository_commit="freeze-sha",
        working_tree_clean=True,
    )
    with pytest.raises(ValueError, match="tagged direct freeze wrapper"):
        validate_stage_manifest(
            "confirmatory",
            frozen,
            config,
            config["models"][0],
            repository_commit="other-sha",
            working_tree_clean=True,
        )

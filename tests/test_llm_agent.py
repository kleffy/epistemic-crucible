"""Tests for the LLM agent — prompting, parsing, episode loop, and caching.

All use the deterministic mock backend, so they need no GPU, network, or model
download and run in CI.
"""

from __future__ import annotations

from crucible.actions import ActionKind, Direction
from crucible.agents.llm_agent import LLMAgent
from crucible.agents.llm_backends import MockBackend, ResponseCache
from crucible.agents.prompting import build_user_message, describe_goal, parse_action
from crucible.env import CrucibleEnv
from crucible.grammar import GoalKind, GoalSpec, TaskFamily, generate_task
from crucible.splits import SplitLabel


def _obs(seed: int = 0):
    spec = generate_task(TaskFamily.AFFORDANCE, seed=seed, split=SplitLabel.TRAIN)
    env = CrucibleEnv(seed=spec.seed, config={"task_spec": spec})
    return env.reset(), spec, env


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


def test_describe_goal_open():
    goal = GoalSpec(kind=GoalKind.OPEN, target_obj_id="gate0")
    assert "gate0" in describe_goal(goal)
    assert "open" in describe_goal(goal).lower()


def test_compact_layout_clusters_objects_near_agent():
    """Decision-focused layout places objects within reach and is deterministic."""
    from crucible.grammar import build_world_from_spec

    spec = generate_task(TaskFamily.AFFORDANCE, seed=1, split=SplitLabel.TRAIN)
    w1 = build_world_from_spec(spec, compact=True)
    w2 = build_world_from_spec(spec, compact=True)
    ax, ay = w1.agent.pos
    dists = [
        abs(o.visible.pos[0] - ax) + abs(o.visible.pos[1] - ay)
        for o in w1.objects.values()
        if o.visible.pos is not None
    ]
    assert max(dists) <= 3, "compact layout must keep objects within a few cells"
    # deterministic
    assert {k: v.visible.pos for k, v in w1.objects.items()} == {
        k: v.visible.pos for k, v in w2.objects.items()
    }


def test_user_message_contains_goal_and_candidates():
    obs, spec, _ = _obs()
    msg, candidates, _ = build_user_message(obs, describe_goal(spec.goal))
    assert "GOAL:" in msg
    assert "Legal actions" in msg
    assert len(candidates) > 1
    # every candidate appears as a numbered JSON payload line
    assert msg.count('{"kind"') >= len(candidates)


def test_prompt_never_leaks_split_or_object_ids():
    """The rendered prompt must not reveal the split or raw object IDs.

    Object IDs embed the split (e.g. ``aff_0_train_gate``); a text agent could
    otherwise read train-vs-test straight off the IDs and defeat the transfer
    diagnostic. The goal text, observation, candidate list, and effect feedback
    must all be anonymized.
    """
    from crucible.agents.prompting import anonymize_text, build_label_map

    for split in (SplitLabel.TRAIN, SplitLabel.TEST):
        spec = generate_task(TaskFamily.AFFORDANCE, seed=0, split=split)
        env = CrucibleEnv(seed=spec.seed, config={"task_spec": spec})
        obs = env.reset()
        msg, _, label_map = build_user_message(obs, describe_goal(spec.goal))
        assert "train" not in msg and "test" not in msg, f"split leaked for {split}"
        for oid in obs["objects"]:
            assert oid not in msg, f"raw object id {oid} leaked"
        # Effect strings (which embed IDs) must anonymize too.
        effect = anonymize_text(f"opened {spec.goal.target_obj_id}", label_map)
        assert "train" not in effect and "test" not in effect
        assert spec.goal.target_obj_id not in effect
    # Labels are stable across steps (hash order, extend-only).
    assert build_label_map(obs, build_label_map(obs)) == build_label_map(obs)


def test_parse_action_index_selection():
    obs, spec, _ = _obs()
    _, candidates, _ = build_user_message(obs, describe_goal(spec.goal))
    # The model picks action #2 from the numbered legal list.
    chosen = parse_action("I'll take the second option.\nACTION: 2", candidates)
    assert chosen.kind == candidates[2].kind and chosen.args == candidates[2].args
    # Out-of-range index falls back to WAIT.
    assert parse_action("ACTION: 999", candidates).kind == ActionKind.WAIT


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_action_handles_reasoning_then_action_line():
    raw = 'I should test the red tool first.\nACTION: {"kind": "inspect", "args": {"obj_id": "t0"}}'
    a = parse_action(raw)
    assert a.kind == ActionKind.INSPECT and a.args["obj_id"] == "t0"


def test_parse_action_handles_direction_and_code_fence():
    raw = 'Let me move.\n```\nACTION: {"kind": "move", "args": {"direction": "North"}}\n```'
    a = parse_action(raw)
    assert a.kind == ActionKind.MOVE and a.args["direction"] == Direction.NORTH


def test_parse_action_garbage_falls_back_to_wait():
    assert parse_action("no json here at all").kind == ActionKind.WAIT
    assert parse_action('{"kind": "not_a_real_kind"}').kind == ActionKind.WAIT


# ---------------------------------------------------------------------------
# Episode loop with the mock backend
# ---------------------------------------------------------------------------


def test_llm_agent_drives_episode_with_mock():
    obs, spec, env = _obs()
    agent = LLMAgent(backend="mock")  # default policy -> WAIT
    agent.reset()
    obs["_goal_text"] = describe_goal(spec.goal)
    steps = 0
    for _ in range(spec.max_steps):
        action = agent.act(obs)
        assert action.kind == ActionKind.WAIT
        obs, _, done, info = env.step(action)
        obs["_goal_text"] = describe_goal(spec.goal)
        agent.observe_result(obs, 0.0, done, info)
        steps += 1
        if done:
            break
    assert steps > 0


def test_llmagent_default_backend_follows_credentials(monkeypatch):
    """LLMAgent() uses an API backend when credentialed, else mock (lazily —
    no backend is constructed and no API call is made here)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = LLMAgent()
    assert a._backend_name == "mock" and a.model_id == "mock"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    a = LLMAgent()
    assert a._backend_name == "anthropic" and "claude" in a.model_id

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    a = LLMAgent()
    assert a._backend_name == "openai"

    # An explicit backend is always respected.
    assert LLMAgent(backend="mock")._backend_name == "mock"


def test_llm_agent_parses_non_wait_action():
    obs, spec, _ = _obs()
    policy = lambda system, conv: 'ACTION: {"kind": "move", "args": {"direction": "south"}}'  # noqa: E731
    agent = LLMAgent(backend="mock")
    agent._backend = MockBackend(policy=policy)
    obs["_goal_text"] = describe_goal(spec.goal)
    action = agent.act(obs)
    assert action.kind == ActionKind.MOVE and action.args["direction"] == Direction.SOUTH


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_response_cache_avoids_recompute(tmp_path):
    calls = {"n": 0}

    def counting_policy(system, conv):
        calls["n"] += 1
        return 'ACTION: {"kind": "wait", "args": {}}'

    cache = ResponseCache(tmp_path / "cache.jsonl")
    backend = MockBackend(policy=counting_policy, cache=cache)
    conv = [{"role": "user", "content": "hello"}]
    backend.generate_batch("sys", [conv])
    backend.generate_batch("sys", [conv])  # identical -> cache hit
    assert calls["n"] == 1, "second identical call must hit the cache"

    # A fresh cache loaded from disk still has the entry (persisted).
    reloaded = ResponseCache(tmp_path / "cache.jsonl")
    key = ResponseCache.key("mock", "sys", conv, backend.gen_params)
    assert reloaded.get(key) is not None


def test_response_cache_persists_complete_generation_provenance(tmp_path):
    cache = ResponseCache(tmp_path / "records.jsonl")
    backend = MockBackend(
        cache=cache,
        run_manifest={"protocol": "affordance_quartet", "revision": "abc123"},
        max_new_tokens=1024,
    )
    conversation = [{"role": "user", "content": "state"}]
    first = backend.generate_batch_records("system", [conversation])[0]
    second = backend.generate_batch_records("system", [conversation])[0]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.requested_params["run_manifest"]["revision"] == "abc123"
    key = ResponseCache.key("mock", "system", conversation, backend.gen_params)
    record = ResponseCache(tmp_path / "records.jsonl").get_record(key)
    assert record is not None
    for field in (
        "model_id",
        "backend",
        "requested_params",
        "effective_params",
        "usage",
        "finish_reason",
        "response_id",
    ):
        assert field in record


def test_experimental_cache_context_separates_identical_hidden_condition_calls(tmp_path):
    calls = {"n": 0}

    def policy(system, conversation):
        calls["n"] += 1
        return 'ACTION: {"kind": "wait", "args": {}}'

    backend = MockBackend(policy=policy, cache=ResponseCache(tmp_path / "cache.jsonl"))
    conversation = [{"role": "user", "content": "byte-identical public state"}]
    backend.generate_batch_records(
        "system",
        [conversation, conversation],
        cache_contexts=[{"condition_id": "m0_c0"}, {"condition_id": "m1_c0"}],
    )
    assert calls["n"] == 2
    backend.generate_batch_records(
        "system",
        [conversation, conversation],
        cache_contexts=[{"condition_id": "m0_c0"}, {"condition_id": "m1_c0"}],
    )
    assert calls["n"] == 2


def test_fewshot_modes_differ():
    """Cue names a colour rule, mechanistic teaches testing, anti-cue varies colour."""
    from experiments.run_llm_eval import build_fewshot_prefix

    cue = build_fewshot_prefix(TaskFamily.AFFORDANCE, 3, mode="cue")[0]["content"]
    mech = build_fewshot_prefix(TaskFamily.AFFORDANCE, 3, mode="mechanistic")[0]["content"]
    anti = build_fewshot_prefix(TaskFamily.AFFORDANCE, 3, mode="anticue")[0]["content"]
    assert "opened the gate" in cue
    assert cue.lower().count("applying the red tool") >= 2
    assert "Testing the tools" in mech and "applying the red tool" not in mech.lower()
    assert anti.lower().count("applying the red tool") < cue.lower().count("applying the red tool")


def test_oracle_hint_levels_and_anonymization():
    from crucible.agents.prompting import build_label_map, oracle_hint

    spec = generate_task(TaskFamily.AFFORDANCE, seed=0, split=SplitLabel.TRAIN)
    env = CrucibleEnv(seed=spec.seed, config={"task_spec": spec})
    obs = env.reset()
    lm = build_label_map(obs)
    assert oracle_hint(spec, lm, None) == "" and oracle_hint(spec, lm, "none") == ""
    assert "applying a tool to the gate" in oracle_hint(spec, lm, "intervention")
    prop = oracle_hint(spec, lm, "property")
    assert "object_" in prop
    for oid in obs["objects"]:
        assert oid not in prop
    assert "opens it" in oracle_hint(spec, lm, "rule")


def test_cache_label_namespaces_quantization():
    """Quantization changes outputs but not gen_params, so it must namespace the
    cache file; otherwise 4-bit and full-precision runs share one cache and reuse
    each other's completions. Full-precision keeps the bare label (existing caches
    stay valid), and distinct quantizations get distinct cache files."""
    from experiments.run_llm_eval import _cache_label

    label = "qwen2.5-32b-instruct"
    assert _cache_label(label, None) == label
    assert _cache_label(label, "4bit") == f"{label}-4bit"
    # A quantized run never collides with the full-precision run or another mode.
    assert _cache_label(label, "4bit") != _cache_label(label, None)
    assert _cache_label(label, "4bit") != _cache_label(label, "8bit")

"""LLM agent: reasons over a public observation and emits one action per step.

Backend-agnostic (local transformers / Anthropic / OpenAI / mock) via
``crucible.agents.llm_backends``. Reads only public observations plus the
injected goal text. For throughput, the batched rollout harness
(``experiments/run_llm_eval.py``) drives many episodes at once using the same
prompting + parsing helpers; this class is the single-episode path used by the
standard runner and tests.
"""

from __future__ import annotations

import os

from crucible.actions import Action
from crucible.agents.base import Agent
from crucible.agents.llm_backends import ResponseCache, make_backend
from crucible.agents.prompting import (
    SYSTEM_PROMPT,
    anonymize_text,
    build_user_message,
    parse_action,
)
from crucible.utils.logging import get_logger

_log = get_logger(__name__)

_DEFAULT_GOAL = "Achieve the task goal."

# Default model per backend when none is given explicitly.
_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "mock": "mock",
}


def _default_backend() -> str:
    """Pick a backend from available credentials (else mock)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "mock"


class LLMAgent(Agent):
    """Single-episode LLM agent. The backend is created lazily on first act().

    With no explicit ``backend``, defaults to an API backend when the matching
    credential is present (Anthropic, then OpenAI), otherwise the mock backend —
    so ``LLMAgent()`` uses real credentials when they exist rather than silently
    returning WAIT actions.
    """

    name = "llm"

    def __init__(
        self,
        model_id: str | None = None,
        backend: str | None = None,
        *,
        cache_path: str | None = None,
        grid_size: int = 6,
        name: str | None = None,
        max_history: int = 9,
        **gen_params,
    ) -> None:
        backend = backend or _default_backend()
        self.model_id = model_id or _DEFAULT_MODELS.get(backend, "mock")
        self._backend_name = backend
        self._grid_size = grid_size
        self._max_history = max_history
        if name:
            self.name = name
        self._cache = ResponseCache(cache_path) if cache_path else None
        self._gen_params = gen_params
        self._backend = None
        self._history: list[dict] = []
        self._label_map: dict = {}

    @classmethod
    def is_available(cls) -> bool:
        """True if an API key is present (for the dormant API backends)."""
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    def _ensure_backend(self):
        if self._backend is None:
            self._backend = make_backend(
                self._backend_name, self.model_id, cache=self._cache, **self._gen_params
            )
        return self._backend

    def reset(self) -> None:
        self._history = []
        self._label_map = {}

    def act(self, obs: dict) -> Action:
        backend = self._ensure_backend()
        goal_text = obs.get("_goal_text", _DEFAULT_GOAL)
        user_msg, candidates, self._label_map = build_user_message(
            obs, goal_text, self._grid_size, label_map=self._label_map
        )
        self._history.append({"role": "user", "content": user_msg})
        raw = backend.generate_batch(SYSTEM_PROMPT, [self._history[-self._max_history :]])[0]
        self._history.append({"role": "assistant", "content": raw})
        return parse_action(raw, candidates)

    def observe_result(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        effects = info.get("effects", [])
        if effects:
            # Anonymize effect text too — effect strings embed real object IDs.
            anon = anonymize_text(str(effects), self._label_map)
            self._history.append({"role": "user", "content": f"Result of last action: {anon}"})

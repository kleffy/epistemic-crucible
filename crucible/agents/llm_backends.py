"""LLM backends + a response cache for the LLM agent.

Backends share one interface: ``generate_batch(system, conversations)`` returns
one completion per conversation. This lets a single agent step and the batched
rollout harness use the same code path. Greedy decoding + an on-disk cache make
runs reproducible and free to repeat.

Backends:
- ``mock``         — deterministic, no network/GPU (tests, CI).
- ``transformers`` — local Hugging Face model on GPU (the primary path).
- ``anthropic``    — Claude API (built but dormant until ANTHROPIC_API_KEY is set).
- ``openai``       — OpenAI / any OpenAI-compatible server (e.g. a local vLLM).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from crucible.utils.logging import get_logger

_log = get_logger(__name__)

Conversation = list[dict]  # list of {"role": "user"|"assistant", "content": str}


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


class ResponseCache:
    """Disk-backed prompt→response cache (one JSONL file per model)."""

    def __init__(self, path: str | pathlib.Path | None) -> None:
        self.path = pathlib.Path(path) if path else None
        self._mem: dict[str, str] = {}
        if self.path and self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec["response"]
            _log.info("loaded %d cached responses from %s", len(self._mem), self.path)

    @staticmethod
    def key(model_id: str, system: str, messages: Conversation, gen: dict) -> str:
        blob = json.dumps(
            {"model": model_id, "system": system, "messages": messages, "gen": gen},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        return self._mem.get(key)

    def put(self, key: str, response: str) -> None:
        if key in self._mem:
            return
        self._mem[key] = response
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps({"key": key, "response": response}) + "\n")


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class LLMBackend(ABC):
    model_id: str
    gen_params: dict

    def __init__(self, model_id: str, cache: ResponseCache | None = None, **gen_params) -> None:
        self.model_id = model_id
        self.cache = cache
        self.gen_params = {"max_new_tokens": 192, "temperature": 0.0, **gen_params}

    def generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        """Cache-aware batch generation. Subclasses implement ``_generate_batch``."""
        results: list[str | None] = [None] * len(conversations)
        misses, miss_idx = [], []
        for i, conv in enumerate(conversations):
            key = None
            if self.cache is not None:
                key = ResponseCache.key(self.model_id, system, conv, self.gen_params)
                hit = self.cache.get(key)
                if hit is not None:
                    results[i] = hit
                    continue
            misses.append(conv)
            miss_idx.append((i, key))
        if misses:
            generated = self._generate_batch(system, misses)
            for (i, key), text in zip(miss_idx, generated):
                results[i] = text
                if self.cache is not None and key is not None:
                    self.cache.put(key, text)
        return [r if r is not None else "" for r in results]

    @abstractmethod
    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        ...


# ---------------------------------------------------------------------------
# Mock backend (tests / CI)
# ---------------------------------------------------------------------------


class MockBackend(LLMBackend):
    """Deterministic backend. ``policy(system, conversation) -> str`` controls output.

    Default policy returns a WAIT action line.
    """

    def __init__(
        self,
        model_id: str = "mock",
        policy: Callable[[str, Conversation], str] | None = None,
        cache: ResponseCache | None = None,
        **gen_params,
    ) -> None:
        super().__init__(model_id, cache, **gen_params)
        self._policy = policy or (lambda system, conv: 'ACTION: {"kind": "wait", "args": {}}')

    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        return [self._policy(system, conv) for conv in conversations]


# ---------------------------------------------------------------------------
# Local transformers backend (GPU)
# ---------------------------------------------------------------------------

# Module-level cache so large weights load once and are shared across agents.
_HF_MODELS: dict[str, tuple] = {}


class TransformersBackend(LLMBackend):
    def __init__(
        self,
        model_id: str,
        cache: ResponseCache | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 32,
        quantization: str | None = None,
        **gen_params,
    ) -> None:
        super().__init__(model_id, cache, **gen_params)
        self.device = device
        self.batch_size = batch_size
        self._tok, self._model = _load_hf_model(model_id, device, dtype, quantization)

    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        import torch

        outputs: list[str] = []
        for start in range(0, len(conversations), self.batch_size):
            chunk = conversations[start: start + self.batch_size]
            prompts = [
                self._tok.apply_chat_template(
                    _merge_roles([{"role": "system", "content": system}, *conv]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for conv in chunk
            ]
            enc = self._tok(prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                gen = self._model.generate(
                    **enc,
                    max_new_tokens=self.gen_params["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self._tok.pad_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            outputs.extend(self._tok.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs


def _merge_roles(messages: list[dict]) -> list[dict]:
    """Normalise a conversation for strict chat templates (e.g. Mistral): keep an
    optional leading system message, ensure the first message after it is a user
    turn (drop leading assistant turns introduced by context windowing), and
    collapse consecutive same-role messages by concatenating their contents."""
    out: list[dict] = []
    i = 0
    if messages and messages[0]["role"] == "system":
        out.append(dict(messages[0]))
        i = 1
    while i < len(messages) and messages[i]["role"] != "user":
        i += 1  # must start with a user turn after the system message
    for m in messages[i:]:
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _load_hf_model(model_id: str, device: str, dtype: str, quantization: str | None = None):
    key = (model_id, quantization)
    if key in _HF_MODELS:
        return _HF_MODELS[key]
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from crucible.utils.seeding import seed_torch

    # Only one local model fits in GPU memory at a time; evict any previously
    # loaded model before loading a new one (the harness runs models serially).
    if _HF_MODELS:
        _HF_MODELS.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    seed_torch(0)
    _log.info("loading %s (%s, quant=%s) on %s ...", model_id, dtype, quantization, device)
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs: dict = {"device_map": device}
    if quantization == "4bit":
        # nf4 double-quant; verified on Blackwell (sm_120). Lets ~32B fit in 32 GB.
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    _HF_MODELS[key] = (tok, model)
    return tok, model


# ---------------------------------------------------------------------------
# API backends (dormant until credentials are provided)
# ---------------------------------------------------------------------------

_TRANSIENT_EXC = {
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "OverloadedError",
    "ServiceUnavailableError",
}


def _retry(fn: Callable[[], Any], *, retries: int = 8, base: float = 2.0, cap: float = 60.0) -> Any:
    """Call ``fn`` with exponential backoff on transient API errors (e.g. 429)."""
    import random
    import time

    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified below
            status = getattr(exc, "status_code", None)
            transient = type(exc).__name__ in _TRANSIENT_EXC or status in (429, 500, 502, 503, 529)
            if not transient or attempt == retries:
                raise
            delay = min(cap, base * (2 ** attempt)) + random.uniform(0, 1)
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    delay = max(delay, float(resp.headers.get("retry-after")))
                except (TypeError, ValueError, AttributeError):
                    pass
            _log.warning(
                "transient API error %s; retry %d/%d in %.1fs",
                type(exc).__name__, attempt + 1, retries, delay,
            )
            time.sleep(delay)


class AnthropicBackend(LLMBackend):
    """Claude API backend. Caches the static system prompt across calls."""

    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        import anthropic

        client = anthropic.Anthropic()

        def one(conv: Conversation) -> str:
            kwargs = dict(
                model=self.model_id,
                max_tokens=self.gen_params["max_new_tokens"],
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=conv,
            )
            temp = self.gen_params.get("temperature", 0.0)
            if temp is not None:
                kwargs["temperature"] = temp
            try:
                resp = _retry(lambda: client.messages.create(**kwargs))
            except anthropic.BadRequestError as exc:
                # Some models (e.g. Opus 4.8) deprecate temperature — drop and retry.
                if "temperature" in str(exc) and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    resp = _retry(lambda: client.messages.create(**kwargs))
                else:
                    raise
            return "".join(b.text for b in resp.content if b.type == "text")

        workers = int(self.gen_params.get("concurrency", 4))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, conversations))


class OpenAIBackend(LLMBackend):
    """OpenAI or any OpenAI-compatible server (set base_url for a local vLLM)."""

    def __init__(self, model_id: str, cache=None, base_url: str | None = None, **gen_params):
        super().__init__(model_id, cache, **gen_params)
        self._base_url = base_url

    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        import openai

        client = openai.OpenAI(base_url=self._base_url)
        # Reasoning models (gpt-5*, o1/o3/o4) reject temperature and consume
        # tokens on hidden reasoning, so give a generous completion budget and
        # use low reasoning effort to bound cost/latency.
        reasoning = bool(re.match(r"^(gpt-5|o1|o3|o4)", self.model_id))

        def one(conv: Conversation) -> str:
            kwargs: dict = {
                "model": self.model_id,
                "messages": [{"role": "system", "content": system}, *conv],
                "max_completion_tokens": max(self.gen_params["max_new_tokens"], 3000)
                if reasoning
                else self.gen_params["max_new_tokens"],
            }
            if reasoning:
                kwargs["reasoning_effort"] = self.gen_params.get("reasoning_effort", "low")
            else:
                kwargs["temperature"] = self.gen_params.get("temperature", 0.0)
            resp = _retry(lambda: client.chat.completions.create(**kwargs))
            return resp.choices[0].message.content or ""

        workers = int(self.gen_params.get("concurrency", 4))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, conversations))


_BACKENDS = {
    "mock": MockBackend,
    "transformers": TransformersBackend,
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
}


def make_backend(backend: str, model_id: str, **kwargs) -> LLMBackend:
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {list(_BACKENDS)}")
    return _BACKENDS[backend](model_id, **kwargs)
